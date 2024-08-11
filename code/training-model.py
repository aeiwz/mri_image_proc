import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Activation, BatchNormalization, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from glob import glob
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px

# Set paths and image dimensions
data_path = './input'
img_width, img_height = 256, 256

# Prepare training and mask data
mask_files = glob(f"{data_path}/*/*_mask*")
images_train = [i.replace('_mask', '') for i in mask_files]

# Create DataFrame and split data
df = pd.DataFrame({'images_train': images_train, 'masks': mask_files})
df_train, df_test = train_test_split(df, test_size=0.10)
df_train, df_valid = train_test_split(df_train, test_size=0.05)

# Normalize function
def normalize_data(img, mask):
    img, mask = img / 255, mask / 255
    mask = (mask > 0.5).astype(float)
    return img, mask

# Data generator
def train_generator(data_frame, batch_size, augmentation_dict, target_size=(256, 256), seed=1):
    image_datagen = ImageDataGenerator(**augmentation_dict)
    mask_datagen = ImageDataGenerator(**augmentation_dict)

    image_generator = image_datagen.flow_from_dataframe(
        data_frame, x_col="images_train", class_mode=None, color_mode="rgb",
        target_size=target_size, batch_size=batch_size, seed=seed
    )
    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame, x_col="masks", class_mode=None, color_mode="grayscale",
        target_size=target_size, batch_size=batch_size, seed=seed
    )
    return ((normalize_data(img, mask) for img, mask in zip(image_generator, mask_generator)))

# U-Net model
def unet_model(input_shape=(256, 256, 3)):
    def encoder_block(inputs, filters):
        conv = Conv2D(filters, (3, 3), padding="same", activation="relu")(inputs)
        conv = Conv2D(filters, (3, 3), padding="same")(conv)
        conv = Activation("relu")(BatchNormalization(axis=3)(conv))
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool

    def decoder_block(inputs, conv, num_filters):
        conv_trans = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(inputs)
        x = concatenate([conv_trans, conv], axis=3)
        x = Activation("relu")(Conv2D(num_filters, (3, 3), padding="same")(x))
        x = Activation("relu")(BatchNormalization(axis=3)(Conv2D(num_filters, (3, 3), padding="same")(x)))
        return x

    inputs = Input(input_shape)
    conv1, pool1 = encoder_block(inputs, 64)
    conv2, pool2 = encoder_block(pool1, 128)
    conv3, pool3 = encoder_block(pool2, 256)
    conv4, pool4 = encoder_block(pool3, 512)

    b1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(1024, (3, 3), padding='same', activation='relu')(Conv2D(1024, (3, 3), padding='same')(pool4))))

    s5 = decoder_block(b1, conv4, 512)
    s6 = decoder_block(s5, conv3, 256)
    s7 = decoder_block(s6, conv2, 128)
    s8 = decoder_block(s7, conv1, 64)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(s8)
    return Model(inputs, outputs)

# Dice Coefficient and Loss
def dice_coefficient(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    return (2 * intersection + smooth) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

# Compile and train the model
model = unet_model()
model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_coefficient_loss, metrics=["binary_accuracy", dice_coefficient])

train_gen = train_generator(df_train, batch_size=32, augmentation_dict={
    "rotation_range": 0.25, "width_shift_range": 0.05, "height_shift_range": 0.05,
    "shear_range": 0.05, "zoom_range": 0.05, "horizontal_flip": True
})

model_checkpoint = ModelCheckpoint('unet_model_best.keras', verbose=1, save_best_only=True)

history = model.fit(train_gen, steps_per_epoch=len(df_train) // 32, epochs=100,
                    validation_data=train_generator(df_valid, 32, {}),
                    validation_steps=len(df_valid) // 32, callbacks=[model_checkpoint])

# Plot loss and accuracy graphs
plt.plot(history.history['loss'], 'r-', label='Train Loss')
plt.plot(history.history['val_loss'], 'b-', label='Validation Loss')
plt.title('Loss Graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['dice_coefficient'], 'r-', label='Train Dice Coefficient')
plt.plot(history.history['val_dice_coefficient'], 'b-', label='Validation Dice Coefficient')
plt.title('Dice Coefficient Graph')
plt.xlabel('Epochs')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.show()

# Evaluation
test_gen = train_generator(df_test, 32, {})
results = model.evaluate(test_gen, steps=len(df_test) // 32)
print(f'Test Loss: {results[0]}')
print(f'Test Accuracy: {results[1]}')
print(f'Test Dice Coefficient: {results[2]}')

# Generate predictions and save side-by-side comparisons
for i in range(10):
    idx = np.random.randint(0, len(df_test))
    img = cv2.imread(df_test['images_train'].iloc[idx])
    img = cv2.resize(img, (img_width, img_height)) / 255.0
    img = img[np.newaxis, :, :, :]
    pred_img = model.predict(img)

    fig = make_subplots(rows=1, cols=3, subplot_titles=["Original Image", "True Mask", "Predicted Mask"])
    fig.add_trace(px.imshow(img[0]).data[0], row=1, col=1)
    fig.add_trace(px.imshow(cv2.imread(df_test['masks'].iloc[idx])).data[0], row=1, col=2)
    fig.add_trace(px.imshow(pred_img[0] > 0.5).data[0], row=1, col=3)
    fig.write_html(f'./evaluate/{i}.html')
