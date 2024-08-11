

# Brain MRI Segmentation Using U-Net

This project implements a U-Net model for segmenting Brain MRI images. The model is trained using a dataset of MRI images and their corresponding masks, and it uses data augmentation to improve generalization.

---


## Dependencies

To run this project, you'll need to install the following dependencies:

- TensorFlow
- Keras
- scikit-image
- OpenCV
- scikit-learn
- matplotlib
- pandas
- plotly

You can install these dependencies using pip:

```bash
pip install tensorflow scikit-image opencv-python-headless scikit-learn matplotlib pandas plotly
```

## Dataset

The dataset used in this project consists of Brain MRI images and their corresponding masks. The dataset should be organized in the following structure:

```
input/
│
├── image_1.png
├── image_1_mask.png
├── image_2.png
├── image_2_mask.png
└── ...
```

## Model Architecture

This project uses a U-Net architecture, which is widely used for image segmentation tasks. The U-Net model is composed of an encoder (downsampling) path and a decoder (upsampling) path.

### Encoder
- The encoder path consists of a series of convolutional layers followed by max-pooling layers.

### Decoder
- The decoder path consists of convolutional layers with upsampling layers, concatenated with the corresponding layers from the encoder path.

### Output
- The output layer is a single convolutional layer with a sigmoid activation function to generate the segmentation mask.

## Training

The model is trained using the Dice Coefficient Loss, which is particularly effective for image segmentation tasks. The Adam optimizer is used for optimization.

### Training Parameters:
- Epochs: 1000
- Batch size: 32
- Learning rate: 1e-4

The model is saved to `unet_model_best.keras` after each epoch if it achieves a better validation score.

## Evaluation

The trained model is evaluated on a test set, and the performance is measured using the following metrics:
- Dice Coefficient
- Intersection over Union (IoU)
- Binary Accuracy

### Evaluation Results:
- Test Loss: *Displayed on evaluation*
- Test Dice Coefficient: *Displayed on evaluation*
- Test IoU: *Displayed on evaluation*

## Results

### Performance
![Performance](https://github.com/aeiwz/mri_image_proc/blob/982f3197534c505ba362f1070ed2e9543a33d95f/src/img/Performance%20model.png)


