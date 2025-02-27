# Water Segmentation Using Deep Learning

## Project Overview
This project implements water segmentation using deep learning models. The goal is to segment water bodies from satellite imagery using two different segmentation architectures: U-Net and DeepLabV3. The models are trained and evaluated using performance metrics such as IoU (Intersection over Union), Precision, Recall, and F1-Score.

## Dataset
- **Input Data**: Satellite images with **12 spectral channels**.
- **Labels**: Binary masks indicating water bodies.
- **Image Size**: 128x128 pixels.
- **Dataset Split**: 80% training, 20% validation.

## Preprocessing
1. Load images and corresponding binary masks.
2. Resize images and masks to 128x128.
3. Normalize image pixel values by dividing by 10,000.
4. Expand the dimension of masks to match expected input format.
5. Convert data into PyTorch tensors and create `DataLoader` for training and validation.

## Model Architectures
### 1. U-Net (Encoder: ResNet-34)
- Pretrained on ImageNet.
- **Input channels**: 12 (to match dataset).
- **Output channels**: 1 (binary mask prediction).

### 2. DeepLabV3 (Encoder: ResNet-50)
- Pretrained on ImageNet.
- **Input channels**: 12.
- **Output channels**: 1.

## Training Process
- Loss function: Binary Cross Entropy with Logits (`BCEWithLogitsLoss`).
- Optimizer: Adam with a learning rate of 0.001.
- Batch size: 16.
- Number of epochs: 50.
- Training loop includes:
  1. Forward pass.
  2. Compute loss.
  3. Backpropagation.
  4. Update weights.

## Results
The models were evaluated using IoU, Precision, Recall, and F1-Score. Below are the results:

### **Custom U-Net Model (12 Channels)**
- **IoU**: 0.6645
- **Precision**: 0.9382
- **Recall**: 0.6949
- **F1-Score**: 0.7984

### **Transfer Learning Models**
#### **U-Net (ResNet-34 Encoder)**
- **IoU**: 0.6708
- **Precision**: 0.9500
- **Recall**: 0.6954
- **F1-Score**: 0.8030

#### **DeepLabV3 (ResNet-50 Encoder)**
- **IoU**: 0.8138
- **Precision**: 0.9247
- **Recall**: 0.8715
- **F1-Score**: 0.8973

## Visualization
The model's performance is visualized in two ways:
1. **Training Curves**: Loss curves for both models over epochs.
2. **Segmentation Results**: Comparison of input images, ground truth masks, and predicted masks for validation samples.

## Conclusion
- The DeepLabV3 model with ResNet-50 encoder performed best, achieving the highest IoU and F1-score.
- U-Net with ResNet-34 also performed well but slightly lower than DeepLabV3.
- The custom-built U-Net (12-channel) was comparable but benefited less from transfer learning.
- Future improvements can involve experimenting with other architectures, fine-tuning hyperparameters, and incorporating additional preprocessing techniques.

## Requirements
- Python 3.11
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- ImageIO
- Segmentation Models PyTorch (smp)

## Running the Code
1. Install dependencies using `pip install -r requirements.txt`.
2. Place dataset images and labels in the specified directories.
3. Run the notebook or Python script to train and evaluate the models.

