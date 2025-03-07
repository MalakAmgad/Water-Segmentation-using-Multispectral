# Water Segmentation Using Deep Learning

## Deployment Overview
![Deployment Overview](https://github.com/MalakAmgad/Water-Segmentation-using-Multispectral/blob/main/water_segmentation/static/Screenshot%20(904).png)
![Deployment Overview](https://github.com/MalakAmgad/Water-Segmentation-using-Multispectral/blob/main/water_segmentation/static/Screenshot%20(901).png)

## Project Overview
This project implements water segmentation using deep learning models. The goal is to segment water bodies from satellite imagery using two different segmentation architectures: U-Net and DeepLabV3. The models are trained and evaluated using performance metrics such as IoU (Intersection over Union), Precision, Recall, and F1-Score.

## Project Structure
```
/water-segmentation-app
│── model/
    │── deeplabv3_water_segmentation.pth                  # Trained model files
│── uploads/                 # Folder for storing uploaded images
│── templates/               # HTML templates for Flask app
│   │── index.html           # Web interface
│── static/
     │── styles.css                # CSS and JS files
│── images/                  # Example images (optional)
│── app.py                   # Flask application
│── requirements.txt         # Dependencies
│── README.md                # Documentation
```

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
- ![accuracy](https://github.com/MalakAmgad/Water-Segmentation-using-Multispectral/blob/main/water_segmentation/static/826beccd-0bc7-4146-af38-7ea0ca987bce.png)
- ![loss](https://github.com/MalakAmgad/Water-Segmentation-using-Multispectral/blob/main/water_segmentation/static/47ca39ea-9e40-475d-8eef-21d7b132d0d9.png)
- ![prediction](https://github.com/MalakAmgad/Water-Segmentation-using-Multispectral/blob/main/water_segmentation/static/a4babb85-4b3b-4ab4-ae30-8f915dc5043f.png)


### **Transfer Learning Models**
#### **U-Net (ResNet-34 Encoder)**
- **IoU**: 0.6708
- **Precision**: 0.9500
- **Recall**: 0.6954
- **F1-Score**: 0.8030
- 
- ![prediction](https://github.com/MalakAmgad/Water-Segmentation-using-Multispectral/blob/main/water_segmentation/static/49b8ad06-0224-4a26-9aeb-3120ea787199.png)
- ![prediction](https://github.com/MalakAmgad/Water-Segmentation-using-Multispectral/blob/main/water_segmentation/static/733d4210-27b4-46c9-bbc8-0872173d5986.png)

#### **DeepLabV3 (ResNet-50 Encoder)**
- **IoU**: 0.8138
- **Precision**: 0.9247
- **Recall**: 0.8715
- **F1-Score**: 0.8973

- ![prediction](https://github.com/MalakAmgad/Water-Segmentation-using-Multispectral/blob/main/water_segmentation/static/6ffda21b-a982-416b-8833-298fe1d85ef9.png)
- ![prediction](https://github.com/MalakAmgad/Water-Segmentation-using-Multispectral/blob/main/water_segmentation/static/cf3f1542-6f2a-4c44-8c15-9de2cb15b3ae.png)

## Visualization
The model's performance is visualized in two ways:
1. **Training Curves**: Loss curves for both models over epochs.
2. **Segmentation Results**: Comparison of input images, ground truth masks, and predicted masks for validation samples.

## Conclusion
- The DeepLabV3 model with ResNet-50 encoder performed best, achieving the highest IoU and F1-score.
- U-Net with ResNet-34 also performed well but slightly lower than DeepLabV3.
- The custom-built U-Net (12-channel) was comparable but benefited less from transfer learning.
- Future improvements can involve experimenting with other architectures, fine-tuning hyperparameters, and incorporating additional preprocessing techniques.

## Deployment
This project is deployed using **Flask**. You can run the web app locally or deploy it on a cloud platform like **AWS, Heroku, or Google Cloud**.

### Running the Web App Locally:
```bash
pip install -r requirements.txt
python app.py
```
Access the app at `http://127.0.0.1:5000/`.

### Running Model Inference:
```bash
python inference.py --image test.tif --model deeplabv3
```

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

