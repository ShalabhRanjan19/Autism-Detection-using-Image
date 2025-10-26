# Autism Detection using Image

## Overview

This project creates a machine learning system to detect autism through image analysis. It examines facial features and expressions using convolutional neural networks (CNNs) to identify autism-related cues. The goal is to provide a user-friendly tool for early and accurate autism detection, supporting timely interventions.

## Features

- **Binary Classification**: Distinguishes between autistic and non-autistic individuals based on facial images
- **CNN-based Analysis**: Utilizes deep learning techniques for feature extraction and pattern recognition
- **Early Detection**: Designed to support early intervention through timely autism spectrum disorder identification
- **User-friendly Interface**: Easy-to-use system for healthcare professionals and researchers
- **Image Processing**: Automated preprocessing and analysis of facial images

## Dataset

The project uses the Autism Image Dataset available on Kaggle:

- **Dataset Source**: [Autism Image Data on Kaggle](https://www.kaggle.com/datasets/cihan063/autism-image-data)
- **Pre-trained Model**: [H5 Model File](https://drive.google.com/file/d/1dnmjQRBYE2TAk1JPMlTxJrKaDQlMtq_L/view?usp=drive_link)
- **Classes**: Binary classification (Autistic: 1, Non-autistic: 0)
- **Data Type**: Facial images of children and individuals

## Requirements

```
Python 3.7+
TensorFlow 2.x
Keras
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
Pillow
Flask (for web interface)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ShalabhRanjan19/Autism-Detection-using-Image.git
cd Autism-Detection-using-Image
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in the `data/` directory

4. Download the pre-trained model (H5 file) from the provided Google Drive link

## Usage

### Training the Model

```bash
python train_model.py
```

### Making Predictions

```bash
python predict.py --image_path path/to/your/image.jpg
```

### Running the Web Application

```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

## Model Architecture

The system employs a Convolutional Neural Network (CNN) architecture optimized for facial feature analysis:

- **Input Layer**: Accepts preprocessed facial images
- **Convolutional Layers**: Multiple conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling for dimensionality reduction
- **Fully Connected Layers**: Dense layers for final classification
- **Output Layer**: Single neuron with sigmoid activation for binary classification

### Preprocessing Steps

1. Image resizing to standard dimensions
2. Normalization of pixel values
3. Face detection and cropping
4. Data augmentation for training robustness

## Results

The model achieves the following performance metrics:

- **Accuracy**: 81.08%
- **Precision**: 88.23 %
- **Recall**: 84.50704 %
- **F1-Score**: 86.28 %


## File Structure

```
Autism-Detection-using-Image/
├── data/
│   ├── train/
│   └── test/
├── models/
│   └── trained_model.h5
├── src/
│   ├── train_model.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── utils.py
├── notebooks/
│   └── data_analysis.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions or suggestions, please contact:
- **Author**: Shalabh Ranjan
- **GitHub**: [@ShalabhRanjan19](https://github.com/ShalabhRanjan19)
