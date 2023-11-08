# CAT - DOG IMAGE CLASSIFICATION PROJECT

This project is an implementation of an image classification model that can distinguish between cats and dogs. The model is built using deep learning techniques and utilizes the MobileNetV2 architecture for feature extraction. In this README, we will cover various aspects of the project, including the model architecture, data preprocessing, training process, and usage.

## Table of Contents

1. Introduction
2. Prerequisites
3. Data Preparation
4. Model Architecture
5. Training
6. Model Performance
7. Usage
8. Contributing

## 1. Introduction
The goal of this project is to create a machine learning model that can classify images as either cats or dogs. Image classification is a fundamental task in computer vision, and this project demonstrates the use of transfer learning, data augmentation, and model checkpointing to achieve accurate results.


## 2. Prerequisites
Before running the code, make sure you have the following prerequisites:

- Python version: 3.10.12 
- TensorFlow version: 2.14.0
- Keras version: 2.14.0
- Matplotlib version: 3.7.1
- NumPy version: 1.23.5
- Scikit-Learn version: 1.2.2

## 3. Data Preparation
Data for this project is loaded from CSV files containing image data and labels. Labels are encoded as 0 for 'dog' and 1 for 'cat'. The following files are required:

1. input.csv: Contains image data for training.
2. labels.csv: Contains corresponding labels for training.
3. input_test.csv: Contains image data for testing.
4. labels_test.csv: Contains corresponding labels for testing.

## 4. Model Architecture
We use the MobileNetV2 architecture as the backbone for our model. MobileNetV2 is a lightweight and efficient convolutional neural network architecture that has been pre-trained on a large dataset. We fine-tune this architecture for our binary classification task.

The custom top layers added to MobileNetV2 include:

- Global Average Pooling
- A fully connected layer with 64 units and ReLU activation
- The final output layer with a single unit and sigmoid activation for binary classification

The model is compiled with the Adam optimizer and binary cross-entropy loss function.

## 5. Training
Our model is trained using the fit method with data augmentation. We save the best model weights using the ModelCheckpoint callback, which continuously monitors training loss to ensure that the best weights are saved.

Parameters for Training
- Batch size: 32
- Number of epochs: 5

## 6. Model Performance
After training, the model is evaluated on the test data to assess its performance. The performance analysis of the modle is as follows : 


### 6.1 Performance during Training
| Epoch Number | Loss | Accuracy | 
| -------- | -------- | -------- |
| 1 | 0.4169 | 0.8005 | 
| 2 | 0.2586 | 0.8890 | 
| 3 | 0.2143 | 0.9050 | 
| 4 | 0.1827 | 0.9195 |
| 5 | 0.1604 | 0.9405 |



### 6.2 Performance on Test Data
| | Precision | Recall | F1-Score | Support |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| 0 | 0.89 | 0.73 | 0.81 | 200 |
| 1 | 0.82 | 0.94 | 0.90 | 200 |
| Accuracy | | | 0.85 | 400 |
| Macro Avg | 0.86 | 0.85 | 0.85 | 400 |
| Weighted Avg | 0.86 | 0.85 | 0.85 | 400 |


*NOTE - Labels are encoded as 0 for 'dog' and 1 for 'cat'.*

## 7. Usage
To train the model and evaluate it, you can follow these steps:

1. Prepare your data in CSV format, including training and testing datasets.
2. Load and preprocess your data as shown in the code.
3. Run the model training code provided in the script.
4. The trained model will be saved as a checkpoint file.
5. Evaluate the model on your testing dataset to assess its performance.
   
We can also use the model for making predictions on new images. Simply load the saved model checkpoint and use it to classify new images. Here is an example of how to make predictions using the model:

<pre>
# python code
    
# Load the model
from tensorflow.keras.models import load_model

model = load_model("Classification_Model.h5")

# Make predictions on a test image
y_pred = model.predict(new_image.reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5

if y_pred == 0:
    pred = 'dog'
else:
    pred = 'cat'

print("Our model says it is a:", pred)
</pre>

## 7. Files Structure
The project's structure is organized as follows:

- `data/`       :         Contains datasets used for training and testing the model (in 7Zip archived format).
- `src/`         :        Source code and Google Collab Notebook
- `model/`     :       Contains the trained model : Classification_Model.h5 
- `README.md`     :       This README file

## 8. Contributing
We welcome contributions to this project. If you have suggestions for improvements, bug reports, or want to extend the functionality, please feel free to open an issue or submit a pull request.


Thank you for considering this project as a demonstration of my skills and capabilities in deep learning and computer vision. If you have any questions or would like to see a live demonstration, please don't hesitate to contact me.

