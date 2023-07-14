# APPLICATION OF DIFFERENT CNN ON AID DATASET
The application of various Convolutional Neural Networks on the Aid dataset for satellite image classification, featuring custom layers, transfer learning with EfficientNetB3, MobileNetV2, and ResNet50, and multiple experiments to enhance accuracy.


# Description
This GitHub repository showcases an in-depth exploration of satellite image classification using various Convolutional Neural Network (CNN) architectures. The project focuses on the Application of Different Convolution Neural Networks on an Aid dataset, with the objective of achieving accurate classification results.

Key Features:

* **Custom Layers**: One notable aspect of this project is the implementation of custom layers from scratch. These custom layers are designed to enhance the CNN models' performance in satellite image classification tasks.

* **Transfer Learning**: The repository also demonstrates the utilization of transfer learning, a technique where pre-trained models are employed as a starting point for satellite image classification. Specifically, three different pre-trained models were employed: EfficientNetB3, MobileNetV2, and ResNet50.

* **Experimental Approach**: To improve the accuracy of the classification models, two separate experiments were conducted for each of the three aforementioned pre-trained models. Each experiment is presented as a separate file, providing detailed insights into the specific modifications and enhancements applied to the models.

# Set up
## Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

## Install requirements
```bash
pip install -r requirements.txt
```

## Run the Notebook to set up the data for training
Go to Notebook.ipynb and run all cells for splitting the dataset and performing data augmentation.

## Train the model
```bash
# Run the training script to train all the models
./train_all.fish
# or train each model separately
pyhon mymodel.py
```

## Evaluate the model
You can evaluate the models by running the evaluation cells in the Notebook.ipynb for the model you want to evaluate.


# Dataset
## Download the dataset
You can download the dataset from [here](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets) 
