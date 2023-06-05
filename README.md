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