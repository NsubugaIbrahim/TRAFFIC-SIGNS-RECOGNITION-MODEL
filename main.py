# Traffic Signs Recognition using CNN and Keras
# ================================================

# Set backend before importing keras
import os
os.environ['KERAS_BACKEND'] = 'jax'

# Data handling and preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Image processing
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Deep Learning
import keras
from keras import layers, models
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Data utilities
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("=" * 50)
print("All imports successful!")
print("=" * 50)
print(f"Keras version: {keras.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print("=" * 50)
print("\nYour environment is ready for training!")
print("Dataset structure:")
print("  - Train/: Contains image subdirectories (0-42)")
print("  - Test/: Contains test images")
print("  - Meta.csv, Train.csv, Test.csv: Label information")
print("=" * 50)
