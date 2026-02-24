# Example code snippets for Traffic Signs Recognition

import numpy as np
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
import keras
from keras import layers, models

# ============================================
# 1. Load and Explore Dataset
# ============================================

def load_images_from_folder(folder_path, target_size=(32, 32)):
    """Load all images from a folder."""
    images = []
    labels = []
    
    for class_dir in sorted(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.resize(img, target_size)
                images.append(img)
                labels.append(int(class_dir))
    
    return np.array(images), np.array(labels)

# ============================================
# 2. Build CNN Model
# ============================================

def build_cnn_model(input_shape=(32, 32, 3), num_classes=43):
    """Build a CNN model for traffic sign classification."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# ============================================
# 3. Preprocess Images
# ============================================

def preprocess_images(images):
    """Normalize images to [0, 1] range."""
    return images.astype('float32') / 255.0

# ============================================
# 4. Usage Example (uncomment to run)
# ============================================

# if __name__ == "__main__":
#     # Load training data
#     X_train, y_train = load_images_from_folder('Train/')
#     X_train = preprocess_images(X_train)
#     
#     # Split into train and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.2, random_state=42
#     )
#     
#     # Build model
#     model = build_cnn_model(num_classes=43)
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=0.001),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     
#     # Train model
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=30,
#         batch_size=64,
#         verbose=1
#     )
#     
#     # Save model
#     model.save('traffic_signs_model.h5')
#     print("Model training completed and saved!")
