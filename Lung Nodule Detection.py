#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pydicom
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from PIL import Image
from skimage import morphology
import pandas as pd

# Define the main directory containing DICOM files
main_directory = 'C:/Users/vaish/Downloads/manifest-1600709154662/training/'

# Directory to save processed data
save_directory = 'C:/Users/vaish/Downloads/manifest-1600709154662/processed_data1/'

# Function to process a single DICOM file and extract features
def process_dicom(dicom_file_path):
    try:
        # Load the DICOM image
        dcm_image = pydicom.read_file(dicom_file_path)
        image_array1 = dcm_image.pixel_array.astype('float32')

        # Apply manual thresholding
        thresh_value = 604
        binary_image = image_array1 < thresh_value

        # Clear the border
        cleared_image = clear_border(binary_image)

        # Define a kernel for closing operation
        kernel = np.ones((3, 3), dtype=np.int64)

        # Create a disk-shaped structuring element with a radius of 2 pixels
        selem = morphology.disk(2)

        # Apply binary erosion
        eroded_image = morphology.binary_erosion(cleared_image, selem)

        # Perform closing operation
        closed_image = ndimage.binary_closing(eroded_image, structure=kernel)

        # Label the connected components
        labeled_image = label(closed_image)

        # Get the region properties for each connected component
        regions = regionprops(labeled_image)

        # Initialize lists for features and labels
        features = []
        labels = []

        # Define thresholds for classification
        area_threshold = 100
        size_threshold = 10
        eccentricity_threshold = 0.7

        # Extract features from each region
        for region in regions:
            area = region.area
            size = region.equivalent_diameter
            eccentricity = region.eccentricity

            # Determine label based on region properties
            if area > area_threshold and size > size_threshold and eccentricity < eccentricity_threshold:
                label_value = 1  # Cancerous
            else:
                label_value = 0  # Non-cancerous

            # Append features and labels
            features.append([dicom_file_path, area, size, eccentricity])  # Include file name here
            labels.append(label_value)

        return features, labels

    except Exception as e:
        print(f"Error processing {dicom_file_path}: {e}")
        return None, None

# Function to traverse the directory and process all DICOM files
def process_all_dicom_files(main_directory, save_directory):
    # Create a directory to save processed data
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Initialize lists to store all features and labels
    all_features = []
    all_labels = []

    # Traverse the directory
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            if file.lower().endswith('.dcm'):
                dicom_file_path = os.path.join(root, file)

                try:
                    # Process the DICOM file
                    features, labels = process_dicom(dicom_file_path)

                    if features is not None and labels is not None:
                        all_features.extend(features)
                        all_labels.extend(labels)

                        # Optionally, display the processed images
                        # display_processed_images(dicom_file_path, b_img, non_cancerous_nodules_image, cancerous_nodules_image)

                except Exception as e:
                    print(f"Error processing {dicom_file_path}: {e}")

    # Save all features and labels to CSV files
    features_df = pd.DataFrame(all_features, columns=['File', 'Area', 'Size', 'Eccentricity'])  # Include 'File' column
    labels_df = pd.DataFrame(all_labels, columns=['Label'])

    features_df.to_csv(os.path.join(save_directory, 'features1.csv'), index=False)
    labels_df.to_csv(os.path.join(save_directory, 'labels1.csv'), index=False)

# Execute the processing function
process_all_dicom_files(main_directory, save_directory)

import pandas as pd
from sklearn.model_selection import train_test_split

# Load features and labels from CSV files
features_df = pd.read_csv('C:/Users/vaish/Downloads/manifest-1600709154662/processed_data1/features1.csv')
labels_df = pd.read_csv('C:/Users/vaish/Downloads/manifest-1600709154662/processed_data1/labels1.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_df[['Area', 'Size', 'Eccentricity']],
                                                    labels_df['Label'],
                                                    test_size=0.2,
                                                    random_state=42)

# Print shapes to verify
print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model architecture
model = Sequential([
    Dense(64, input_shape=(3,), activation='relu'),  # Input layer with 3 features
    Dense(32, activation='relu'),                    # Hidden layer with 32 neurons
    Dense(1, activation='sigmoid')                   # Output layer (binary classification)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary (optional)
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")


import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




