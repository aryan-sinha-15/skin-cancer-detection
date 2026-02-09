# data_loader.py - Prepare data for AI
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    """
    Load images, resize them, and split into train/test
    """
    print("📂 Loading data...")
    
    # Read labels
    df = pd.read_csv('HAM10000_metadata.csv')
    
    # We'll use 3 common types for simplicity
    # nv = Benign mole, mel = Melanoma, bcc = Basal Cell Carcinoma
    df = df[df['dx'].isin(['nv', 'mel', 'bcc'])]
    
    images = []
    labels = []
    
    # Convert labels to numbers
    label_map = {'nv': 0, 'mel': 1, 'bcc': 2}
    
    # Load each image
    for idx, row in df.iterrows():
        img_id = row['image_id']
        
        # Try first folder, then second
        path1 = f'HAM10000_images_part_1/{img_id}.jpg'
        path2 = f'HAM10000_images_part_2/{img_id}.jpg'
        
        img_path = path1 if os.path.exists(path1) else path2
        
        if img_path and os.path.exists(img_path):
            # Read image
            img = cv2.imread(img_path)
            
            # Resize to standard size (224x224 pixels)
            img = cv2.resize(img, (224, 224))
            
            # Normalize pixel values to 0-1 range
            img = img / 255.0
            
            images.append(img)
            labels.append(label_map[row['dx']])
    
    print(f"✅ Loaded {len(images)} images")
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Split: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Testing set: {X_test.shape[0]} images")
    
    return X_train, X_test, y_train, y_test

# Test if this file works
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()