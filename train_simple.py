# train_simple.py - SIMPLE WORKING VERSION
import pandas as pd
import numpy as np
import cv2
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("🤖 Training Skin Cancer AI Model...\n")

# 1. Read the CSV file
df = pd.read_csv('HAM10000_metadata.csv')
print(f"Dataset has {len(df)} total images")

# 2. Focus on 2 main types (simpler)
df = df[df['dx'].isin(['nv', 'mel'])]  # nv=Benign, mel=Melanoma
print(f"Using {len(df)} images (Benign vs Melanoma)")

# 3. Load 100 images (fast training)
images = []
labels = []
count = 0

for idx, row in df.iterrows():
    if count >= 100:
        break
    
    img_id = row['image_id']
    
    # Check both folders
    path1 = f'HAM10000_images_part_1/{img_id}.jpg'
    path2 = f'HAM10000_images_part_2/{img_id}.jpg'
    
    img_path = path1 if os.path.exists(path1) else path2
    
    if os.path.exists(img_path):
        # Load and resize image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))  # Small size for speed
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale
        img = img.flatten() / 255.0  # Normalize to 0-1
        
        images.append(img)
        labels.append(0 if row['dx'] == 'nv' else 1)  # 0=benign, 1=melanoma
        count += 1

print(f"✅ Loaded {len(images)} images")

# 4. Prepare data
X = np.array(images)
y = np.array(labels)

# 5. Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} images")
print(f"Testing on {len(X_test)} images")

# 6. Train model
print("\n🧠 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 7. Check accuracy
train_acc = model.score(X_train, y_train) * 100
test_acc = model.score(X_test, y_test) * 100

print(f"\n📊 RESULTS:")
print(f"Training Accuracy: {train_acc:.1f}%")
print(f"Testing Accuracy: {test_acc:.1f}%")

# 8. Save model
joblib.dump(model, 'skin_cancer_model.joblib')
print("\n💾 Model saved as 'skin_cancer_model.joblib'")
print("🎯 READY FOR WEB APP!")