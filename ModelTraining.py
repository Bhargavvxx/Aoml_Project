import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import kagglehub

# Download dataset
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print("Path to dataset files:", path)

# Define dataset paths
metadata_path = os.path.join(path, "HAM10000_metadata.csv")
images_path_1 = os.path.join(path, "HAM10000_images_part_1")
images_path_2 = os.path.join(path, "HAM10000_images_part_2")

# Load metadata
metadata = pd.read_csv(metadata_path)
subset = metadata.sample(frac=0.30, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

def load_images(file_paths, img_size=(224, 224)):
    images = []
    for file in file_paths:
        img_path = os.path.join(images_path_1, file)
        if not os.path.exists(img_path):
            img_path = os.path.join(images_path_2, file)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        images.append(img_array)
    return np.array(images)

print("Loading images...")
X_images = load_images(subset["image_id"] + ".jpg")
X_images = datagen.flow(X_images, batch_size=len(X_images), shuffle=False)._getitem_(0)

# Define Custom CNN
def build_custom_cnn(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

num_classes = len(set(subset['dx']))
input_shape = (224, 224, 3)

model = build_custom_cnn(input_shape, num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Compute class weights
class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(subset['dx']), y=subset['dx'])
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train CNN
print("Training Custom CNN...")
model.fit(X_images, pd.factorize(subset["dx"])[0], epochs=15, batch_size=32, class_weight=class_weight_dict)

# Extract Features
X_features = model.predict(X_images)

# Prepare labels
y = pd.factorize(subset["dx"])[0]
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Train XGBoost Classifier
print("Training XGBoost Classifier...")
clf = XGBClassifier(
    n_estimators=700,
    max_depth=20,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train, sample_weight=[class_weight_dict[i] for i in y_train])

# Evaluate Model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save Model
print("Saving model...")
joblib.dump(clf, "skin_cancer_xgb_model_vv.pkl")
print("Training completed successfully!")