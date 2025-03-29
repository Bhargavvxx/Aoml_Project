# Skin Cancer Classification using CNN & XGBoost

## Problem Statement
Skin cancer is one of the most common types of cancer, and early detection is critical for successful treatment. Traditional diagnostic methods require expert dermatologists and can be time-consuming. This project aims to develop an AI-driven solution to classify different types of skin lesions using **Convolutional Neural Networks (CNN) for feature extraction** and **XGBoost for classification**.

## Dataset
- **Source**: [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
- **Description**: The dataset contains **10,015 dermatoscopic images** labeled into **7 different classes** of skin lesions:
  1. **Actinic keratoses** (akiec)
  2. **Basal cell carcinoma** (bcc)
  3. **Benign keratosis-like lesions** (bkl)
  4. **Dermatofibroma** (df)
  5. **Melanoma** (mel)
  6. **Melanocytic nevi** (nv)
  7. **Vascular lesions** (vasc)

## Solution Approach
1. **Data Preprocessing**:
   - Images resized to **224x224**.
   - Normalization applied (**pixel values scaled between 0 and 1**).
   - Data augmentation used to improve model generalization.

2. **Feature Extraction using CNN**:
   - A custom **Convolutional Neural Network (CNN)** is trained on the dataset.
   - Features are extracted from the last convolutional layer.

3. **Classification using XGBoost**:
   - The extracted CNN features are used as input to an **XGBoost classifier**.
   - XGBoost is trained to classify images into **one of the 7 categories**.

## Model Training
- **CNN Training**:
  - Optimizer: **Adam**
  - Loss Function: **Sparse Categorical Crossentropy**
  - Epochs: **15**
  - Class Weights applied to handle imbalanced dataset

- **XGBoost Training**:
  - **700 estimators**, **max depth = 20**, **learning rate = 0.03**
  - Class weights used for balanced learning

 **Streamlit Web Application**
A **Streamlit-based UI** is developed to allow users to:
1. **Upload an image**
2. **Run classification** using the trained models
3. **View predicted class and confidence scores**

 Installation & Usage
 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

 2️⃣ Run Streamlit App
```bash
streamlit run app.py
```

 3️⃣ Upload an image and get classification results!

 Future Improvements
- Fine-tuning CNN on HAM10000 dataset
- Implementing Explainable AI (Grad-CAM visualization)
- Adding additional ensemble learning techniques

 Contributors
Bhargav Patil - J014
Rahul Yanamandra - J047
Parth Khandagale - J045




