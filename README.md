# Motion-Based Fall vs Non-Fall Detection

A machine learning-based system for detecting human falls from video sequences. The project uses **Motion History Images (MHI)** to convert temporal movement information from video frames into a compact motion representation, which is then classified into **fall** and **non-fall** activities using traditional machine learning algorithms.

## Overview

Fall detection is useful in applications such as elderly care, assisted living, and video-based safety monitoring. Instead of processing complete video sequences directly with a deep learning model, this project focuses on extracting motion information from consecutive frames using **Motion History Images (MHI)** and using machine learning models for classification.

The system processes video frames, captures recent motion patterns, converts them into MHI representations, and uses these representations as input for classification.

## Approach

The project follows these main steps:

1. **Input Video Frames**
   - Video/image sequences are organized into fall and non-fall classes.
   - Consecutive frames are processed in temporal order.

2. **Frame Preprocessing**
   - Frames are resized to **64 × 64** pixels.
   - Frames are converted from BGR to grayscale.
   - Histogram equalization is applied to improve the representation of image intensity.

3. **Motion History Image Generation**
   - Differences between consecutive frames are calculated using OpenCV.
   - A threshold is applied to identify areas containing motion.
   - Motion information is accumulated over a temporal window of **15 frames**.
   - Recent motion is given greater importance in the resulting MHI.

4. **Feature Preparation**
   - The generated MHI is converted into a numerical feature vector by flattening the image representation.

5. **Machine Learning Classification**
   - The extracted features are used to train and evaluate multiple classification algorithms.
   - The implemented models are:
     - Support Vector Machine (SVM)
     - Decision Tree
     - Random Forest
     - k-Nearest Neighbors (k-NN)
     - Logistic Regression

6. **Model Evaluation**
   - The dataset is divided into training and testing sets using an **80:20 split**.
   - Models are evaluated using:
     - Accuracy
     - F1-score
     - Precision
     - Recall
     - Classification Report
     - Confusion Matrix

## Technologies Used

- **Python**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**

