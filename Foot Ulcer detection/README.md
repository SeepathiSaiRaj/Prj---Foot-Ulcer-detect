# Overview:

This project, CNN-Based Diagnostic System for Diabetic Foot Ulcer Analysis uses machine learning to analyze images uploaded by diabetic patients to detect and classify foot ulcers. Based on the severity of the ulcer, the system provides personalized medical suggestions and dietary recommendations to support healing and overall health. This approach helps in early intervention, better management of diabetes-related complications, and improved patient care.

# Features:

**Ulcer Classification Model**: Detects and classifies the severity of diabetic foot ulcers from uploaded images using deep learning.

**Personalized Medical Suggestions**: Provides treatment advice based on the ulcer's severity to guide patients on proper wound care and medical actions.

**Dietary Recommendations**: Offers tailored diet plans to help manage diabetes and support ulcer healing.

**Image Upload Interface**: Allows patients to upload foot images through a user-friendly web interface for instant analysis.


# Technologies Used:


Python

Flask (for UI)

OpenCV – For image preprocessing and enhancement

Matplotlib & Seaborn (for data visualization)

TensorFlow / Keras – For building and training the deep learning model for image classification


# Dataset:

The project utilizes a Diabetic Foot Ulcer dataset, which includes:

Ulcer Image: High-resolution photographs of patients’ feet capturing ulceration areas


# Methodology

**Data Collection** – Diabetic Foot Ulcer image dataset containing ulcer images.

**Data Preprocessing** – Applied image resizing, normalization, and augmentation techniques to improve model generalization and handle imbalanced classes.

**Feature Extraction** – Used Convolutional Neural Networks (CNNs) to automatically extract visual features from foot ulcer images for accurate classification.

**Model Selection & Training** – Implemented a CNN-based deep learning model and trained it on labeled image data to classify ulcers.

**Model Evaluation** – Evaluated the model using metrics like accuracy, precision, recall, F1-score, and confusion matrix to validate performance.

**Recommendation System** – Mapped classification results to specific medical suggestions and diet plans tailored to the severity of the ulcer.

**Deployment & User Interface** – Built a Flask-based web application allowing users to upload foot images, receive real-time predictions, and view medical and dietary recommendations.


# Usage:

Open the application in your browser.

Upload a clear image of the affected foot area.

Click Submit to detect the ulcer severity.

View personalized medical suggestions and dietary recommendations based on the prediction to support recovery and diabetes management.


# Project Outcomes:

Accurate Ulcer Classification – Successfully detects and classifies diabetic foot ulcers into severity levels using deep learning techniques.

Personalized Health Recommendations – Provides tailored medical advice and dietary plans to assist patients in managing ulcer conditions and diabetes effectively.

