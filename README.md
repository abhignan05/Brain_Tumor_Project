# 🧠 Automated Brain Tumor Detection using Hybrid Machine Learning Models

> **An AI-powered MRI analysis system for automated brain tumor detection, classification, tumor size estimation, and risk assessment.**

**Project Type:** AI/ML + Healthcare
**Technology:** Python, Flask, TensorFlow, MobileNet, SVM, OpenCV
**Batch:** 72
**Application:** Web-based MRI Analysis System

---

## 🌐 Live Demo

**Live Application:**
https://brain-tumor-project-8e69.onrender.com

> **Note:** The application is deployed on Render. If the service has been inactive for some time, the first request may take around **1–2 minutes** while the application starts. Once the service is active, subsequent requests should load normally.

---

## 📌 Project Overview

Brain tumors are serious neurological conditions where early and accurate detection can play an important role in diagnosis and treatment planning. Conventional MRI analysis requires experienced medical professionals to examine scans carefully, which can be time-consuming and may introduce variability in interpretation.

This project presents an **automated brain tumor detection system** that analyzes MRI images using a hybrid Artificial Intelligence approach. The system combines **MobileNet-based deep feature extraction** with a **Support Vector Machine (SVM) classifier** to identify tumor patterns from MRI scans.

The system further performs post-processing to provide additional information such as **tumor presence, tumor type, estimated tumor size, confidence level, and risk indication**.

The application provides a simple web interface through which users can enter patient information, upload an MRI image, and obtain an AI-generated analysis report.

> **Important:** This application is intended as an AI-based decision-support and educational research system. It is not a replacement for professional medical diagnosis.

---

# 🎯 Objectives

The major objectives of the project are:

* Automate brain tumor detection from MRI images.
* Reduce the time required for preliminary MRI analysis.
* Extract meaningful deep features from MRI scans using MobileNet.
* Classify MRI images using an SVM model.
* Identify tumor presence and tumor type.
* Estimate the approximate tumor size from the MRI image.
* Provide a risk-level indication based on the analysis.
* Generate an understandable result for healthcare professionals.
* Provide a user-friendly web interface for MRI analysis.
* Demonstrate the application of hybrid AI techniques in healthcare.

---

# 🚨 Problem Statement

Manual analysis of MRI scans requires specialized medical expertise and can be time-consuming, particularly when large numbers of scans need to be examined.

Traditional image-classification approaches may also struggle to capture complex visual patterns present in MRI images.

Therefore, there is a need for an automated system that can:

**MRI Image → Process Image → Extract Features → Classify → Analyze → Generate Result**

The proposed system addresses this requirement using a combination of deep learning and machine learning techniques.

---

# 💡 Proposed Solution

The proposed solution uses a **hybrid machine learning pipeline**.

Instead of using a single classification algorithm, the system combines:

### MobileNet

A lightweight pre-trained convolutional neural network used to extract meaningful visual features from MRI images.

### SVM

A Support Vector Machine classifier that uses the extracted features to perform the final classification.

This combination allows the project to benefit from the representation-learning capability of deep learning while using SVM for classification.

---

# 🔄 System Workflow

```text
                MRI IMAGE
                    │
                    ▼
          ┌──────────────────┐
          │ Image Upload     │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Preprocessing    │
          │ Resize/Normalize │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │    MobileNet     │
          │ Feature Extractor│
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │       SVM        │
          │   Classification │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ Post Processing  │
          │ Segmentation     │
          │ Size Estimation  │
          │ Risk Analysis    │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  Result Report   │
          └──────────────────┘
```

---

# 🧩 Main Modules

## 1. Data Preprocessing

The uploaded MRI image is prepared before being passed to the machine learning pipeline.

The preprocessing stage handles operations such as:

* Image loading
* Image resizing
* Image normalization
* Image format handling
* Preparation of the image for feature extraction

Proper preprocessing ensures that the input is in a format compatible with the trained model.

---

## 2. Feature Extraction – MobileNet

MobileNet is used as the deep learning feature extractor.

Instead of training a large CNN completely from scratch, the project uses a lightweight pre-trained architecture to extract important visual representations from MRI scans.

The extracted features contain meaningful information about patterns and structures present in the image.

These features are then passed to the SVM classifier.

---

## 3. Classification – SVM

The extracted MobileNet features are provided to a **Support Vector Machine (SVM)** classifier.

The SVM model performs the classification task and determines the predicted category of the MRI image.

The hybrid approach can be represented as:

**MRI → MobileNet Features → SVM → Prediction**

---

## 4. Tumor Detection and Post-Processing

After classification, the system performs additional image analysis.

The post-processing stage is used to obtain information such as:

* Tumor presence
* Tumor region
* Tumor type
* Approximate tumor size
* Risk indication

This makes the output more informative than a simple classification label.

---

## 5. Risk Analysis

The system provides a simplified risk indication based on the generated analysis.

The result interface presents the information in a clear format so that users can easily understand the output.

---

## 6. Web Application

The complete AI pipeline is integrated into a web application using **Flask**.

The interface allows the user to:

1. Enter patient information.
2. Upload an MRI image.
3. Submit the image for analysis.
4. Process the image through the AI pipeline.
5. View the generated analysis.
6. Review the result and risk information.

---

# 🖥️ Application Features

### 🧠 Brain Tumor Information

The application provides an informational section explaining brain tumors, symptoms, causes, and the importance of early awareness.

### 📤 MRI Upload

Users can upload an MRI image through the web interface.

### 🤖 Automated AI Analysis

The uploaded image is processed automatically using the trained hybrid ML pipeline.

### 🔬 Tumor Classification

The system predicts whether the MRI image indicates a tumor and provides the corresponding classification.

### 📏 Tumor Size Estimation

The post-processing stage provides an approximate tumor-size estimation based on the detected region.

### ⚠️ Risk Indicator

The system presents an understandable risk indication along with the analysis.

### 📄 Diagnostic Report

The result page organizes the prediction, patient information, AI analysis, and other insights into a structured report.

---

# 🛠️ Technology Stack

| Category             | Technology                   |
| -------------------- | ---------------------------- |
| Programming Language | Python                       |
| Web Framework        | Flask                        |
| Deep Learning        | TensorFlow / Keras           |
| Feature Extraction   | MobileNet                    |
| Machine Learning     | Support Vector Machine (SVM) |
| Image Processing     | OpenCV                       |
| Image Handling       | Pillow                       |
| Model Serialization  | Joblib                       |
| Frontend             | HTML, CSS                    |
| Deployment           | Render                       |
| Version Control      | Git & GitHub                 |

---

# 📁 Project Structure

```text
Brain_Tumor_Project/
│
├── app.py
│
├── data_check.py
├── preprocess.py
├── feature_extraction.py
├── train_svm.py
├── post_processing.py
├── test_post_processing.py
│
├── models/
│   ├── trained model files
│   └── supporting model resources
│
├── static/
│   └── images/
│       └── sample MRI images
│
├── templates/
│   ├── HTML templates
│   └── application pages
│
├── requirements.txt
├── runtime.txt
├── .python-version
├── Procfile
├── .gitignore
│
└── README.md
```

---

# 📄 File Description

## `app.py`

The main Flask application.

It connects the web interface with the machine learning pipeline and handles:

* Web routes
* Patient information
* MRI image upload
* Model prediction
* Result generation
* Report display

This is the main entry point of the deployed application.

---

## `preprocess.py`

Contains the image preprocessing functionality.

It prepares MRI images for the subsequent feature extraction stage.

Typical processing includes image resizing, normalization, and conversion into the required input format.

---

## `feature_extraction.py`

Handles deep feature extraction using the MobileNet architecture.

The MRI image is passed through the feature extraction model and converted into a numerical feature representation that can be used by the SVM classifier.

---

## `train_svm.py`

Contains the SVM training pipeline.

It is responsible for training the Support Vector Machine using the extracted feature representations and corresponding labels.

The trained model is then used by the application for prediction.

---

## `post_processing.py`

Handles analysis after the primary model prediction.

It is responsible for additional information such as:

* Tumor-region analysis
* Tumor-size estimation
* Risk-related information
* Result interpretation

---

## `test_post_processing.py`

Used for testing and validating the post-processing functionality before integrating it into the final application.

---

## `data_check.py`

Used for checking and validating the dataset and image-related information during development.

---

## `models/`

Contains the trained machine learning/deep learning model resources required by the application.

These files allow the deployed application to perform predictions without retraining the model for every request.

---

## `static/images/`

Contains static sample MRI images used by the application, such as images displayed in the informational section or demonstration interface.

---

## `templates/`

Contains the HTML templates used to create the web application's user interface.

These templates control pages such as:

* Home page
* Brain tumor information page
* Patient information page
* MRI upload page
* Analysis/result page

---

## `requirements.txt`

Contains the Python dependencies required to install and run the application.

---

## `Procfile`

Specifies the command used by the deployment platform to start the application.

---

## `runtime.txt` / `.python-version`

Used to specify the Python runtime required by the deployment environment.

---

## `.gitignore`

Prevents unnecessary or generated files from being uploaded to the Git repository.

Examples include:

* Python cache files
* Temporary files
* Generated outputs
* Local datasets
* Unnecessary model-development files

---

# 🧪 Model Pipeline

The core AI pipeline follows a hybrid architecture:

### Step 1 — Input

An MRI brain scan is uploaded through the web interface.

### Step 2 — Preprocessing

The MRI image is resized and prepared for model processing.

### Step 3 — Deep Feature Extraction

MobileNet extracts high-level visual features from the MRI image.

### Step 4 — Classification

The extracted features are passed to the trained SVM classifier.

### Step 5 — Post-Processing

The prediction is further analyzed to obtain tumor-related information.

### Step 6 — Result Generation

The application presents the final analysis through a structured web-based report.

---

# 📊 Model Performance

The project evaluation achieved approximately:

* **Accuracy:** 97–99%
* **Precision:** 96–98%
* **Recall:** 95–97%
* **F1-Score:** 96–98%

These values represent the project's reported evaluation results and may vary depending on the dataset, preprocessing pipeline, and test samples.

---

# 👨‍⚕️ Intended Use

The system is designed as an **AI-assisted decision-support tool** for healthcare professionals and researchers.

Potential applications include:

* Preliminary MRI screening
* Research and academic demonstrations
* AI-assisted image analysis
* Supporting early diagnostic assessment
* Patient monitoring research
* Medical AI experimentation

The system should **not be used as an independent medical diagnostic system**. Final clinical decisions should always be made by qualified healthcare professionals.

---

# 🚀 Deployment

The application is deployed as a Flask web service on **Render**.

The deployment uses the project's GitHub repository and installs the required Python dependencies before starting the Flask application.

Render web services can automatically deploy updated code from the connected Git repository.

### Live Application

[Open the Brain Tumor Detection Live Demo](https://brain-tumor-project-8e69.onrender.com?utm_source=chatgpt.com)

### ⚠️ First Load

The live application may take approximately **1–2 minutes to become available** if the hosted service has been inactive.

This is normal behavior for services that are temporarily spun down after inactivity and then started again when a new request arrives. Render documents that free web services can spin down after periods of inactivity and take time to start again.

---

# 💻 Running the Project Locally

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd Brain_Tumor_Project
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Application

```bash
python app.py
```

The Flask application will start locally.

Open the local address shown in the terminal, typically:

```text
http://127.0.0.1:5000
```

---

# 🔐 Privacy and Data Considerations

The application is intended for demonstration, research, and academic purposes.

Users should avoid uploading personally identifiable patient information or sensitive medical records to a publicly accessible demonstration deployment.

Only appropriate, anonymized, and authorized MRI data should be used for testing.

---

# 🔮 Future Scope

Integration with hospital management systems can enable real-time clinical use and assist doctors in faster decision-making. The system can be extended into a mobile application to provide easy access and remote diagnosis capabilities. It can also be enhanced to support multi-class tumor classification such as glioma, meningioma, and pituitary tumors. Improvements in segmentation techniques can help achieve more accurate tumor boundary detection. Real-time MRI analysis can be achieved with faster processing using GPU optimization. Cloud integration can allow large-scale deployment and data handling. Additionally, incorporating explainable AI can improve transparency in predictions, while continuous training with larger and more diverse datasets can further enhance model performance.

---

# 🏆 Key Contribution

The major contribution of this project is the integration of **deep learning-based feature extraction and machine learning-based classification into a single automated MRI analysis pipeline**.

```text
MobileNet
    ↓
Deep Feature Extraction
    ↓
SVM Classification
    ↓
Post-Processing
    ↓
Tumor Analysis
    ↓
Risk & Report Generation
```

This hybrid architecture demonstrates how different AI techniques can be combined to build an efficient healthcare-oriented image analysis system.

---

# 📚 Research & Publication

### Paper Title

**Automated Brain Tumor Detection using Hybrid Machine Learning Models**

### Conference

**2nd International Engineering Data Analytics and Management Conference (EAMCON 2026)**

### Submission ID

**137**

### Presentation

**Accepted for Oral Presentation in Hybrid Mode**

### Conference Venue

**Shinawatra University, Bangkok, Thailand**

### Conference Dates

**8–10 July 2026**

The project was submitted for conference consideration and received acceptance for oral presentation.

---

# 👥 Project Information

**Project:** Automated Brain Tumor Detection using Hybrid Machine Learning Models
**Batch:** 72
**Domain:** Artificial Intelligence & Machine Learning
**Application Area:** Healthcare / Medical Image Analysis

---

# ⚠️ Disclaimer

This project is developed for **academic, research, and demonstration purposes**. The predictions generated by the system should not be considered a definitive medical diagnosis. MRI interpretation and treatment decisions must be performed by qualified healthcare professionals using appropriate clinical information and diagnostic procedures.

---

## ⭐ Project Highlights

* 🧠 Automated brain tumor detection
* 🔬 MRI image analysis
* 🤖 MobileNet-based feature extraction
* 📊 SVM-based classification
* 📏 Tumor size estimation
* ⚠️ Risk-level indication
* 📄 Structured result reporting
* 🌐 Flask web application
* ☁️ Render deployment
* 🎓 Research conference acceptance

---

## 🙌 Acknowledgement

This project was developed as an academic implementation demonstrating the application of Artificial Intelligence and Machine Learning techniques to medical image analysis, with a focus on supporting faster and more accessible preliminary brain tumor screening.
