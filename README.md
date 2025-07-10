
# 🧠 Alzheimer's Disease Prediction using Random Forest

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#-license)
[![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-ff4b4b?logo=streamlit)](https://streamlit.io)

This project predicts whether a patient has Alzheimer's Disease based on various health metrics and behavioral symptoms using machine learning techniques, specifically the Random Forest Classifier. It also includes an interactive web interface built with Streamlit for data exploration and prediction.

## 📑 Table of Contents
- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Data Processing](#data-processing)  
- [Tech Stack](#-tech-stack)  
- [Dependencies](#dependencies)  
- [Setup and Installation](#setup-and-installation)  
- [Features](#features)  
- [Usage](#usage)  
- [Model Evaluation](#model-evaluation)  
- [Demo Video](#demo-video)  
- [Contributing](#contributing)  
- [Developer](#-developer)  
- [License](#-license)

## 🔍 Project Overview
This project builds a machine learning model to predict the likelihood of Alzheimer's Disease (AD) based on clinical and behavioral data. The Random Forest algorithm is used as the core classifier, and the dataset is preprocessed with techniques like:
- Dropping irrelevant columns (PatientID, DoctorInCharge)
- Handling missing values by imputing column means
- Balancing the dataset through undersampling the majority class

The model’s performance is evaluated using accuracy, a confusion matrix, and a classification report. An interactive Streamlit web app is included, allowing users to perform EDA, check feature importance, analyze distributions, and make real-time predictions.

## 📊 Dataset
The dataset used in this project is sourced from Kaggle:  
👉 [Alzheimer’s Disease Dataset on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data)

- **Rows:** 2,149  
- **Columns include:** Age, Gender, BMI, AlcoholConsumption, PhysicalActivity, DietQuality, MMSE, FamilyHistory, Symptoms, Diagnosis

## ⚙️ Data Processing
- Dropped irrelevant columns: `PatientID`, `DoctorInCharge`
- Imputed missing values with column-wise mean
- Converted target labels (`Diagnosis`) to binary (0 and 1)
- Addressed class imbalance using undersampling
- Split data into training and testing sets (80/20 split)
- Trained a `RandomForestClassifier` with specific hyperparameters

## 🧰 Tech Stack
| Category         | Tools & Libraries           |
|------------------|-----------------------------|
| Language         | Python                      |
| ML Libraries     | scikit-learn                |
| EDA & Plots      | pandas, matplotlib, seaborn |
| Web App          | Streamlit                   |

## 📦 Dependencies
```bash
pip install pandas numpy seaborn matplotlib scikit-learn streamlit
```

## 💻 Setup and Installation
```bash
git clone https://github.com/yourusername/alzheimers-disease-prediction.git
cd alzheimers-disease-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 🌟 Features
- EDA: statistics, heatmaps, plots, outlier detection, feature importance
- Model Evaluation: Accuracy, Confusion Matrix, Classification Report
- Interactive Prediction using a Streamlit form

## ▶️ Usage
- Navigate the sidebar for Introduction, EDA, Model, and Conclusion
- Input health metrics and get Alzheimer's prediction

## 📈 Model Evaluation
Metrics used: Accuracy, Confusion Matrix, Classification Report

## 🎬 Demo Video
📽️👉 [Watch demo video](https://www.linkedin.com/posts/amnaa-yaseen_datascience-machinelearning-alzheimersprediction-activity-7282565514055307264-Ivdk)

## 🤝 Contributing
1. Fork the repo  
2. Create a feature branch  
3. Submit a pull request

## 👩‍💻 Developer
**Amna Yaseen**  
[GitHub](https://github.com/AmnaaYaseen) • [LinkedIn](https://linkedin.com/in/amnaa-yaseen)

## 📄 License
This project is licensed under the MIT License.  
© 2025 Amna Yaseen. All rights reserved.
