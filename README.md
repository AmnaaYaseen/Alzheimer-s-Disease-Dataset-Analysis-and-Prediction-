
# 🧠 Alzheimer's Disease Prediction using Random Forest

This project predicts whether a patient has Alzheimer's Disease based on various health metrics and behavioral symptoms using machine learning techniques, specifically the Random Forest Classifier. It also includes an interactive web interface built with Streamlit for data exploration and prediction.

![Streamlit App](https://img.shields.io/badge/Streamlit-App-red) 
[![Dataset from Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

---

## 📑 Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Data Processing](#data-processing)  
- [Dependencies](#dependencies)  
- [Setup and Installation](#setup-and-installation)  
- [Features](#features)  
- [Usage](#usage)  
- [Model Evaluation](#model-evaluation)  
- [Demo Video](#demo-video)  
- [Contributing](#contributing)  
- [License](#license)

---

## 🔍 Project Overview

This project builds a machine learning model to predict the likelihood of Alzheimer's Disease (AD) based on clinical and behavioral data. The Random Forest algorithm is used as the core classifier, and the dataset is preprocessed with techniques like:

- Dropping irrelevant columns (PatientID, DoctorInCharge)

- Handling missing values by imputing column means

- Balancing the dataset through undersampling the majority class

The model’s performance is evaluated using accuracy, a confusion matrix, and a classification report. An interactive Streamlit web app is included, allowing users to perform EDA, check feature importance, analyze distributions, and make real-time predictions.


---

## 📊 Dataset

The dataset used in this project is sourced from Kaggle:  
👉 [Alzheimer’s Disease Dataset on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data)

- **Rows:** 2,149

- **Columns include:**
  - `Age`
  - `Gender`
  - `BMI`
  - `AlcoholConsumption`
  - `PhysicalActivity`
  - `DietQuality`
  - `MMSE` (Mini-Mental State Examination Score)
  - `Family History of Alzheimer's`
  - `Symptoms` (Memory Complaints, Behavioral Problems, etc.)
  - `Diagnosis` (Target: 0 = No Alzheimer’s, 1 = Alzheimer’s)

**Note:** The dataset is assumed to be mostly clean. Missing numerical values are filled using column means.

---

## ⚙️ Data Processing

- Dropped irrelevant columns: `PatientID`, `DoctorInCharge`
- Imputed missing values with column-wise mean
- Converted target labels in `Diagnosis` to binary (0 and 1)
- Addressed class imbalance using **undersampling** of the majority class
- Split data into training and testing sets (80/20)
- Trained a `RandomForestClassifier` with:
  - `n_estimators=100`
  - `max_depth=10`
  - `min_samples_split=10`
  - `random_state=42`

---

## 📦 Dependencies

Install all required Python packages using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn streamlit
```

**Required Python Libraries:**

- pandas, numpy – Data handling

- matplotlib, seaborn – Visualization

- scikit-learn – Machine learning

- streamlit – Web interface


---

## 💻 Setup and Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/alzheimers-disease-prediction.git
cd alzheimers-disease-prediction
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit App:**

```bash
streamlit run app.py
```

The app will launch in your default browser.

---

## 🌟 Features

- **Exploratory Data Analysis (EDA):**
  - Summary statistics
  - Missing value heatmaps
  - Histograms, boxplots, scatter plots
  - Skewness and correlation analysis
  - Outlier detection via IQR
  - Feature importance from Random Forest

- **Model Evaluation:**
  - Accuracy, Confusion Matrix, Classification Report

- **Interactive Prediction:**
  - Input health metrics via Streamlit form
  - Predict Alzheimer’s diagnosis instantly

---

## ▶️ Usage

Once the app is running:

- Navigate using the sidebar:
  - `Introduction`
  - `EDA` – Explore insights and visualizations
  - `Model` – View prediction results
  - `Conclusion`

- In `Model` section, input features like Age, BMI, Physical Activity, etc., and receive prediction on whether the patient may have Alzheimer's.

---

## 📈 Model Evaluation

The model's performance is evaluated using:

- ✅ **Accuracy**
- 📊 **Confusion Matrix**
- 🧾 **Classification Report**: Precision, Recall, F1-score for both classes

These are all displayed in the `Model` tab of the Streamlit app.

---

## 🎬 Demo Video

📽️👉 [Click here to watch the demo video](https://www.linkedin.com/posts/amnaa-yaseen_datascience-machinelearning-alzheimersprediction-activity-7282565514055307264-Ivdk?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEG7gb4BpMMFhZMbWK6xsHuMPApLT98cIBE)


---

## 🤝 Contributing

Want to improve the project or fix bugs?  
1. Fork the repo  
2. Create a feature branch  
3. Submit a pull request

Issues and suggestions are welcome!

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

> **Developed by [Amna Yaseen](https://www.linkedin.com/in/amnaa-yaseen) — BS Data Science**
