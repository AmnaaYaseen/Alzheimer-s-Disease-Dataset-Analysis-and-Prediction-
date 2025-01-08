# Alzheimer's Disease Prediction using Random Forest

This project predicts whether a patient has Alzheimer's Disease based on various health metrics and behavioral symptoms using machine learning techniques, 
specifically the Random Forest Classifier. It also provides an interactive web interface using Streamlit for data exploration and prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Setup and Installation](#setup-and-installation)
- [Features](#features)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project involves building a classification model to predict whether a person has Alzheimer's Disease (AD) based on various health metrics. 
The dataset includes several columns such as Age, Gender, BMI, and numerous health factors, including symptoms like Memory Complaints, Behavioral Problems, and more.

The model is built using the Random Forest algorithm, which is trained on a balanced dataset where the minority class is undersampled. After training, 
the model's performance is evaluated using accuracy, confusion matrix, and classification report.

Additionally, the project features an interactive web application developed with Streamlit that allows users to perform Exploratory Data Analysis (EDA), 
view model results, and input custom data for prediction.

## Dataset

The dataset used in this project contains 2,149 rows of data collected from patients. It includes the following key features:

- **Age**: Age of the patient
- **Gender**: Gender of the patient (male/female)
- **BMI**: Body Mass Index
- **AlcoholConsumption**: Weekly alcohol consumption (units)
- **PhysicalActivity**: Physical activity level (1 to 10 scale)
- **DietQuality**: Diet quality (1 to 10 scale)
- **MMSE**: Mini-Mental State Examination score (1 to 30 scale)
- **Family History of Alzheimer's**: Whether the patient has a family history of Alzheimer's (yes/no)
- **Symptoms**: Includes symptoms like Memory Complaints, Behavioral Problems, and more
- **Diagnosis**: The target column (0: No Alzheimer's, 1: Alzheimer's)

The dataset is assumed to be clean, and missing values are imputed with the mean of the respective columns.

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **seaborn**: Statistical data visualization
- **matplotlib**: Plotting and graphing
- **scikit-learn**: Machine learning algorithms and tools
- **streamlit**: Building the interactive web application

Install the required dependencies by running the following command:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn streamlit
```

## Setup and Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/yourusername/alzheimers-disease-prediction.git
    ```

2. Change directory to the project folder:

    ```bash
    cd alzheimers-disease-prediction
    ```

3. Install the required dependencies (if you haven't already):

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

The application will open in your default web browser.

## Features

- **Exploratory Data Analysis (EDA)**: View summary statistics, visualizations, and missing value analysis.
- **Model Evaluation**: View accuracy, confusion matrix, and classification report of the trained model.
- **Interactive Prediction**: Enter personal health data through a form and receive predictions on whether you have Alzheimer's Disease.
  
## Usage

1. Launch the Streamlit app as described above.
2. Navigate through the sections:
   - **Introduction**: Learn about the project.
   - **EDA**: Explore data visualizations and summary statistics.
   - **Model**: View model evaluation metrics and make predictions.
   - **Conclusion**: See a summary of the results and model performance.
3. For prediction, input numerical and categorical data (e.g., Age, BMI, Smoking status) to receive a prediction of whether the person has Alzheimer's Disease or not.

## Model Evaluation

After training the model, we evaluate its performance using:

- **Accuracy**: The proportion of correct predictions made by the model.
- **Confusion Matrix**: A matrix that shows the classification results, highlighting true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Precision, recall, F1-score, and support for each class (Alzheimer's vs No Alzheimer's).

The model's accuracy is displayed on the web app, and users can evaluate its performance.

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request. You can also open issues for any bugs or feature requests.
