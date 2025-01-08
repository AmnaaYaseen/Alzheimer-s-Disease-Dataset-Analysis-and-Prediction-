import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
import streamlit as st

# Load Dataset
data_path = 'alzheimers_disease_data.csv'
data = pd.read_csv(data_path)

# Drop irrelevant columns if they exist
if 'PatientID' in data.columns and 'DoctorInCharge' in data.columns:
    data.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Convert Diagnosis column to binary (0: No Alzheimer's, 1: Alzheimer's)
data['Diagnosis'] = data['Diagnosis'].apply(lambda x: 1 if x == 1 else 0)

# Visualize the class distribution before balancing
print("Class distribution before balancing:")
print(data['Diagnosis'].value_counts())

# Split the dataset into two groups: Majority class (No Alzheimer's) and Minority class (Alzheimer's)
majority_class = data[data['Diagnosis'] == 0]
minority_class = data[data['Diagnosis'] == 1]

# Ensure there is a valid minority class before balancing
if len(minority_class) > 0:
    # Undersample the majority class
    majority_class_undersampled = resample(majority_class,
                                           replace=False,    # Without replacement
                                           n_samples=len(minority_class), # Match minority class size
                                           random_state=42)  # For reproducibility

    # Combine minority class with undersampled majority class
    balanced_data = pd.concat([majority_class_undersampled, minority_class])

    # Visualize the class distribution after balancing
    print("Class distribution after balancing:")
    print(balanced_data['Diagnosis'].value_counts())
else:
    print("Minority class is missing or not correctly loaded.")

# Split features and target
X = balanced_data.drop('Diagnosis', axis=1)
y = balanced_data['Diagnosis']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestClassifier with some parameters
model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Check prediction distribution
print("Prediction distribution on test set:", np.bincount(y_pred))

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Streamlit App
st.title("Alzheimer's Disease Prediction")

st.sidebar.header("Navigation")
nav_option = st.sidebar.radio("Choose a section", ["Introduction", "EDA", "Model", "Conclusion"])

if nav_option == "Introduction":
    st.subheader("Project Introduction")
    st.write("This project predicts whether a patient has Alzheimer's Disease (AD) based on various health metrics.")
    st.write("Dataset contains 2,149 rows and multiple features like Age, Gender, BMI, health metrics, and symptoms.")


# Streamlit EDA Section
if nav_option == "EDA":
    st.subheader("Exploratory Data Analysis")

    # 1. Summary Statistics (Mean, Median, Mode, etc.)
    st.write("### 1. Summary Statistics")
    st.write(data.describe())  # Provides mean, median, mode, min, max, etc.

    # 2. Missing Value Analysis
    st.write("### 2. Missing Value Analysis")
    st.write("Missing values in each column:")
    st.write(data.isnull().sum())
    
    # Visualize missing values
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis", ax=ax)
    st.pyplot(fig)
    plt.clf()

    # 3. Data Types and Unique Value Counts
    st.write("### 3. Data Types and Unique Value Counts")
    st.write("Data Types and the number of unique values for each column:")
    st.write(data.dtypes)
    st.write(data.nunique())

    # 4. Feature Distribution (Histograms)
    st.write("### 4. Feature Distribution (Histograms)")

    column_hist = st.selectbox("Select a column for histogram", X.columns)
    fig, ax = plt.subplots(figsize=(10, 6))
    data_column = data[column_hist]
    
    # Generate unique color for each bin
    color_map = sns.color_palette("coolwarm", n_colors=20)
    num_bins = 20
    counts, bins, patches = ax.hist(data_column, bins=num_bins, alpha=0.7)
    
    # Color each bar differently
    for i, patch in enumerate(patches):
        patch.set_facecolor(color_map[i % len(color_map)])  # Apply color
    
    ax.set_title(f"Histogram of {column_hist}")
    st.pyplot(fig)
    plt.clf()

    # 5. Boxplot for Outlier Detection
    st.write("### 5. Boxplot for Outlier Detection")

    column_box = st.selectbox("Select a column for boxplot", X.columns)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=data[column_box], palette="Set2", ax=ax)
    ax.set_title(f"Boxplot of {column_box}")
    st.pyplot(fig)
    plt.clf()

    # 6. Correlation Matrix (Correlation Analysis)
    st.write("### 6. Correlation Matrix")
    
    corr_columns = st.multiselect("Select columns for correlation matrix", options=list(X.columns), default=X.columns[:5])
    
    if corr_columns:
        corr_matrix = data.corr()
        selected_corr_matrix = corr_matrix[corr_columns][corr_columns]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(selected_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.clf()

    # 7. Skewness Analysis
    st.write("### 7. Skewness of Columns")
    skewness = data.skew()
    st.write("Skewness values for each column:")
    st.write(skewness)
    
    # Plot skewness for each feature
    fig, ax = plt.subplots(figsize=(10, 6))
    skewness.plot(kind='bar', ax=ax)
    ax.set_title("Skewness of Features")
    st.pyplot(fig)
    plt.clf()

    # Scatter plots
    st.write("### 8. Scatter Plot Between Features")
    scatter_column_x = st.selectbox("Select a column for X-axis", X.columns)
    scatter_column_y = st.selectbox("Select a column for Y-axis", X.columns)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data[scatter_column_x], data[scatter_column_y], alpha=0.7, color="coral")
    ax.set_xlabel(scatter_column_x)
    ax.set_ylabel(scatter_column_y)
    ax.set_title(f"Scatter Plot Between {scatter_column_x} and {scatter_column_y}")
    st.pyplot(fig)
    plt.clf()


    # 9. Trend Analysis (if applicable)
    if 'Date' in data.columns:
        st.write("### 9. Trend Analysis")
        st.line_chart(data.set_index('Date')['value'])  # Example if you have a date column
    
    # 10. Grouped Aggregations (e.g., mean by gender)
    st.write("### 10. Grouped Aggregations")
    st.write("Grouped data by Gender:")
    st.write(data.groupby('Gender').mean())

    # 11. Feature Importance Analysis (using RandomForest)
    st.write("### 11. Feature Importance Analysis")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importances.plot(kind='bar', ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)
    plt.clf()

    # 12. Outlier Detection (IQR Method)
    st.write("### 12. Outlier Detection (IQR Method)")
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
    st.write("Outliers detected using IQR method:")
    st.write(outliers)

    # 13. Distribution of Numerical Features
    st.write("### 13. Distribution of Numerical Features")
    numerical_features = data.select_dtypes(include=[np.number]).columns

    # Dropdown to select a column
    selected_column = st.selectbox("Select a numerical column to display its distribution:", numerical_features)

    # Plot the distribution of the selected column
    if selected_column:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data[selected_column], bins=30, kde=True, color='skyblue', ax=ax)  # Using histplot for better aesthetics
        ax.set_title(f"Distribution of {selected_column}")
        st.pyplot(fig)
        plt.clf()

    # 14. Heatmap of Feature Correlations (Excluding Target)
    st.write("### 14. Heatmap of Feature Correlations (Excluding Target)")

    # Select the first 15 numerical columns (excluding the target column)
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude 'Diagnosis' from the initial selection
    numerical_features = [col for col in numerical_features if col != 'Diagnosis']

    # Default columns (first 15 or fewer if there are less than 15 columns)
    default_columns = numerical_features[:15]

    # Dropdown to select columns for correlation matrix
    selected_columns = st.multiselect("Select numerical columns to include in the correlation matrix:", numerical_features, default=default_columns)

    # If the user selects columns
    if selected_columns:
        # Create correlation matrix for selected columns
        corr_matrix = data[selected_columns].corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
        plt.clf()

elif nav_option == "Model":
    st.subheader("Model Evaluation")
    st.write(f"### Accuracy: {accuracy:.2f}")
    
    st.write("### Confusion Matrix")
    st.write(conf_matrix)
    
    st.write("### Classification Report")
    
    # Convert classification report to pandas DataFrame
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report_dict).transpose()

    # Display the classification report as a nice table
    st.dataframe(class_report_df)

    st.write("### Make a Prediction")
    
    # Numerical Inputs
    input_data = {
    "Age": st.number_input("Enter Age", value=0.0),
    "BMI": st.number_input("Enter BMI", value=0.0),
    "AlcoholConsumption": st.number_input("Enter Alcohol Consumption (0 to 20 units per week)", min_value=0, max_value=20, value=0),
    "PhysicalActivity": st.number_input("Enter Physical Activity (1 to 10 scale)", min_value=1, max_value=10, value=5),
    "DietQuality": st.number_input("Enter Diet Quality (1 to 10 scale)", min_value=1, max_value=10, value=5),
    "SleepQuality": st.number_input("Enter Sleep Quality (4 to 10 scale)", min_value=4, max_value=10, value=7),
    "MMSE": st.number_input("Enter MMSE (1 to 30 scale)", min_value=1, max_value=30, value=15),
    "FunctionalAssessment": st.number_input("Enter Functional Assessment (1 to 10 scale)", min_value=1, max_value=10, value=5),
    "ADL": st.number_input("Enter ADL (1 to 10 scale)", min_value=1, max_value=10, value=5),
    "SystolicBP": st.number_input("Enter Systolic BP (mmHg)", min_value=90, max_value=180, value=120),
    "DiastolicBP": st.number_input("Enter Diastolic BP (mmHg)", min_value=60, max_value=120, value=80),
    "CholesterolTotal": st.number_input("Enter Total Cholesterol (mg/dL)", min_value=150, max_value=300, value=200),
    "CholesterolLDL": st.number_input("Enter LDL Cholesterol (mg/dL)", min_value=50, max_value=200, value=100),
    "CholesterolHDL": st.number_input("Enter HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50),
    "CholesterolTriglycerides": st.number_input("Enter Triglycerides (mg/dL)", min_value=50, max_value=400, value=150)
}

# Categorical Inputs
    categorical_data = {
    "Gender": st.selectbox("Enter Gender", ["male", "female"]),
    "Ethnicity": st.selectbox("Enter Ethnicity (0=Caucasian, 1=African American, 2=Asian, 3=Other)", [0, 1, 2, 3]),
    "EducationLevel": st.selectbox("Enter Education Level (0=None, 1=High School, 2=Bachelor's, 3=Higher)", [0, 1, 2, 3]),
    "Smoking": st.selectbox("Enter Smoking", ["yes", "no"]),
    "FamilyHistoryAlzheimers": st.selectbox("Enter Family History of Alzheimer's", ["yes", "no"]),
    "CardiovascularDisease": st.selectbox("Enter Cardiovascular Disease", ["yes", "no"]),
    "Diabetes": st.selectbox("Enter Diabetes", ["yes", "no"]),
    "Depression": st.selectbox("Enter Depression", ["yes", "no"]),
    "HeadInjury": st.selectbox("Enter Head Injury", ["yes", "no"]),
    "Hypertension": st.selectbox("Enter Hypertension", ["yes", "no"]),
    "MemoryComplaints": st.selectbox("Enter Memory Complaints", ["yes", "no"]),
    "BehavioralProblems": st.selectbox("Enter Behavioral Problems", ["yes", "no"]),
    "Confusion": st.selectbox("Enter Confusion", ["yes", "no"]),
    "Disorientation": st.selectbox("Enter Disorientation", ["yes", "no"]),
    "PersonalityChanges": st.selectbox("Enter Personality Changes", ["yes", "no"]),
    "DifficultyCompletingTasks": st.selectbox("Enter Difficulty Completing Tasks", ["yes", "no"]),
    "Forgetfulness": st.selectbox("Enter Forgetfulness", ["yes", "no"]),
}

    # Encode categorical inputs
    for key in categorical_data:
        categorical_data[key] = 1 if categorical_data[key] == "yes" or categorical_data[key] == "female" else 0

    # Combine all inputs
    final_input_data = {**input_data, **categorical_data}
    input_df = pd.DataFrame([final_input_data])

    # Reindex to match the training data columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Display input data (if needed for debugging or testing)
    st.write("Input Data:", input_df)

    # Encode categorical inputs
    for key in categorical_data:
        categorical_data[key] = 1 if categorical_data[key] == "yes" or categorical_data[key] == "female" else 0
    
    # Combine all inputs
    final_input_data = {**input_data, **categorical_data}
    input_df = pd.DataFrame([final_input_data])

    # Align input data columns with model training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)  # Reindex to match the training data columns
    
    # Prediction
    prediction = model.predict(input_df)[0]
    st.write("Prediction: Alzheimer's Disease" if prediction == 1 else "Prediction: No Alzheimer's Disease")


elif nav_option == "Conclusion":
    st.subheader("Conclusion")
    st.write(f"The Random Forest Classifier achieved an accuracy of {accuracy:.2f}, demonstrating its effectiveness in predicting Alzheimer's Disease.")
