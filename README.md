# 📊 Holistic Data Preparer Project

## 📌 Project Title
Holistic Data Preparer – Data Preprocessing and Feature Engineering

---

## 🎯 Objective
The objective of this project is to clean, preprocess, and transform raw data into a structured format suitable for machine learning models.

---

## 📂 Dataset
The dataset used in this project contains customer information including:
- Demographics (age, gender, region)
- Financial data (income, loan amount, credit score)
- Behavioral data (transactions, spending)
- Target variable (default_flag)

---

## 🛠️ Tools & Technologies Used
- Python 🐍
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- ydata-profiling

---

## ⚙️ Project Workflow

### 1. Data Loading
- Imported dataset using Pandas
- Checked structure using `.info()` and `.describe()`

### 2. Data Cleaning
- Handled missing values:
  - Numerical → Mean
  - Categorical → Most Frequent
- Removed duplicate records

### 3. Feature Engineering
- Created new feature:
  - Debt-to-Income Ratio
- Extracted date features:
  - Year, Month, Day

### 4. Outlier Handling
- Applied IQR method to detect and treat outliers

### 5. Encoding
- Label Encoding for binary variables
- One-Hot Encoding for categorical variables

### 6. Feature Scaling
- Applied StandardScaler to normalize numerical data
- Excluded target variable (`default_flag`) from scaling

### 7. Exploratory Data Analysis (EDA)
- Histogram for distribution
- Heatmap for correlation
- Pairplot for feature relationships

### 8. Data Profiling
- Generated automated report using `ydata-profiling`

---

## 📊 Key Insights
- Missing values were successfully handled
- Outliers were treated using IQR method
- Strong relationships observed between financial features
- New feature improved data understanding
- Dataset is now ready for machine learning

---

## ✅ Conclusion
The dataset has been successfully cleaned, transformed, and prepared for further analysis and machine learning tasks.

---

## 📁 Output Files
- Cleaned dataset (`cleaned_data.csv`)
- Profiling report (`final_report.html`)
- Visualizations (graphs)

---

## 🚀 Future Scope
- Apply machine learning models for prediction
- Improve dataset with additional features
- Deploy model as a web application

---

## 🙏 Acknowledgement
This project was completed as part of academic learning in data preprocessing and analysis.
