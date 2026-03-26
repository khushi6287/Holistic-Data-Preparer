
"""
# **Import Libraries**
"""

!pip install ydata-profiling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

"""# **Upload Dataset**"""

df = pd.read_csv("holistic_dataset.csv")
df.head()

"""# **Data Understanding**"""

df.shape
df.columns
df.info()
df.describe()

"""# **Check Missing Values**"""

df.isnull().sum()

"""# **Handle Missing Values**"""

from sklearn.impute import SimpleImputer

# Numerical
num_cols = df.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Categorical
cat_cols = df.select_dtypes(include='object').columns

imputer_cat = SimpleImputer(strategy='most_frequent')
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

"""# **Remove Duplicates**"""

df.drop_duplicates(inplace=True)

"""# **Date Handling**"""

df['join_date'] = pd.to_datetime(df['join_date'])

df['year'] = df['join_date'].dt.year
df['month'] = df['join_date'].dt.month
df['day'] = df['join_date'].dt.day

"""# **Outlier Handling (IQR Method)**"""

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

"""# **Encoding**"""

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
if 'gender' in df.columns:
    df['gender'] = le.fit_transform(df['gender'])

df = pd.get_dummies(df, drop_first=True)

"""# **Feature Engineering**"""

df['debt_income_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)

"""# **Scaling**"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols = df.select_dtypes(include=np.number).columns
num_cols = num_cols.drop('default_flag')

df[num_cols] = scaler.fit_transform(df[num_cols])

"""# **EDA**
**Histogram**
"""

df.hist(figsize=(12,10))
plt.show()

"""**Heatmap**"""

sns.heatmap(df.corr(numeric_only=True), annot=True)

"""**Pairplot**"""

sns.pairplot(df.sample(200))
plt.show()

"""# **Data Profiling**"""

from ydata_profiling import ProfileReport

profile = ProfileReport(df)
profile.to_file("final_report.html")

"""# **Save Final Dataseta**"""

df.to_csv("cleaned_data.csv", index=False)

"""# **Insights (VERY IMPORTANT)**"""

print("Insights:")
print("1. Missing values handled using mean and mode")
print("2. Outliers treated using IQR method")
print("3. Encoding applied to categorical variables")
print("4. Debt to Income ratio created as new feature")
print("5. Dataset is now ready for machine learning")

