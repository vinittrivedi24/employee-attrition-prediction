import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
print("First Five Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nStatistical Summary:")
print(df.describe())
sns.set_style("whitegrid")
plt.figure(figsize=(6, 4))
sns.countplot(x="Attrition", data=df)
plt.title("Employee Attrition Count")
plt.show()
plt.figure(figsize=(8, 5))
plt.hist(df["Age"], bins=15)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Employees")
plt.show()
plt.figure(figsize=(6, 4))
sns.countplot(x="Gender", hue="Attrition", data=df)
plt.title("Attrition by Gender")
plt.show()
plt.figure(figsize=(8, 5))
sns.countplot(x="Department", hue="Attrition", data=df)
plt.title("Attrition by Department")
plt.xticks(rotation=15)
plt.show()
plt.figure(figsize=(6, 4))
sns.countplot(x="OverTime", hue="Attrition", data=df)
plt.title("Attrition Based on Overtime")
plt.show()
categorical_columns = df.select_dtypes(include=["object"]).columns
print("Categorical Columns:")
print(categorical_columns)
for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
print("\nData After Encoding:")
print(df.head())
X = df.drop("Attrition", axis=1)
y = df["Attrition"]
print("\nFeature Matrix Shape:", X.shape)
print("Target Variable Shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("\nModel Training Completed Successfully!")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
importance = importance.sort_values(by="Importance", ascending=False)
print("\nTop 10 Important Features:")
print(importance.head(10))
plt.figure(figsize=(10, 6))
plt.barh(importance["Feature"][:10], importance["Importance"][:10])
plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.show()