import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Load dataset
df=pd.read_csv("survey lung cancer.csv")
# Encode categorical features 
le = LabelEncoder()
for col in df.columns:
	df[col] = le.fit_transform(df[col])
#Features and label
X=df.drop("LUNG_CANCER", axis=1)
y=df["LUNG_CANCER"]

#Train-test split
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.2, random_state=42)

#Model training
model=RandomForestClassifier()
model.fit(X_train, y_train)

#Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
