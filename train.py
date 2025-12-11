import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'iris_model.pkl')