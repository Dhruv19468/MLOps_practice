import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train simple model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)