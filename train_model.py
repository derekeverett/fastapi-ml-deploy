import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# load the iris dataset
X, y = load_iris(return_X_y=True)
# fit a Random Forest Model
model = RandomForestClassifier()
model.fit(X, y)
# serialize the model and save to disk
joblib.dump(model, "model.joblib")