from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle
import joblib

# Train model
iris = load_iris()
X, y = iris.data, iris.target
clf = LogisticRegression()
clf.fit(X, y)

# Save as pickle
with open("iris_model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Alternative: save as joblib
joblib.dump(clf, "iris_model.joblib")
