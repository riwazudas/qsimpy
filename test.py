import ray
from ray import tune
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

def train_model(config):
    # Train a RandomForest with the provided config (hyperparameters)
    clf = RandomForestClassifier(n_estimators=config["n_estimators"], max_depth=config["max_depth"])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    tune.report(accuracy=accuracy)

# Define search space for hyperparameters
search_space = {
    "n_estimators": tune.choice([10, 50, 100]),
    "max_depth": tune.choice([3, 5, 10])
}

# Run the Ray Tune experiment
analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=10,  # Number of samples to try from the hyperparameter space
    resources_per_trial={"cpu": 1, "gpu": 0},  # Adjust based on your hardware
    metric="accuracy",  # Specify the metric to optimize
    mode="max"  # Specify whether to maximize or minimize the metric
)

# Print best configuration and accuracy
print("Best config: ", analysis.best_config)
print("Best accuracy: ", analysis.best_result["accuracy"])
