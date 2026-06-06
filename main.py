import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("=" * 50)
print("DECISION TREE")
print("=" * 50)

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

dt_acc = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", round(dt_acc, 4))

print("\nClassification Report")
print(classification_report(y_test, y_pred_dt))

plt.figure(figsize=(20, 10))

plot_tree(
    dt,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    max_depth=3,
    fontsize=10
)

plt.title("Decision Tree (First 3 Levels)")
plt.show()

print("\n" + "=" * 50)
print("OVERFITTING ANALYSIS")
print("=" * 50)

depths = [1, 2, 3, 4, 5, 6, 7, 8]

train_scores = []
test_scores = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    train_scores.append(train_acc)
    test_scores.append(test_acc)

    print(
        f"Depth={depth} | Train Accuracy={train_acc:.4f} | Test Accuracy={test_acc:.4f}"
    )

plt.figure(figsize=(8, 5))

plt.plot(
    depths,
    train_scores,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    depths,
    test_scores,
    marker="o",
    label="Testing Accuracy"
)

plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting Analysis")
plt.legend()
plt.grid(True)

plt.show()

best_depth = depths[test_scores.index(max(test_scores))]

print("\nBest Depth Based on Test Accuracy:", best_depth)

print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", round(rf_acc, 4))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

print("\n" + "=" * 50)
print("ACCURACY COMPARISON")
print("=" * 50)

print("Decision Tree Accuracy :", round(dt_acc, 4))
print("Random Forest Accuracy :", round(rf_acc, 4))

comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [dt_acc, rf_acc]
})

plt.figure(figsize=(6, 4))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.show()

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.show()

print("\nTop Important Feature:")
print(importance.iloc[0]["Feature"])

print("\n" + "=" * 50)
print("5-FOLD CROSS VALIDATION")
print("=" * 50)

cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=5
)

print("Cross Validation Scores:")
print(cv_scores)

print("\nAverage CV Accuracy:")
print(round(cv_scores.mean(), 4))