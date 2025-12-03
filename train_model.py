import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    df = pd.read_csv("dataset/keypoints.csv")

    # last column is the label
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]   

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # model
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )

    clf.fit(X_train, y_train)

    # evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.3f}")

    # save model
    joblib.dump(clf, "model.pkl")
    print("Saved model.pkl")

if __name__ == "__main__":
    main()
