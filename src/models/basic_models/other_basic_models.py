import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support

def classify_random_forest(train_X, train_y, test_X, test_y, n_estimators=100, random_state=42):
    """
    Train and evaluate a Random Forest classifier.
    """
    from sklearn.ensemble import RandomForestClassifier

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)

    accuracy = sum(predictions == test_y) / len(test_y)
    p, r, f1, _ = precision_recall_fscore_support(test_y, predictions, average='macro')
    return accuracy, p, r, f1

def classify_logistic_regression(train_X, train_y, test_X, test_y, random_state=42, max_iter=1000):
    """
    Train and evaluate a Logistic Regression classifier.
    """
    from sklearn.linear_model import LogisticRegression

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    clf = LogisticRegression(random_state=random_state, max_iter=max_iter)
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)

    accuracy = sum(predictions == test_y) / len(test_y)
    p, r, f1, _ = precision_recall_fscore_support(test_y, predictions, average='macro')
    return accuracy, p, r, f1

# Gradient Boosting Classifier
def classify_gradient_boosting(train_X, train_y, test_X, test_y, n_estimators=100, random_state=42):
    """
    Train and evaluate a Gradient Boosting classifier.
    """
    from sklearn.ensemble import GradientBoostingClassifier

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)

    accuracy = sum(predictions == test_y) / len(test_y)
    p, r, f1, _ = precision_recall_fscore_support(test_y, predictions, average='macro')
    return accuracy, p, r, f1