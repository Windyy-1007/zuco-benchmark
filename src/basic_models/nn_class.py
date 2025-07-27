import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

def classify_knn(train_X, train_y, test_X, test_y, n_neighbors=5):
    """
    Train and evaluate a KNN classifier.
    """

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)

    accuracy = sum(predictions == test_y) / len(test_y)
    p, r, f1, _ = precision_recall_fscore_support(test_y, predictions, average='macro')
    return accuracy, p, r, f1
