import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
import config

def classify_voting(train_X, train_y, test_X, test_y, voting='hard'):
    """
    Train a voting ensemble classifier (KNN, SVM, RF, LR) and evaluate it.

    Args:
    train_X (list): Training data.
    train_y (list): Training labels.
    test_X (list): Test data.
    test_y (list): Test labels.
    voting (str): 'hard' or 'soft' voting strategy.

    Returns:
    tuple: Accuracy, precision, recall, and F1-score.
    """
    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    # Initialize base classifiers
    knn = KNeighborsClassifier(n_neighbors=5)
    svm = SVC(random_state=config.seed, kernel=config.kernel, gamma='scale', 
              probability=True if voting == 'soft' else False, cache_size=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=config.seed)
    lr = LogisticRegression(random_state=config.seed, max_iter=1000)

    # Create voting classifier
    estimators = [
        ('knn', knn),
        ('svm', svm),
        ('rf', rf),
        ('lr', lr)
    ]
    
    voting_clf = VotingClassifier(estimators=estimators, voting=voting)
    
    # Train the voting classifier
    voting_clf.fit(train_X, train_y)
    predictions = voting_clf.predict(test_X)

    # Evaluate the classifier
    accuracy = sum(predictions == test_y) / len(test_y)
    p, r, f1, _ = precision_recall_fscore_support(test_y, predictions, average='macro')
    return accuracy, p, r, f1

def classify_voting_soft(train_X, train_y, test_X, test_y):
    """
    Train a soft voting ensemble classifier and evaluate it.
    """
    return classify_voting(train_X, train_y, test_X, test_y, voting='soft')

def classify_voting_hard(train_X, train_y, test_X, test_y):
    """
    Train a hard voting ensemble classifier and evaluate it.
    """
    return classify_voting(train_X, train_y, test_X, test_y, voting='hard')
