import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
from pyswarms.single.global_best import GlobalBestPSO
import config

def classify_svm(train_X, train_y, test_X, test_y):
    """
    Train an SVM classifier and evaluate it.

    Args:
    train_X (list): Training data.
    train_y (list): Training labels.
    test_X (list): Test data.
    test_y (list): Test labels.

    Returns:
    tuple: Accuracy, precision, recall, and F1-score.
    """
    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    # Train the SVM classifier
    clf = SVC(random_state=config.seed, kernel=config.kernel, gamma='scale', cache_size=1000)
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)

    # Evaluate the classifier
    accuracy = sum(predictions == test_y) / len(test_y)
    p, r, f1, _ = precision_recall_fscore_support(test_y, predictions, average='macro')
    return accuracy, p, r, f1

def classify_svm_pso(train_X, train_y, test_X, test_y, n_particles=20, iters=30):
    """
    SVM-PSO hybrid: Use Particle Swarm Optimization to select features, then classify with SVM.
    Requires: pip install pyswarms
    """
    

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    n_features = train_X.shape[1]

    def fitness(mask):
        # mask: shape (n_particles, n_features)
        scores = []
        for m in mask:
            if np.count_nonzero(m) == 0:
                scores.append(0)
                continue
            idx = np.where(m > 0.5)[0]
            clf = SVC(random_state=config.seed, kernel=config.kernel, gamma='scale', cache_size=1000)
            try:
                clf.fit(train_X[:, idx], train_y)
                score = clf.score(train_X[:, idx], train_y)
            except Exception:
                score = 0
            scores.append(score)
        return -np.array(scores)  # minimize negative accuracy

    optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=n_features, options={'c1': 2, 'c2': 2, 'w': 0.9})
    best_cost, best_pos = optimizer.optimize(fitness, iters=iters, verbose=False)
    selected_idx = np.where(best_pos > 0.5)[0]
    if len(selected_idx) == 0:
        selected_idx = np.arange(n_features)  # fallback: use all

    clf = SVC(random_state=config.seed, kernel=config.kernel, gamma='scale', cache_size=1000)
    clf.fit(train_X[:, selected_idx], train_y)
    predictions = clf.predict(test_X[:, selected_idx])
    accuracy = sum(predictions == test_y) / len(test_y)
    p, r, f1, _ = precision_recall_fscore_support(test_y, predictions, average='macro')
    return accuracy, p, r, f1

def classify_svm_rfe(train_X, train_y, test_X, test_y, n_features_to_select=10):
    """
    SVM-RFE hybrid: Use Recursive Feature Elimination to select features, then classify with SVM.
    """
    from sklearn.feature_selection import RFE

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    estimator = SVC(random_state=config.seed, kernel=config.kernel, gamma='scale', cache_size=1000)
    selector = RFE(estimator, n_features_to_select=min(n_features_to_select, train_X.shape[1]), step=1)
    selector = selector.fit(train_X, train_y)
    train_X_rfe = selector.transform(train_X)
    test_X_rfe = selector.transform(test_X)

    clf = SVC(random_state=config.seed, kernel=config.kernel, gamma='scale', cache_size=1000)
    clf.fit(train_X_rfe, train_y)
    predictions = clf.predict(test_X_rfe)
    accuracy = sum(predictions == test_y) / len(test_y)
    p, r, f1, _ = precision_recall_fscore_support(test_y, predictions, average='macro')
    return accuracy, p, r, f1

