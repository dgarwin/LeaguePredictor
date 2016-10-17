from classification import xgboo
from PlayerCollection import PlayerCollection
from classification import svm, random_forest, mlp
from modeling import get_save_results
# File for testing models, as in the paper


def train_svm(count):
    pc = PlayerCollection(size=count)
    X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=False)
    model = svm()
    get_save_results(X_train, X_test, y_train, y_test, model, 'SVM')


def train_rf(count):
    pc = PlayerCollection(size=count)
    X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=False)
    model = random_forest()
    get_save_results(X_train, X_test, y_train, y_test, model, 'RF')


def train_xgb(count):
    pc = PlayerCollection(size=count)
    X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=False)
    model = xgboo()
    get_save_results(X_train, X_test, y_train, y_test, model, 'XGBoost')


def train_nn(count):
    pc = PlayerCollection(size=count)
    X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=True)
    model = mlp()
    get_save_results(X_train, X_test, y_train, y_test, model, 'NN')


if __name__ == '__main__':
    # Train all classifiers
    train_svm(15000)
    train_xgb(15000)
    train_rf(15000)
    train_nn(15000)
