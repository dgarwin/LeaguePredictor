from classification import load_classification, train_classifier
from regression import load_regression, get_regression_divisions, train_regressor
from getgames import get_players, get_champion_masteries


def classify(count):
    # tree_error(X_train, X_test, y_train, y_test, random_state)
    p, d_c = load_classification(str(count))
    # print d_c.describe()
    train_classifier(p, d_c, 'NN')


def regress(count):
    players, divisions = load_regression(str(count))
    train_regressor(players, divisions, 'SVR')


def other(count):
    # get_players(20649224, count)
    # get_regression_divisions(str(count))
    get_champion_masteries(count)


if __name__ == '__main__':
    count = 10000
    other(count)
