from classification import load_classification, train_classifier
from regression import load_regression, get_regression_divisions, train_regressor
from getgames import get_players


if __name__ == '__main__':
    count = 10000
    # get_players(20649224, count)
    # get_regression_divisions(str(count))
    if False:
        players, divisions = load_regression(str(count))
        train_regressor(players, divisions, 'SVR')
    else:
        # tree_error(X_train, X_test, y_train, y_test, random_state)
        p, d_c = load_classification(str(count))
        train_classifier(p, d_c, 'NN')
