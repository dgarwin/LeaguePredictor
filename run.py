from classification import load_classification, train_classifier
from regression import load_regression, get_regression_divisions
from getgames import get_players


if __name__ == '__main__':
    count = 10000
    #get_players(20649224, count)
    #p, d_c = load_classification(str(count))
    get_regression_divisions(str(count))
    #train_classifier(p, d_c)