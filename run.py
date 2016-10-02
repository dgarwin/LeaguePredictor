from PlayerCollection import PlayerCollection
from classification import nn, m
from modeling import get_save_results


def fetch_data():
    pc = PlayerCollection(size=15000, load=False)
    pc.get_players(20649224)


def load_data(count, description):
    pc = PlayerCollection(size=count)
    # Get raw Data
    X_train, X_test, y_train, y_test = pc.get_classification_data()
    print X_train.shape, X_test.shape
    model = nn(m)
    get_save_results(X_train, X_test, y_train, y_test, model, description)

if __name__ == '__main__':
    load_data(15000, 'NN')
