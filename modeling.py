from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from PlayerCollection import PlayerCollection


def get_save_results(X_train, X_test, y_train, y_test, model, description, params=None):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    write = description + '\n'
    if hasattr(model, 'best_params_'):
        write += 'Best params: ' + str(model.best_params_) + '\n'
    if params:
        write += 'Params: ' + str(params) + '\n'
    write += 'Training Score: ' + str(model.score(X_train, y_train)) + '\n'
    write += 'Testing Score: ' + str(model.score(X_test, y_test)) + '\n'
    if description == 'NN':
        y_test = pd.DataFrame(y_test).stack()
        y_test = pd.Series(pd.Categorical(y_test[y_test != 0].index.get_level_values(1)))
    write += str(classification_report(y_test, predictions)) + '\n'
    write += str(confusion_matrix(y_test, predictions)) + '\n'
    print write
    with open('notes/experiments', 'a') as f:
        f.write(write)
    return model


# As Training Data is added (all best models)
# As model grows (RF, XGB, NN)
# Compare best models (Table)

def sequence_graph(model_func, params, sequence, sequence_name, sequence_display, title, division_dummies, pc):
    test_accuracys = []
    train_accuracys = []
    X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=division_dummies)
    for s in sequence:
        params[sequence_name] = s
        model = model_func(**params)
        model.fit(X_train, y_train)
        train_accuracys.append(model.score(X_train, y_train))
        test_accuracys.append(model.score(X_test, y_test))
    for test_accuracy, train_accuracy, s in zip(test_accuracys, train_accuracys, sequence):
        plt.annotate('{:0.2f}'.format(test_accuracy), (s, test_accuracy))
        plt.annotate('{:0.2f}'.format(train_accuracy), (s, train_accuracy))
    plt.scatter(sequence, test_accuracys)
    plt.scatter(sequence, train_accuracys)
    plt.plot(sequence, test_accuracys, label='Test')
    plt.plot(sequence, train_accuracys, label='Train')
    plt.legend(loc='best')
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel(sequence_display)
    plt.show()


def rf_estimators_progression(count):
    rf_params = {'random_state': 42}
    player_collection = PlayerCollection(size=count)
    sequence_graph(RandomForestClassifier, rf_params, [10, 20, 40, 80, 100, 160, 320, 460, 640],
                   'n_estimators', 'Number of Trees', 'Random Forest', False, player_collection)


def xgb_estimators_progression(count):
    xgb_params = {'seed': 42, 'nthread': 4}
    player_collection = PlayerCollection(size=count)
    sequence_graph(XGBClassifier, xgb_params, [10, 20, 40, 80, 100, 160, 320, 460, 640],
                   'n_estimators', 'Number of Trees', 'XGBoost', False, player_collection)
if __name__ == '__main__':
    xgb_estimators_progression(15000)
