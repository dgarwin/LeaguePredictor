from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd


def get_save_results(X_train, X_test, y_train, y_test, model, description):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    write = description + '\n'
    if hasattr(model, 'best_params_'):
        write += 'Best params: ' + str(model.best_params_) + '\n'
    write += 'Training Score' + str(model.score(X_train, y_train)) + '\n'
    write += 'Testing Score: ' + str(model.score(X_test, y_test)) + '\n'
    if description == 'NN':
        y_test = pd.DataFrame(y_test).stack()
        y_test = pd.Series(pd.Categorical(y_test[y_test != 0].index.get_level_values(1)))
    write += str(classification_report(y_test, predictions)) + '\n'
    write += str(confusion_matrix(y_test, predictions)) + '\n'
    print write
    with open('experiments', 'a') as f:
        f.write(write)
    return model