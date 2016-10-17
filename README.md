# LeaguePredictor
The core libraries used are:
Keras
Sklearn
Seaborn (for visualizations)
XGBoost

To load experimental data used (for all but MLP), run:
`X_train, X_test, y_train, y_test = PlayerCollection().get_classification_data(division_dummies=False)`
To load NN dummy data, run:
`X_train, X_test, y_train, y_test = PlayerCollection().get_classification_data(division_dummies=True)`
(The difference is only in whether or not the output classes are one-hot encoded or not.)

See test_model.py for how the four classifier experiments were executed.

The data can be obtained (~114 MB) here: https://drive.google.com/file/d/0B-OcR_i9yty1N3Z3LUNqS25YeTQ/view?usp=sharing