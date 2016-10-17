from PlayerCollection import PlayerCollection
import numpy as np
import seaborn as sns
from pylab import pie, title, figure, axes, savefig, subplots, xlabel, ylabel, plot, legend


def class_distributions():
    # Create the Class Distributions Diagram
    labels = ['Diamond', 'Platinum', 'Gold', 'Silver', 'Bronze']
    fracs = [1.89, 8.05, 23.51, 38.96, 27.59]
    figure(1, figsize=(6,6))
    ax = axes([0.1, 0.1, 0.8, 0.8])
    pie(fracs, labels=labels, autopct='%1.1f%%')
    title('Tier Population Distribution', bbox={'facecolor': '0.8', 'pad': 5})
    savefig('images/pie.png')


def sns_triangle(matrix, plt_title, only_class=None):

    sns.set(style="white")
    # Generate a mask for the upper triangle
    mask = np.zeros_like(matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(matrix.as_matrix(), mask=mask, cmap=cmap, vmax=.3,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    title(plt_title)
    xlabel('Preprocessed Features')
    ylabel('Preprocessed Features')
    if only_class is None:
        only_class = ''
    savefig('images/triangle'+only_class+'.png')


def many_pairwise_correlations(size, print_top=10, only_class=None):
    # Create the correlation heat map triangle diagram
    pc = PlayerCollection(size=size)
    players, divisions = pc.get_raw_transform(only_class=only_class)
    corr_train = players.corr()
    columns = corr_train.columns.tolist()
    dah = [(x, y, corr_train.as_matrix()[x, y])
           for x in xrange(len(players.columns))
           for y in xrange(len(players.columns))
           if x > y]
    dah = sorted(dah, key=lambda tup: -abs(tup[2]))
    for i in range(print_top):
        print dah[i][2], columns[dah[i][1]], columns[dah[i][0]]
    tit = "Pairwise Correlations "
    if only_class is not None:
        tit += only_class
    sns_triangle(corr_train, tit, only_class)


def feature_count_sequence(count, model_func, division_dummies=False):
    # Create feature count sequence diagram
    pc = PlayerCollection(size=count)
    train_score = []
    test_score = []
    features = []
    percentiles = [10, 25, 50, 75, 90, 100]
    for p in percentiles:
        X_train, X_test, y_train, y_test = pc.get_classification_data(division_dummies=division_dummies, percentile=p)
        model = model_func(X_train.shape[1])
        model.fit(X_train, y_train)
        test_score.append(model.score(X_test, y_test))
        train_score.append(model.score(X_train, y_train))
        features.append(X_train.shape[1])
    print features
    print test_score
    print train_score
    plot(features, train_score, label='Train')
    plot(features, test_score, label='Test ')
    legend(loc='upper right')
    title('Accuracy as Feature Count Increases')
    ylabel('Accuracy')
    xlabel('Feature count')
    savefig('images/featureSequence.png')

if __name__ == '__main__':
    many_pairwise_correlations(15000)


