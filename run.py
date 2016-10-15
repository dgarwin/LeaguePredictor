from PlayerCollection import PlayerCollection
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def fetch_data(size):
    pc = PlayerCollection(size=size)
    pc.get_players()


def get_best_model():
    pass


def graph_model_depth():
    pass


def graph_model_width():
    pass


def sns_triangle(matrix, title):
    sns.set(style="white")
    # Generate a mask for the upper triangle
    mask = np.zeros_like(matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(matrix, mask=mask, cmap=cmap, vmax=.3,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.title(title)
    plt.show()


def many_pairwise_correlations(size,print_top=10):
    pc = PlayerCollection(size=size)
    players, divisions = pc.get_raw_transform()
    corr_train = players.corr()
    columns = corr_train.columns.tolist()
    dah = [(x, y, corr_train.as_matrix()[x, y]) for x in xrange(len(players.columns)) for y in xrange(len(players.columns)) if
           x > y]
    dah = sorted(dah, key=lambda tup: -abs(tup[2]))
    for i in range(print_top):
        print dah[i][2], columns[dah[i][1]], columns[dah[i][0]]
    sns_triangle(corr_train, "Pairwise Correlations")


if __name__ == '__main__':
    many_pairwise_correlations(15000)

