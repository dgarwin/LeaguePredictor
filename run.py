from PlayerCollection import PlayerCollection


def fetch_data(size):
    pc = PlayerCollection(size=size)
    pc.get_players()


def get_best_model():
    pass


def graph_model_depth():
    pass


def graph_model_width():
    pass

if __name__ == '__main__':
    fetch_data(45000)
