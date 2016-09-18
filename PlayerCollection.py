import pandas as pd


# TODO: Ignore items to ignore before processing


class PlayerCollection():
    # Setup
    def __init__(self, size=10000):
        self.size = size
        self.raw = {}
        self.division_counts = {'CHALLENGER': [], 'MASTER': [], 'DIAMOND': [], 'PLATINUM': [], 'GOLD': [], 'SILVER': [],
                                'BRONZE': [], 'UNRANKED': []}  # init to avoid div by zero errors
        self.per_div_min = self.size / len(self.division_counts)
        self.ignore = ['item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'team']
        self.categorical = ['playerPosition', 'playerRole']

    # Do we need more in this division?
    def need(self, player_id, division):
        return player_id not in self.raw and \
               len(self.division_counts[division]) < 1.0 * self.size / len(self.division_counts)

    # Add new player to collection
    def add(self, player_id, division, player_stats):
        self.raw[player_id] = player_stats
        self.division_counts[division].append(player_id)

    # Get player ids from recent games
    @staticmethod
    def get_player_ids(recent_games):
        return [player['summonerId'] for game in recent_games['games'] for player in game.get('fellowPlayers', [])]

    # Are we done?
    def full(self):
        if len(self.raw) >= self.size:
            return sum([len(v) > self.per_div_min for _, v in self.division_counts]) == len(self.division_counts)
        return False

    # Get stats from recent games
    def get_player_stats(self, recent_games):
        games = pd.DataFrame([v['stats'] for v in recent_games['games'] if v['gameMode'] == 'CLASSIC'])
        if games.empty:
            return games, False
        games = games.fillna(0)  # Fill before getting dummies
        games = pd.get_dummies(games, columns=self.categorical)
        # games = games.drop(self.ignore, axis=1)
        games = games.mean(axis=0)
        return games, True
