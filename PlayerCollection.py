import pandas as pd
import numpy as np

# TODO: Ignore items to ignore before processing


class PlayerCollection():
    ignore = ['item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'team']
    categorical = ['playerPosition', 'playerRole']
    distributions = {'CHALLENGER': 0.0002, 'MASTER': 0.0004, 'DIAMOND': 0.0183, 'PLATINUM': 0.0805, 'GOLD': 0.2351,
                     'SILVER': 0.3897, 'BRONZE': 0.2759}

    # Setup
    def __init__(self, api, size=10000):
        self.api = api
        self.size = size
        self.raw = {}
        self.division_counts = {'CHALLENGER': [], 'MASTER': [], 'DIAMOND': [], 'PLATINUM': [], 'GOLD': [], 'SILVER': [],
                                'BRONZE': []}  # init to avoid div by zero errors
        self.per_div_min = self.size / len(self.division_counts)

    def save(self):
        np.save('players.npy', self.raw)
        np.save('division_counts.npy', self.division_counts)

    # Did we see this player already?
    def has(self, player_id):
        return player_id in self.raw

    # Do we need this player?
    def need(self, player_id, division):
        return division in self.distributions and \
               player_id not in self.raw and \
               len(self.division_counts[division]) < 1.0 * self.size / len(self.division_counts)

    # Add new player to collection. Return next_ids for convenience.
    def add(self, player_id, division, data):
        if not self.need(player_id, division):
            return []
        player_stats, next_ids = data
        self.raw[player_id] = player_stats
        self.division_counts[division].append(player_id)
        return next_ids

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
        games = games.fillna(0)  # Fill before getting
        try:
            games = pd.get_dummies(games, columns=self.categorical)
        except ValueError:  # Ignore the column for now
            pass
        # games = games.drop(self.ignore, axis=1)
        games = games.mean(axis=0)
        return games, True

    def get_player_data(self, player_id):
        if player_id in self.raw:
            return {}, []
        recent_games = self.api.recent_games(player_id)
        player_stats, valid = self.get_player_stats(recent_games)
        if not valid:  # No valid games found
            return {}, []
        next_ids = self.get_player_ids(recent_games)
        return player_stats, next_ids
