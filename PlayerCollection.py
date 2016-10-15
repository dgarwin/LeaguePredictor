import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
from LolApi import LolApi
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter


# TODO: Ignore items to ignore before processing


class PlayerCollection():
    ignore = ['item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'championId', 'ipEarned']
    # Ignore champId for now
    categorical = ['playerPosition', 'playerRole']
    distributions = {'CHALLENGER': 0, 'MASTER': 0, 'DIAMOND': 0.0189, 'PLATINUM': 0.0805, 'GOLD': 0.2351,
                     'SILVER': 0.3896, 'BRONZE': 0.2759, 'UNRANKED': 0}
    players_prefix = 'players_'
    matches_prefix = 'matches_'
    player_matches_prefix = 'player_matches_'
    player_top_champions_prefix = 'top_champions_'
    directory = 'npy'
    WRITE_EVERY = 100
    BUFF_SIZE = 10

    # Setup
    def __init__(self, api=None, size=10000, load=True):
        if api is None:
            api = LolApi()
        self.api = api
        self.size = size
        self.raw = {}
        self.matches = {}
        self.player_matches = {}
        self.top_champions = {}
        self.suffix = str(size)
        self.per_div_min = {}
        for key, value in self.distributions.iteritems():
            self.per_div_min[key] = round(self.size * value)
        if load:
            self.load()

    # Save/load helpers
    def suffix_to_filename(self, prefix):
        return self.directory + '/' + prefix + self.suffix + '.npy'

    def get_collection_tuples(self):
        return [(self.raw, self.players_prefix),
                (self.player_matches, self.player_matches_prefix),
                (self.matches, self.matches_prefix),
                (self.top_champions, self.player_top_champions_prefix)]

    def save(self):
        collections = self.get_collection_tuples()
        for tup in collections:
            if len(tup[0]) > 0:
                np.save(self.suffix_to_filename(tup[1]), tup[0])

    def load(self):
        collections = self.get_collection_tuples()
        for collection in collections:
            filename = self.suffix_to_filename(collection[1])
            try:
                current = np.load(filename)
                collection[0].update(current.tolist())
            except IOError:
                print 'Could not find: {0}'.format(filename)

    def save_add(self, collection, elements, start_time):
        collection.update(elements)
        if len(collection) % self.WRITE_EVERY == 0:
            self.save()
            print '{0:3.2f} Saving {1} players' \
                .format((datetime.now() - start_time).total_seconds() / 60.0, len(collection))
            return True
        return False

    # Get top champs
    def get_top_champions(self):
        now = datetime.now()
        for player_id in self.raw.keys():
            if player_id in self.top_champions:
                continue
            tc = self.api.top_champions(self.suffix)
            self.save_add(self.top_champions, {player_id: tc}, now)
        np.save(self.suffix_to_filename(self.player_top_champions_prefix), self.top_champions)

    # Get matches
    def get_matches(self):
        now = datetime.now()
        for i, player_id in enumerate(self.raw.keys()):
            if player_id in self.player_matches:
                continue
            recent_matches = [m['gameId'] for m in self.api.recent_games(player_id)['games']]
            matches = {}
            for match_id in recent_matches:
                matches[match_id] = self.api.get_match(match_id)
            self.save_add(self.matches, matches, now)
            self.save_add(self.player_matches,
                          {player_id: [match['match_id'] for match in matches]}, now)

    # INITIAL FETCH
    def full(self, division_counts):
        # Did we see this player already?
        # Are we done?
        if len(self.raw) >= self.size:
            for division in self.distributions:
                if division_counts[division] < self.per_div_min[division]:
                    return False
            return True
        return False

    # Do we need this player?
    def need(self, player_id, division, division_counts):
        return division in self.distributions and \
               player_id not in self.raw and \
               self.per_div_min[division] > division_counts[division]

    # Get stats from recent games
    @staticmethod
    def get_player_stats(recent_games):
        ret = []
        for game in recent_games['games']:
            if game['subType'] == 'RANKED_SOLO_5x5':
                to_add = game['stats']
                to_add['ipEarned'] = game['ipEarned']
                to_add['championId'] = game['championId']
                ret.append(to_add)
        return ret

    def seed_queue(self):
        games = self.api.featured_games()
        summoner_names = [p['summonerName'] for game in games['gameList'] for p in game['participants']
                          if game['gameMode'] == 'CLASSIC' and game['gameType'] == 'MATCHED_GAME']
        ret = []
        for i in range(0, len(summoner_names), 10):
            id_map = self.api.summoner_ids(summoner_names[i:i + 10])
            ret.extend([p['id'] for p in id_map.values()])
        return ret

    # main logic loop
    def get_players(self):
        now = datetime.now()
        queue = deque(self.seed_queue())
        seen = set(self.raw.keys())
        division_counts = dict.fromkeys(self.distributions.keys(), 0)
        if len(self.raw) > 0:
            counts = Counter([x[1] for x in self.raw.values()])
            for k, v in counts.iteritems():
                division_counts[k] = v
        print division_counts
        print self.per_div_min
        # Get at least max_players players
        while not self.full(division_counts) and len(queue) > 0:
            # Fill buffer with not seen players
            buff = set()
            while len(buff) < self.BUFF_SIZE and len(queue) > 0:
                player_id = queue.pop()
                if player_id not in seen:
                    buff.add(player_id)
                    seen.add(player_id)
            # Get player's divisions
            try:
                solo_divisions = self.api.solo_divisions(buff)
            except Exception, e:
                print 'Could not get solo divisions.', e, buff
                continue
            # Get stats for needed players
            for player_id in buff:
                division = solo_divisions[player_id]
                if not self.need(player_id, division, division_counts):
                    continue
                try:
                    recent_games = self.api.recent_games(player_id)
                    player_stats = self.get_player_stats(recent_games)
                    if len(player_stats) >= 6:
                        self.save_add(self.raw, {player_id: (player_stats, division)}, now)
                        division_counts[division] += 1
                    if len(player_stats) > 0:
                        next_ids = [player['summonerId'] for game in recent_games['games'] for player in
                                    game.get('fellowPlayers', [])]
                        queue.extend(next_ids)
                except Exception, e:
                    print 'Bad player: ' + str(player_id) + str(e)
                    continue
        self.save()

    @staticmethod
    def transform_game(data, player_id, division):
        game = data.copy()
        game['playerId'] = player_id
        game['division'] = division
        return game

    def get_raw_transform(self):
        for id in self.raw.keys():
            d = self.raw[id][1]
            if d == 'MASTER' or d == 'CHALLENGER':
                del self.raw[id]
        feature_vectors = [self.transform_game(game, player_id, self.raw[player_id][1])
                           for player_id in self.raw
                           for game in self.raw[player_id][0]]
        df = pd.DataFrame(feature_vectors)
        df = df.fillna(0)
        df = df.drop(PlayerCollection.ignore, axis=1)
        df = pd.get_dummies(df, columns=PlayerCollection.categorical)

        grouped = df.groupby(['playerId', 'division'])
        players = grouped.aggregate([np.mean, np.std])
        divisions = pd.DataFrame(players.index.tolist())[1]
        return players, divisions

    def get_undivided_classification_data(self, samples=None):
        players, divisions = self.get_raw_transform()
        divisions = divisions.as_matrix()
        players = players.as_matrix()
        if samples is not None:
            players = players[0:samples, :]
            divisions = divisions[0:samples]
        return players, divisions

    def get_classification_data(self, division_dummies=True, samples=None):
        players, divisions = self.get_undivided_classification_data(samples)
        X_train, X_test, y_train, y_test = train_test_split(
            players, divisions, random_state=42, stratify=divisions)
        if division_dummies:
            y_train = pd.get_dummies(y_train).as_matrix()
            y_test = pd.get_dummies(y_test).as_matrix()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
