import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
from LolApi import LolApi

WRITE_EVERY = 100
BUFF_SIZE = 10


# TODO: Ignore items to ignore before processing


class PlayerCollection():
    ignore = ['item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'team']
    categorical = ['playerPosition', 'playerRole']
    distributions = {'CHALLENGER': 0.0002, 'MASTER': 0.0004, 'DIAMOND': 0.0183, 'PLATINUM': 0.0805, 'GOLD': 0.2351,
                     'SILVER': 0.3896, 'BRONZE': 0.2759, 'UNRANKED': 0}
    players_prefix = 'players_'
    division_counts_prefix = 'division_counts_'
    matches_prefix = 'matches_'
    player_matches_prefix = 'player_matches_'
    player_top_champions_prefix = 'top_champions_'
    directory = 'npy'

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
        self.division_counts = {'CHALLENGER': [], 'MASTER': [], 'DIAMOND': [], 'PLATINUM': [], 'GOLD': [], 'SILVER': [],
                                'BRONZE': [], 'UNRANKED': []}  # init to avoid div by zero errors
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
                (self.division_counts, self.division_counts_prefix),
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

    @staticmethod
    def get_player_ids(recent_games):
        return [player['summonerId'] for game in recent_games['games'] for player in game.get('fellowPlayers', [])]

    def save_add(self, collection, elements, start_time):
        collection.update(elements)
        if len(collection) % WRITE_EVERY == 0:
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
    def full(self):
        # Did we see this player already?
        # Are we done?
        if len(self.raw) >= self.size:
            for division in self.distributions:
                if self.need(0, division):
                    return False
            return True
        return False

    def has(self, player_id):
        return player_id in self.raw

    # Do we need this player?
    def need(self, player_id, division):
        return division in self.distributions and \
               player_id not in self.raw and \
               self.per_div_min[division] > len(self.division_counts[division])

    # Add new player to collection. Return next_ids for convenience.
    def add(self, player_id, division, data):
        player_stats, next_ids = data
        if not self.need(player_id, division):
            next_ids = []  # If we don't need this player, don't look at neighboring players
        self.raw[player_id] = player_stats
        self.division_counts[division].append(player_id)
        return next_ids

    # Get stats from recent games
    def get_player_stats(self, recent_games):
        games = pd.DataFrame([v['stats'] for v in recent_games['games'] if v['gameMode'] == 'CLASSIC'])
        if games.empty:
            return games, False
        try:
            games = pd.get_dummies(games, columns=self.categorical)
        except ValueError:  # Ignore the column for now
            pass
        # games = games.drop(self.ignore, axis=1)
        games = games.mean(axis=0).to_dict()
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

    def get_buffered_divisions(self, queue):
        player_set = {}
        while len(player_set) < BUFF_SIZE and len(queue) > 0:
            player_id = queue.pop()
            if player_id not in player_set and not self.has(player_id):
                player_set[player_id] = {}
        # Get solo divs from buffer
        solo_divisions = self.api.solo_divisions(player_set.keys())
        for player_id in player_set:
            if not self.need(player_id, solo_divisions[player_id]):
                del solo_divisions[player_id]
        return solo_divisions

    # main logic loop
    def get_players(self, seed_player_id):
        if len(self.raw) > 0:
            raise "Fetching players from api after load is forbidden"
        now = datetime.now()
        queue = deque([seed_player_id])
        buff = {}
        solo_divisions = {}

        # Get at least max_players players
        while not self.full() and len(queue) > 0:
            # Get some players we need based on division
            while len(solo_divisions) == 0:
                solo_divisions = self.get_buffered_divisions(queue)
            # Get player data
            for player_id in solo_divisions.keys():
                try:
                    buff[player_id] = self.get_player_data(player_id)
                except Exception, e:
                    print 'Bad player: ' + str(player_id) + str(e)
            # Save player data
            for player_id, data in buff.iteritems():
                if len(self.raw) % WRITE_EVERY == 0:
                    self.save(self.suffix)
                    print '{0:3.2f} Saving {1} players'.format((datetime.now() - now).total_seconds() / 60.0,
                                                               len(players.raw))
                division = solo_divisions[player_id]

                queue.extend(self.add(player_id, division, data))
            # Refresh buffer
            buff.clear()
            solo_divisions.clear()
        self.save(self.suffix)
