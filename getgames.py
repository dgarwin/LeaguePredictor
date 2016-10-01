from collections import deque
from PlayerCollection import PlayerCollection
from LolApi import LolApi
from datetime import datetime
import numpy as np

WRITE_EVERY = 100
BUFF_SIZE = 10


def load_masteries(suffix):
    return np.load('top_champions_' + suffix + '.npy')


def get_champion_masteries(count):
    now = datetime.now()
    api = LolApi()
    p = PlayerCollection(api, count)
    suffix = str(count)
    ret = load_masteries(suffix).tolist()
    p.load(suffix)
    for player_id in p.raw.tolist().keys():
        if player_id in ret:
            continue
        if len(ret) % WRITE_EVERY == 0:
            np.save('top_champions_' + suffix + '.npy', ret)
            print '{0:3.2f} Saving {1} players'.format((datetime.now()-now).total_seconds()/60.0, len(ret))
        tc = api.top_champions(str(count))
        ret[player_id] = tc
    np.save('top_champions_' + suffix + '.npy', ret)


def get_buffered_divisions(queue, players, api):
    player_set = {}
    while len(player_set) < BUFF_SIZE and len(queue) > 0:
        player_id = queue.pop()
        if player_id not in player_set and not players.has(player_id):
            player_set[player_id] = {}
    # Get solo divs from buffer
    solo_divisions = api.solo_divisions(player_set.keys())
    for player_id in player_set:
        if not players.need(player_id, solo_divisions[player_id]):
            del solo_divisions[player_id]
    return solo_divisions


# main logic loop
def get_players(seed_player_id, max_players=10000):
    now = datetime.now()
    api = LolApi()
    players = PlayerCollection(api, max_players)
    queue = deque([seed_player_id])
    buff = {}
    solo_divisions = {}
    suffix = str(max_players)

    # Get at least max_players players
    while not players.full() and len(queue) > 0:
        # Get some players we need based on division
        while len(solo_divisions) == 0:
            solo_divisions = get_buffered_divisions(queue, players, api)
        # Get player data
        for player_id in solo_divisions.keys():
            try:
                buff[player_id] = players.get_player_data(player_id)
            except Exception, e:
                print 'Bad player: ' + str(player_id) + str(e)
        # Save player data
        for player_id, data in buff.iteritems():
            if len(players.raw) % WRITE_EVERY == 0:
                players.save(suffix)
                print '{0:3.2f} Saving {1} players'.format((datetime.now()-now).total_seconds()/60.0, len(players.raw))
            division = solo_divisions[player_id]

            queue.extend(players.add(player_id, division, data))
        # Refresh buffer
        buff.clear()
        solo_divisions.clear()
    players.save(suffix)
    return players
