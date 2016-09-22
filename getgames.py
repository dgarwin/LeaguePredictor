from collections import deque
from PlayerCollection import PlayerCollection
from LolApi import LolApi
WRITE_EVERY = 100
BUFF_SIZE = 10


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
    api = LolApi()
    players = PlayerCollection(api, max_players)
    queue = deque([seed_player_id])
    buff = {}
    solo_divisions = {}

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
                players.save()
            division = solo_divisions[player_id]

            queue.extend(players.add(player_id, division, data))
        # Refresh buffer
        buff.clear()
        solo_divisions.clear()
    return players


if __name__ == '__main__':
    get_players(20649224, 10).save()
