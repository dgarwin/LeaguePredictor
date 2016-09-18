from collections import deque
from PlayerCollection import PlayerCollection
from LolApi import LolApi


def get_player_data(player_id, players, api):
    if player_id in players.raw:
        return {}, []
    recent_games = api.recent_games(player_id)
    player_stats, valid = players.get_player_stats(recent_games)
    if not valid:  # No valid games found
        return {}, []
    next_ids = players.get_player_ids(recent_games)
    return player_stats, next_ids


# main logic loop
def get_players(seed_player_id, max_players=10000):
    players = PlayerCollection(max_players)
    queue = deque([seed_player_id])
    api = LolApi()
    while not players.full() and len(queue) > 0:
        buff = {}
        while len(buff) < 10 and len(queue) > 0:
            try:
                player_id = queue.pop()
                print str(1 + len(players.raw)) + ' ' + str(player_id)
                player_stats, next_ids = get_player_data(player_id, players, api)
                buff[player_id] = (player_stats, next_ids)
            except Exception, e:
                print 'Bad player: ' + str(player_id) + str(e)
        solo_divisions = api.solo_divisions(buff.keys())
        for player_id, data in buff.iteritems():
            player_stats, next_ids = data
            if player_id in solo_divisions:
                division = solo_divisions[player_id]
                if players.need(player_id, division):
                    players.add(player_id, division, player_stats)
                    queue.extend(next_ids)
        buff.clear()
    return players


if __name__ == '__main__':
    print get_players(20886270, 10).raw
