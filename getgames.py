from collections import deque
from PlayerCollection import PlayerCollection
from LolApi import LolApi
import progressbar


# main logic loop
def get_players(seed_player_id, max_players=10000):
    api = LolApi()
    players = PlayerCollection(max_players, api)
    queue = deque([seed_player_id])
    buff = {}
    last_progress = 0
    with progressbar.ProgressBar(max_value=100) as progress:
        # Progress bar
        current_progress = 100 * len(players.raw) / max_players
        if  current_progress > last_progress:
            progress.update(current_progress)
            players.save()
        # Get at least max_players players
        while not players.full() and len(queue) > 0:
            for i in range(10):
                if len(queue) == 0:
                    break
                try:
                    player_id = queue.pop()
                    if player_id not in buff and not players.has(player_id):
                        buff[player_id] = players.get_player_data(player_id)
                except Exception, e:
                    print 'Bad player: ' + str(player_id) + str(e)
            # Get solo divs from buffer
            solo_divisions = api.solo_divisions(buff.keys())
            # Add as needed
            for player_id, data in buff.iteritems():
                division = solo_divisions[player_id]
                queue.extend(players.add(player_id, division, data))
            # Refresh buffer
            buff.clear()
    return players


if __name__ == '__main__':
    get_players(20649224, 10).raw
