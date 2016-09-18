from urllib2 import Request, urlopen, URLError
import json
from datetime import datetime
from collections import deque
import time


class LolApi:
    def __init__(self, api_key='RGAPI-B110C515-1426-43EA-ABD8-F04D6169C9C1'):
        self.base = 'https://na.api.pvp.net'
        self.api_key = '?api_key=' + api_key
        self.region = 'na'
        self.short_req = deque([datetime(2000, 1, 1)])
        self.long_req = deque([datetime(2000, 1, 1)])

    def recent_games(self, player_id):
        url = self.base + '/api/lol/NA/v1.3/game/by-summoner/' + str(player_id) + '/recent' + self.api_key
        return self.request(url)

    def solo_divisions(self, player_ids):
        url = self.base + '/api/lol/' + self.region + '/v2.5/league/by-summoner/' + ','.join(
            [str(p) for p in player_ids]) + self.api_key
        players = self.request(url)
        ret = {}
        for player_id, list_ldto in players.iteritems():
            rank = [ldto['tier'] for ldto in list_ldto if ldto['queue'] == 'RANKED_SOLO_5x5']
            if len(rank) == 0:
                ret[int(player_id)] = 'UNRAKED'
            else:
                ret[int(player_id)] = rank[0]
        return ret

    @staticmethod
    def delay_time(queue, interval, requests, now):
        while len(queue) > 0 and (now - queue[0]).total_seconds() > interval:
            queue.popleft()
        if len(queue) >= requests:
            # wait a while
            return (now - queue[0]).total_seconds()
        else:
            return 0

    def request(self, url):
        # make a request within the rate limit. This could be done in a more async way, yolo
        now = datetime.now()
        short_delay = self.delay_time(self.short_req, 10, 10, now)
        long_delay = self.delay_time(self.long_req, 600, 500, now)
        time.sleep(max([long_delay, short_delay]))
        for try_count in range(10):
            try:
                request = Request(url)
                response = urlopen(request)
                break
            except URLError, e:
                if e.code == 429:
                    print 'Too many requests'
                    time.sleep(10)
                else:
                    raise e
        now = datetime.now()
        self.short_req.append(now)
        self.long_req.append(now)
        return json.loads(response.read())
