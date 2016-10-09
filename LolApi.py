from urllib2 import Request, urlopen, URLError
import json
from datetime import datetime
from collections import deque
import time
from datetime import timedelta


class LolApi:
    def __init__(self, api_key='RGAPI-B110C515-1426-43EA-ABD8-F04D6169C9C1'):
        self.base = 'https://na.api.pvp.net'
        self.api_key = '?api_key=' + api_key
        self.region = 'na'
        self.short_req = deque([datetime(2000, 1, 1)])
        self.long_req = deque([datetime(2000, 1, 1)])

    def solo_divisions_regression(self, player_ids):
        tier_to_int = {'BRONZE': 0, 'SILVER': 1, 'GOLD': 2, 'PLATINUM': 3, 'DIAMOND': 4, 'MASTER': 5, 'CHALLENGER': 6}
        roman_to_int = {'V': 0, 'IV': 0.2, 'III': 0.4, 'II': 0.6, 'I': 0.8}
        url = self.base + '/api/lol/' + self.region + '/v2.5/league/by-summoner/' + ','.join(
            [str(p) for p in player_ids]) + self.api_key
        players = self.request(url)
        ret = {}
        for player_id in player_ids:
            if not str(player_id) in players:
                ret[player_id] = -1
                continue
            list_ldto = players[str(player_id)]
            ldtos = [ldto for ldto in list_ldto if ldto['queue'] == 'RANKED_SOLO_5x5']
            if len(ldtos) == 0:
                ret[player_id] = -1
                continue
            else:
                rank = ldtos[0]['tier']
                division = [entry['division']
                            for entry in ldtos[0]['entries']
                            if entry['playerOrTeamId'] == str(player_id)]
                ret[player_id] = tier_to_int[rank] + roman_to_int[division[0]]
        return ret

    def top_champions(self, player_id, count=10):
        url = self.base + '/championmastery/location/NA1/player/' + str(player_id) + \
              '/topchampions?count=' + str(count) + self.api_key.replace('?', '&')
        return self.request(url)

    def recent_games(self, player_id):
        url = self.base + '/api/lol/NA/v1.3/game/by-summoner/' + str(player_id) + '/recent' + self.api_key
        return self.request(url)

    def get_match(self, match_id):
        url = self.base + '/api/lol/' + self.region + '/v2.2/match/' + str(match_id) + self.api_key
        return self.request(url, 5)

    def featured_games(self):
        url = self.base + '/observer-mode/rest/featured' + self.api_key
        return self.request(url)

    def summoner_ids(self, names):
        names = [name.replace(' ', '') for name in names]
        url = self.base + '/api/lol/'+self.region+'/v1.4/summoner/by-name/' + ','.join(names) + self.api_key
        return self.request(url)

    def solo_divisions(self, player_ids):
        url = self.base + '/api/lol/' + self.region + '/v2.5/league/by-summoner/' + ','.join(
            [str(p) for p in player_ids]) + self.api_key
        players = self.request(url)
        ret = {}
        for player_id in player_ids:
            if not str(player_id) in players:
                ret[player_id] = 'UNRANKED'
                continue
            list_ldto = players[str(player_id)]
            rank = [ldto['tier'] for ldto in list_ldto if ldto['queue'] == 'RANKED_SOLO_5x5']
            if len(rank) == 0:
                ret[player_id] = 'UNRAKED'
            else:
                ret[player_id] = rank[0]
        return ret

    @staticmethod
    def delay_time(queue, interval, requests, now):
        while len(queue) > 0 and len(queue) >= requests:
            delta = (queue[0] + timedelta(0, interval) - now).total_seconds()
            # First pop old off, then check if the queue is too long, then return
            if delta < 0:
                queue.popleft()
            else:
                return delta
        return 0

    def request(self, url, min_wait=0):
        for try_count in range(10):
            now = datetime.now()
            short_delay = self.delay_time(self.short_req, 10, 10, now)
            long_delay = self.delay_time(self.long_req, 500, 500, now)
            wait_time = max([long_delay, short_delay, min_wait])
            time.sleep(wait_time)
            try:
                return self.request_data(url)
            except URLError, e:
                print e.reason, datetime.now()
                time.sleep(10)
        raise Exception("Invalid URL.")

    def request_data(self, url):
        request = Request(url)
        response = urlopen(request)
        now = datetime.now()
        self.short_req.append(now)
        self.long_req.append(now)
        return json.loads(response.read())
