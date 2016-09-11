from urllib2 import Request, urlopen, URLError
import json
from datetime import datetime
from collections import deque
import time
class lol_api():
	def __init__(self):
		self.base = 'https://na.api.pvp.net'
		self.apl_key = '?api_key=RGAPI-B110C515-1426-43EA-ABD8-F04D6169C9C1' 
		self.region = 'NA'
		self.short_req = deque(0)
		self.long_req = deque(0)
	def get_players_recent_games(self, player_id):
		url = self.base + '/api/lol/NA/v1.3/game/by-summoner/' + player_id + '/recent'
		return self.request(url)

	def delay(self, queue, interval, requests):
		now = datetime.now()
		cur = (now - queue.popleft()).total_seconds()
		while len(queue) > 0 and cur < interval:
			cur = queue.popleft()
		if len(queue) >= request:
			wait_time = queue[len(queue) - requests - 1]
			time.sleep((now - wait_time).total_seconds())
	def request(self, url):
		#make a request within the rate limit. This could be done in a more async way, yolo
		delay(self.short_req, 10, 10)
		delay(self.short_req, 600, 500)
		while True:
			try:
				request = Request(url)
				response = urlopen(request)
				break
			except URLError e:
				time.sleep(10)
		now = datetime.now()
		self.short_req.append(now)
		self.long_req.append(now)
		return json.loads(response.read())

class player_collection():
	def __init__(self, size):
		self.size = size
		self.raw = {}
		self.division_counts = {'SILVER': 0 }#init to avoid div by zero errors
	def need(self, player_stats):
		division = 'SILVER'#TODO fix
		return self.division_counts[division] < self.size / len(self.division_counts)
	def add(self, player_id, player_stats):
		#TODO fill
	def full(self):
		if len(self.raw) >= self.size:
			per_div_min = self.size / len(self.division_counts)
			for _, v in self.division_counts:
				if v < per_div_min:
					return False
			return True
		return False

def get_player_data(player_id, players, api):
	if seed_player_id in players:
		return []
	recent_games = api.get_recent_games(seed_player_id)
	player_stats = get_player_stats(seed_player_id, recent_games)
	next_ids = get_player_ids(recent_games)
	return player_stats, next_ids

def get_players(seed_player_id):
	players = player_collection()
	queue = deque(seed_player_id)
	api = lol_api()
	while not players.full():
		player_id = queue.pop()
		player_stats, next_ids = get_player_data(player_id)
		if players.need(player_stats):
			players.add(player_id, player_stats)
			#Only add reachable players if you need them (silver plays mostly with silver)
			for pid in next_ids:
				queue.append(pid)
	return players
