from urllib2 import Request, urlopen, URLError
import json
from datetime import datetime
from collections import deque
import time

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
