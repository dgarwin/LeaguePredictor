from urllib2 import Request, urlopen, URLError
import json
from datetime import datetime
from collections import deque
import time
import numpy as np
import pandas as pd
from lol_api import lol_api
class player_collection():
	def __init__(self, size = 10000):
		self.size = size
		self.raw = {}
		self.division_counts = {'SILVER': 0 }#init to avoid div by zero errors
		self.ignore = ['item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'team']
		self.categorical = ['playerPosition', 'playerRole']

	def need(self, player_stats):
		division = 'SILVER'#TODO fix
		return self.division_counts[division] < self.size / len(self.division_counts)

	def add(self, player_id, player_stats):
		pass

	def full(self):
		if len(self.raw) >= self.size:
			per_div_min = self.size / len(self.division_counts)
			for _, v in self.division_counts:
				if v < per_div_min:
					return False
			return True
		return False

	def get_player_stats(self, player_id, recent_games):
		games = pd.DataFrame([v['stats'] for v in recent_games['games'] if v['gameMode'] == 'CLASSIC'])
		num_games = len(games)
		games = games.fillna(0)#Fill before getting dummies
		games = pd.get_dummies(games, columns=self.categorical)
		games = games.drop(self.ignore, axis=1)
		games = games.sum(axis=0)
		games = games.divide(num_games)
		return games
		

def get_player_data(player_id, players, api):
	if player_id in players:
		return {}, []
	recent_games = api.recent_games(player_id)
	player_stats = players.get_player_stats(player_id, recent_games)
	next_ids = players.get_player_ids(recent_games)
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
