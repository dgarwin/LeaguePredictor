from urllib2 import Request, urlopen, URLError
import json
from datetime import datetime
from collections import deque
import time
import numpy as np
import pandas as pd
from lol_api import lol_api
class player_collection():
#Setup
	def __init__(self, size = 10000):
		self.size = size
		self.raw = {}
		self.division_counts = {'CHALLENGER': [], 'MASTER': [], 'DIAMOND': [], 'PLATINUM': [], 'GOLD': [], 'SILVER': [], 'BRONZE': [], 'UNRANKED': [] }#init to avoid div by zero errors
		self.per_div_min = self.size / len(self.division_counts)
		self.ignore = ['item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'team']
		self.categorical = ['playerPosition', 'playerRole']
#Do we need more in this division?
	def need(self, division):
		return len(self.division_counts[division]) < self.size / len(self.division_counts)
#Add new player to collection
	def add(self, player_id, division, player_stats):
		self.raw[player_id] = player_stats
		self.division_counts[division].append(player_id)
#Get player ids from recent games
	def get_player_ids(self, recent_games):
		return [player['summonerId'] for game in recent_games['games'] for player in game['fellowPlayers'] ]
#Are we done?
	def full(self):
		if len(self.raw) >= self.size:
			return sum([len(v) > self.per_div_min for _,v in self.division_counts]) == len(self.division_counts)
		return False
#Get stats from recent games
	def get_player_stats(self, player_id, recent_games):
		games = pd.DataFrame([v['stats'] for v in recent_games['games'] if v['gameMode'] == 'CLASSIC'])
		num_games = len(games)
		games = games.fillna(0)#Fill before getting dummies
		games = pd.get_dummies(games, columns=self.categorical)
		games = games.drop(self.ignore, axis=1)
		games = games.sum(axis=0)
		games = games.divide(num_games)
		return games

#Get player data and ids
def get_player_data(player_id, players, api):
	if player_id in players:
		return {}, []
	recent_games = api.recent_games(player_id)
	player_stats = players.get_player_stats(player_id, recent_games)
	next_ids = players.get_player_ids(recent_games)
	return player_stats, next_ids
#main logic loop
def get_players(seed_player_id, max_players = 10000):
	players = player_collection(max_players)
	queue = deque(seed_player_id)
	api = lol_api()
	while not players.full() and len(queue) > 0:
		buff = {}
		while len(buff) < 10 and len(queue) > 0:
			try:
				player_id = queue.pop()
				player_stats, next_ids = get_player_data(player_id)
				buff[player_id] = (player_stats, next_ids)
			except Exception, e:
				print 'Bad player: ' + str(player_id)
		for player_id, division in api.solo_division(buff.keys()):			
			if players.need(division):
				players.add(player_id, divison, buff[player_id][0])
				queue.extend([x[1] for x in buff[player_id]])
	return players
