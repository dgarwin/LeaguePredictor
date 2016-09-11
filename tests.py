from lol_api import lol_api
from collections import deque
from getgames import player_collection
import unittest
class test_lol_api(unittest.TestCase):
	def setUp(self):
		self.api = lol_api()
	def test_get_players_recent_games(self):
		try:
			self.api.recent_games(20649224)
		except Exception, e:
			self.fail(e)
class test_player_collection(unittest.TestCase):
	def setUp(self):
		self.api = lol_api()
		self.players = player_collection()
	def test_get_player_stats(self):
		api = lol_api()
		player_id = 20649224
		recent_games = api.recent_games(player_id)
		player_stats = self.players.get_player_stats(player_id, recent_games)
		self.assertGreater(player_stats['playerRole_3.0'], 0.5)
if __name__ == '__main__':
	unittest.main()
