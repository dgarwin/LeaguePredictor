from lol_api import lol_api
import unittest
class test_lol_api_methods(unittest.TestCase):
	def setUp(self):
		self.api = lol_api()
	def test_get_players_recent_games(self):
		try:
			self.api.get_players_recent_games(20649224)
		except Exception, e:
			self.fail(e)

if __name__ == '__main__':
	unittest.main()
