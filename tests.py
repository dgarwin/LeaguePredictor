import unittest
from collections import deque
from datetime import datetime, timedelta

from getgames import PlayerCollection
from lol_api import LolApi


class TestLolApi(unittest.TestCase):
    def setUp(self):
        self.api = LolApi()
        self.player_id = 20649224

    def test_solo_division(self):
        val = self.api.solo_division([self.player_id])
        self.assertEqual(1, len(val))
        self.assertEqual('GOLD', val[self.player_id])

    def test_get_players_recent_games(self):
        try:
            self.api.recent_games(self.player_id)
        except Exception, e:
            self.fail(e)


class TestPlayerCollection(unittest.TestCase):
    def setUp(self):
        self.api = LolApi()
        self.players = PlayerCollection()
        self.player_id = 20649224

    def test_delay_time(self):
        now = datetime.now()
        interval = 600
        past = now - timedelta(seconds=interval)
        long_past = now - timedelta(seconds=interval * 2)
        queue = deque([long_past, past])
        delay_time = self.api.delay_time(queue, interval, 1, now)
        self.assertEqual(1, len(queue))
        self.assertEqual(interval, delay_time)
        delay_time_2 = self.api.delay_time(queue, interval, 2, now)
        self.assertEqual(0, delay_time_2)

    def test_get_player_ids(self):
        recent_games = self.api.recent_games(self.player_id)
        ids = self.players.get_player_ids(recent_games)
        self.assertGreater(len(ids), 0)

    def test_get_player_stats(self):
        recent_games = self.api.recent_games(self.player_id)
        player_stats = self.players.get_player_stats(self.player_id, recent_games)
        self.assertGreater(player_stats['playerRole_3.0'], 0.5)


if __name__ == '__main__':
    unittest.main()
