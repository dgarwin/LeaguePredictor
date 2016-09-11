from urllib2 import Request, urlopen, URLError
import json
from datetime import datetime
from collections import deque
import time
class lol_api():
	def __init__(self, api_key='RGAPI-B110C515-1426-43EA-ABD8-F04D6169C9C1'):
		self.base = 'https://na.api.pvp.net'
		self.api_key = '?api_key='+api_key 
		self.region = 'NA'
		self.short_req = deque([datetime(2000,1,1)])
		self.long_req = deque([datetime(2000,1,1)])
	def get_players_recent_games(self, player_id):
		url = self.base + '/api/lol/NA/v1.3/game/by-summoner/' + str(player_id) + '/recent' + self.api_key
		return self.request(url)

	def delay(self, queue, interval, requests):
		now = datetime.now()
		cur = (now - queue.popleft()).total_seconds()
		while len(queue) > 0 and cur < interval:
			cur = queue.popleft()
		if len(queue) >= requests:
			wait_time = queue[len(queue) - requests - 1]
			time.sleep((now - wait_time).total_seconds())

	def request(self, url):
		#make a request within the rate limit. This could be done in a more async way, yolo
		self.delay(self.short_req, 10, 10)
		self.delay(self.long_req, 600, 500)
		while True:
			try:
				request = Request(url)
				response = urlopen(request)
				break
			except URLError, e:
				print e
				time.sleep(10)
		now = datetime.now()
		self.short_req.append(now)
		self.long_req.append(now)
		return json.loads(response.read())
