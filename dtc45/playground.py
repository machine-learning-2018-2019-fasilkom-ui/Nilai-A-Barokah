import random
import threading
import time
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )

def worker():
    """thread worker function"""
    t = threading.currentThread()
    pause = random.randint(1,5)
    logging.debug('sleeping %s', pause)
    time.sleep(pause)
    logging.debug('ending')
    return

for i in range(3):
    t = threading.Thread(target=worker)
    t.setDaemon(True)
    t.start()

main_thread = threading.currentThread()
for t in threading.enumerate():
    if t is main_thread:
        continue
    logging.debug('joining %s', t.getName())
    t.join()

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy import API
from tweepy.streaming import StreamListener
import csv
import json

# setup access key
consumer_key = "DBc6ti8PckIYYN1DeG0OKoLPU"
consumer_secret = "O6f5sfNwTLVpbHdnSdhk1m68ipN3DGR8So8glmX0A9qUZk1ihS"
access_token = "276896981-65Uknx7rE8duadm9TkZCgby1x319ju41l8eGktWW"
access_token_secret = "ZAFjNn4uHz69Wy5L6RFjtY8USl2V9xVXsmkTTwivM08Xp"


class MyStreamListener(StreamListener):
    def on_data(self, data):
        # keluaran data berupa json file
        all_data = json.loads(data)
        # print("\n\n")
        # print(all_data)
        # print("\n\n")
        if all_data.has_key('extended_tweet'):
            username = all_data["user"]["screen_name"].encode("utf-8")
            join_date = all_data["user"]["created_at"].encode("utf-8") if all_data["user"]["created_at"] != None else None
            location = all_data["user"]["location"].encode("utf-8") if all_data["user"]["location"] != None else None
            statuses_count = all_data["user"]["statuses_count"].encode("utf-8") if all_data["user"][ "statuses_count"] != None else None
            friends_count = all_data["user"]["friends_count"].encode("utf-8") if all_data["user"]["friends_count"] != None else None
            followers_count = all_data["user"]["followers_count"].encode("utf-8") if all_data["user"]["followers_count"] != None else None

            text = all_data["extended_tweet"]["full_text"].encode("utf-8")
            tweet = text.replace("\n", " ")
            # print(str(tweet))
            if all_data.has_key('created_at'):
                text = all_data["created_at"]
                date = text
            # print(str(date))
            print(str(tweet) + " || " + str(username) + " || " + str(join_date) + " || " + str(location) + " || "+
            str(+statuses_count) + " || " + str(friends_count) + " || " + str(followers_count) + " || " + str(date))

    def on_error(self, status):
        print(status)

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, retry_count=10, retry_delay=5,
          retry_errors=5)
twitterStream = Stream(api.auth, MyStreamListener())
twitterStream.filter(track=["jokowi,ma'ruf,prabowo,sandi,nyoblos,pemilihan,pemilu,pilpres,golput,tps,suara,vote,suara"],languages=["in"])
