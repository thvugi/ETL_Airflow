import tweepy
import pandas as pd
import json
from datetime import datetime
import s3fs


def run_twitter_etl():

    #keyss
    consumer_key = "1611844936441135110-6jH2PGmx15uFqR97IavLsyXrqJJMDY"
    consumer_secret = "uuKZRMl4DZSbAOaPp5IcSYhy1pS0hJ5ylXV0kIFl53Cy6"
    access_key = "u5MSqFYpjj7I0Lss5Di81qufB"
    access_secret = "WXufFfswk4CSMKcHmSKt8pJtPhbf6AtrauM5w2Sw1g0Sq9EiYs"


    #twitter authentication
    auth = tweepy.OAuthHandler(access_key, access_secret)
    auth.set_access_token(consumer_key, consumer_secret)


    #creating an API object
    api = tweepy.API(auth)

    tweets = api.user_timeline(
                            screen_name = '@VancityReynolds',
                            count = 200,
                            include_rts = False,
                            tweet_mode = 'extended'

    )

    print(tweets)

    tweet_list = []
    for tweets in tweets:
        text = tweets._json["full_text"]
        refined_tweet = {
            "users":tweets.user.screen_name,
            'text' : text,
            'favorite_count' : tweets.favorite_count,
            'retweet_count' : tweets.retweet_count,
            'created_at' : tweets.created_at

        }
        tweet_list.append(refined_tweet)

        df = pd.DataFrame(tweet_list)
        df.to_csv("s3://thvuvi-twitter-airflow-bucket/twitter_tweet_data.csv")