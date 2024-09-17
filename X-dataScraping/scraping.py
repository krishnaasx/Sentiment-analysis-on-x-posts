import asyncio
from twikit import Client, TooManyRequests
from datetime import datetime
import csv
from configparser import ConfigParser
from random import uniform

MINIMUM_TWEETS = 30
QUERY = 'himalayan_hunk'
MAX_TWEETS_PER_REQUEST = 10
MIN_DELAY = 5
MAX_DELAY = 20

config = ConfigParser()
config.read('config.ini')
username = config['X']['username']
password = config['X']['password']
email = config['X']['email']

async def human_like_delay():
    delay = uniform(MIN_DELAY, MAX_DELAY)
    print(f"Waiting for {delay:.2f} seconds...")
    await asyncio.sleep(delay)

async def main():
    client = Client(language='en-US')
    # await client.login(auth_info_1=username, auth_info_2=email, password=password)
    # client.save_cookies('cookies.json')
    client.load_cookies('cookies.json')
    tweet_count = 0
    tweets = None

    with open('tweets.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Tweet_count', 'Username', 'Text', 'Created At', 'Retweets', 'Likes', 'Tweet ID'])

    while tweet_count < MINIMUM_TWEETS:
        try:
            if tweets is None:
                print(f'{datetime.now()} - Getting tweets....')
                tweets = await client.search_tweet(QUERY, product='Media')
            else:
                print(f'{datetime.now()} - Getting next tweets....')
                tweets = await tweets.next()

            if not tweets:
                print(f'{datetime.now()} - No more tweets found')
                break

            batch_size = min(MAX_TWEETS_PER_REQUEST, MINIMUM_TWEETS - tweet_count)
            batch_count = 0

            for tweet in tweets:
                if batch_count >= batch_size:
                    break

                tweet_count += 1
                batch_count += 1
                tweet_data = [tweet_count, tweet.user.name, tweet.text, tweet.created_at, tweet.retweet_count, tweet.favorite_count, tweet.id]

                with open('tweets.csv', 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(tweet_data)

                await asyncio.sleep(uniform(0.5, 2))

            print(f'{datetime.now()} - Got {batch_count} tweets (Total: {tweet_count})')

            if tweet_count < MINIMUM_TWEETS:
                await human_like_delay()

        except TooManyRequests:
            print("Rate limit exceeded. Waiting for a longer period...")
            await asyncio.sleep(900)  
        except Exception as e:
            print(f"An error occurred: {e}")
            await asyncio.sleep(30)  

    print(f'{datetime.now()} - Done! Got {tweet_count} tweets in total')

if __name__ == "__main__":
    asyncio.run(main())