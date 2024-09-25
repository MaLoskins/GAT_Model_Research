import os
import json
import csv
import warnings
warnings.filterwarnings("ignore")

class PhemeProcessor:
    """
    A class to process PHEME dataset directories and export tweet data to CSV files.
    """
    
    def __init__(self, base_dir, output_dir=None):
        """
        Initializes the PhemeProcessor.

        :param base_dir: The base directory where event directories are located.
        :param output_dir: The directory where CSV files will be saved. Defaults to base_dir.
        """
        self.base_dir = base_dir
        self.output_dir = output_dir if output_dir else base_dir
        self.categories = ['rumours', 'non-rumours']
        self.fieldnames = [
            'tweet_id', 'user_id', 'user_screen_name', 'user_followers_count',
            'user_friends_count', 'text', 'created_at', 'retweet_count',
            'favorite_count', 'lang', 'geo', 'hashtags', 'mentions',
            'urls', 'label', 'reply_to_tweet_id'
        ]
    
    def process_events(self, events):
        """
        Processes specified events. For each event, it checks if the CSV
        already exists. If not, it processes the event and writes the CSV.

        :param events: A list of event names to process.
        """
        for event in events:
            csv_filename = f"{event}.csv"
            csv_path = os.path.join(self.output_dir, csv_filename)
            
            if os.path.exists(csv_path):
                print(f"[SKIP] CSV already exists for event '{event}': {csv_path}. Skipping processing.")
                continue
            
            print(f"[PROCESS] Starting processing for event: {event}")
            tweets_data = self.process_event(event)
            
            if tweets_data:
                self.write_csv(csv_path, tweets_data)
                print(f"[SUCCESS] Data for event '{event}' has been successfully written to {csv_path}\n")
            else:
                print(f"[NO DATA] No data collected for event '{event}'.\n")
    
    def process_event(self, event):
        """
        Processes a single event directory and collects tweet data.

        :param event: The name of the event to process.
        :return: A list of dictionaries containing tweet data.
        """
        event_dir = os.path.join(self.base_dir, event)
        if not os.path.isdir(event_dir):
            print(f"[ERROR] Event directory not found: {event_dir}")
            return []
        
        print(f'[INFO] Processing event: {event}')
        tweets_data = []
        
        for category in self.categories:
            category_dir = os.path.join(event_dir, category)
            if not os.path.isdir(category_dir):
                print(f"  [WARN] Category directory not found: {category_dir}")
                continue
            print(f'  [INFO] Processing category: {category}')
            
            threads = [d for d in os.listdir(category_dir) if not d.startswith('.')]
            for thread in threads:
                thread_dir = os.path.join(category_dir, thread)
                if not os.path.isdir(thread_dir):
                    continue  # Skip if it's not a directory
                print(f'    [INFO] Processing thread: {thread}')
                
                label = self.get_label(thread_dir, category)
                source_tweets_dir = os.path.join(thread_dir, 'source-tweets')
                if os.path.exists(source_tweets_dir):
                    tweets_data.extend(
                        self.process_tweets(source_tweets_dir, label)
                    )
                else:
                    print(f"      [WARN] Source tweets directory not found: {source_tweets_dir}")
                
                reactions_dir = self.find_reactions_dir(thread_dir)
                if reactions_dir:
                    print(f"      [INFO] Processing reactions in directory: {reactions_dir}")
                    tweets_data.extend(
                        self.process_tweets(reactions_dir, label)
                    )
                else:
                    print(f"      [INFO] No reactions directory found in {thread_dir}")
            
            print(f'  [INFO] Total tweets collected for category "{category}": {len(tweets_data)}')
        
        return tweets_data
    
    def get_label(self, thread_dir, category):
        """
        Determines the label for tweets based on the annotation or category.

        :param thread_dir: The directory of the thread.
        :param category: The category of the thread ('rumours' or 'non-rumours').
        :return: The label as an integer (1 for rumour, 0 for non-rumour).
        """
        annotation_file = os.path.join(thread_dir, 'annotation.json')
        label = 'non-rumor'
        
        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotation = json.load(f)
                    label_str = annotation.get('is_rumour', 'non-rumor').lower()
                    if label_str == 'rumour':
                        label = 1
                    elif label_str == 'nonrumour':
                        label = 0
                    else:
                        label = 'non-rumor'
                    print(f"      [INFO] Label from annotation: {label_str} -> {label}")
            except Exception as e:
                print(f"      [ERROR] Error reading annotation file {annotation_file}: {e}")
        else:
            label = 0 if category == 'non-rumours' else 1
            print(f"      [INFO] No annotation found. Assigning label based on category: {label}")
        
        return label
    
    def find_reactions_dir(self, thread_dir):
        """
        Finds the reactions directory within a thread directory.

        :param thread_dir: The directory of the thread.
        :return: The path to the reactions directory or None if not found.
        """
        subdirs = [d for d in os.listdir(thread_dir) if os.path.isdir(os.path.join(thread_dir, d))]
        for d in subdirs:
            if 'reaction' in d.lower():
                return os.path.join(thread_dir, d)
        return None
    
    def process_tweets(self, tweets_dir, label):
        """
        Processes all tweet JSON files in a specified directory.

        :param tweets_dir: The directory containing tweet JSON files.
        :param label: The label to assign to each tweet.
        :return: A list of dictionaries containing tweet data.
        """
        tweets = []
        for filename in os.listdir(tweets_dir):
            if filename.startswith('.') or filename.startswith('._'):
                continue
            if not filename.endswith('.json'):
                continue
            
            tweet_path = os.path.join(tweets_dir, filename)
            try:
                with open(tweet_path, 'r', encoding='utf-8') as f:
                    tweet = json.load(f)
                    tweet_data = self.extract_tweet_data(tweet, label)
                    if tweet_data:
                        tweets.append(tweet_data)
            except Exception as e:
                print(f"      [ERROR] Error reading tweet file {tweet_path}: {e}")
        return tweets
    
    def extract_tweet_data(self, tweet, label):
        """
        Extracts relevant fields from a tweet JSON object.

        :param tweet: The tweet JSON object.
        :param label: The label to assign to the tweet.
        :return: A dictionary with extracted tweet data.
        """
        try:
            tweet_id = tweet.get('id_str', '')
            user = tweet.get('user', {})
            user_id = user.get('id_str', '')
            user_screen_name = user.get('screen_name', '')
            user_followers_count = user.get('followers_count', 0)
            user_friends_count = user.get('friends_count', 0)
            text = tweet.get('text', '').replace('\n', ' ').replace('\r', ' ')
            created_at = tweet.get('created_at', '')
            retweet_count = tweet.get('retweet_count', 0)
            favorite_count = tweet.get('favorite_count', 0)
            lang = tweet.get('lang', '')
            geo = tweet.get('geo', '') if tweet.get('geo') else ''
            reply_to_tweet_id = tweet.get('in_reply_to_status_id_str', '')
            
            # Extract hashtags, user mentions, and URLs from entities
            entities = tweet.get('entities', {})
            hashtags = [hashtag.get('text', '') for hashtag in entities.get('hashtags', [])]
            mentions = [mention.get('screen_name', '') for mention in entities.get('user_mentions', [])]
            urls = [url.get('expanded_url', '') for url in entities.get('urls', [])]
            
            tweet_data = {
                'tweet_id': tweet_id,
                'user_id': user_id,
                'user_screen_name': user_screen_name,
                'user_followers_count': user_followers_count,
                'user_friends_count': user_friends_count,
                'text': text,
                'created_at': created_at,
                'retweet_count': retweet_count,
                'favorite_count': favorite_count,
                'lang': lang,
                'geo': geo,
                'hashtags': ','.join(hashtags),
                'mentions': ','.join(mentions),
                'urls': ','.join(urls),
                'label': label,
                'reply_to_tweet_id': reply_to_tweet_id
            }
            return tweet_data
        except Exception as e:
            print(f"      [ERROR] Error extracting data from tweet: {e}")
            return None
    
    def write_csv(self, csv_path, tweets_data):
        """
        Writes the collected tweet data to a CSV file.

        :param csv_path: The path to the CSV file.
        :param tweets_data: A list of dictionaries containing tweet data.
        """
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
                for tweet in tweets_data:
                    writer.writerow(tweet)
        except Exception as e:
            print(f"[ERROR] Error writing to CSV file {csv_path}: {e}")

