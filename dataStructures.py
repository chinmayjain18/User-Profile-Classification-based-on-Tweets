'''
dataStructures.py
Defines the User, Tweet, and Feature classes we will be using
'''

import re
from textblob import TextBlob
import string
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import datetime
import math

class User:
    '''
    User: defines a single user
    Attributes:
        id: string of root folder.
        tweets: list of Tweet objects
        ngrams: dictionary of ngrams
        replacements: dictionary of replacements
        transforms: dictionary of transforms
        month: String birthMonth
        regions: List of regions they claim
        languages: List of strings of languages
        gender: String of gender
        occupation: String of occupation
        astrology: String of zodiac sign
        education: String of education
        year: int of birth year
    '''
    def __init__(self, id="", tweets=[], ngrams={}, replacements={}, transforms={}, userInfo={}, month="", regions=[], languages=[], gender="", occupation="", astrology="", education="", year=0):
        self.id = id
        self.tweets = tweets
        self.ngrams = ngrams
        self.replacements = replacements
        self.transforms = transforms
        self.userInfo = userInfo

        self.month = month
        self.regions = regions
        self.languages = languages
        self.gender = gender
        self.occupation = occupation
        self.astrology = astrology
        self.education = education
        self.year = year

class Tweet:
    '''
    Tweet: defines a single tweet by a user
    Attributes:
        id: id for the tweet
        tokens: tokens for the tweet
        timestamp: tweet timestamp
        rawText: raw text of the tweet
        numTokens: the number of tokens
        numPunctuation: the number of punctation characters in the tweet
    '''

    def __init__(self, id=0, tokens=[], timestamp=0, rawText='', numTokens=0, numPunctuation=0):
        self.id = id
        self.tokens = tokens
        self.timestamp = timestamp
        self.rawText = rawText
        self.numTokens = numTokens
        self.numPunctuation = numPunctuation

class Feature:
    '''
    Feature: defines a generic feature
    Attributes: None
    '''
    def __init__(self):
        pass

    # Return the feature name for storage in the features dictionary
    def getKey(self):
        return ''

    # Returns the evaluated value for the feature
    def getValue(self):
        return ''

############
# Features #
############

class CapitalizationFeature(Feature):
    '''
    CapitalizationFeature: counts the number of capital letters for a tweet
    '''
    def __init__(self, tweet):
        self.tweet = tweet

    def getKey(self):
        return 'CapitalizationFeature'

    # tweet: tweet to be evaluated
    def getValue(self):
        return sum(1 for c in self.tweet.rawText if c.isupper())

class AverageTweetLengthFeature(Feature):
    '''
    AverageTweetLength: counts the average length of the user's tweets
    '''
    def __init__(self, user):
        self.user = user

    def getKey(self):
        return 'AverageTweetLength'

    def getValue(self):
        val = 0
        for tweet in self.user.tweets:
            val += len(tweet.tokens)
        if not val:
            return 0
        else:
            return val/len(self.user.tweets)

class NumberOfTimesOthersMentionedFeature(Feature):
    '''
    NumberOfTimesOthersMentionedFeature: counts the number of times the user
    mentions someone else in their tweets
    '''
    def __init__(self, user):
        self.user = user

    def getKey(self):
        return 'NumberOfTimesOthersMentioned'

    def getValue(self):
        val = 0
        comp_regex = re.compile('@[A-z]+')
        for tweet in self.user.tweets:
            val += len(re.findall(comp_regex, tweet.rawText))
        return val

class POSTagging(Feature):
	'''
	POSTagging: returns Part of Speech tagging for the tweet
	'''
	def __init__(self, tweetTB):
		self.tweetTB = tweetTB;

	def getKey(self):
		return 'POSTagging';

	def getValue(self):
		return self.tweetTB.tags;

class CountPersonalReferences(Feature):
	'''
	CountPersonalReferences: Counts the number of Personal References used
	'''

	def __init__(self, tweetTB):
		self.tweetTB = tweetTB;

	def getKey(self):
		return 'CountPersonalReferences';

	def getValue(self):
		listOfWords = list(self.tweetTB.tokens);
		count = 0;
		listOfPR = ['I','he','she','we','you','they'];
		for word in listOfWords:
			if word in listOfPR:
				count += 1;
		return count;

class CountPunctuations(Feature):
	'''
	CountPunctuations: Counts the number of Punctuations
	'''

	def __init__(self, tweetTB):
		self.tweetTB = tweetTB;

	def getKey(self):
		return 'CountPunctuations';

	def getValue(self):
		punctuations = string.punctuation;
		listOfWords = list(self.tweetTB.tokens);
		count = 0;
		for word in listOfWords:
			if word in punctuations:
				count += 1;
		return count;

class CountHashTags(Feature):
	'''
	CountHashTags: Counts the number of HashTags in the tweet
	'''

	def __init__(self, tweetTB):
		self.tweetTB = tweetTB;

	def getKey(self):
		return 'CountHashTags';

	def getValue(self):
		pattern = re.compile('#([a-zA-Z0-9]+)');
		count = 0;
		listOfWords = self.tweetTB.split();
		for word in listOfWords:
			if pattern.match(word):
				count += 1;
		return count;

class CountEmoticon(Feature):
	'''
	CountEmoticon:: Counts the number of emoticons in the tweet
	'''

	def __init__(self, tweetTB):
		self.tweetTB = tweetTB;

	def getKey(self):
		return 'CountEmoticon';

	def getValue(self):
		pattern = re.compile(':\)|:\(|:D|:\'\)|=\)|:O|:P|B\)');
		count = 0;
		posTagList = self.tweetTB.tags;
		for (word,tag) in posTagList:
			if pattern.match(word):
				count += 1;
		return count;

class CountEmotionalWords(Feature):
	'''
	CountEmotionalWords: Counts the number of emotional words in the tweet
	'''

	def __init__(self, tweetTB):
		self.tweetTB = tweetTB;

	def getKey(self):
		return 'CountEmotionalWords';

	def getValue(self):
		file = open('EmotionalWords.txt','r');
		listOfWords = [word.lower() for word in (file.read()).split(',')];
		count = 0;
		for (word,tag) in self.tweetTB.tags:
			if word in listOfWords:
				count += 1;
		return count;

class CountMisspelledWords(Feature):
	'''
	CountMisspelledWords: Counts the number of misspelled words in the tweet
	'''

	def __init__(self,tweetTB):
		self.tweetTB = tweetTB;

	def getKey(self):
		return 'CountMisspelledWords';

	def getValue(self):
		count = 0;
		stopwordList = stopwords.words('english');
		for (word,tag) in self.tweetTB.tags:
			if not wordnet.synsets(word) and word.lower() not in stopwordList and tag != 'SYM':
				count += 1;
		return count;

class FrequencyOfTweetingFeature(Feature):
    '''
    FrequencyOfTweetingFeature: Builds histogram broken into times when user tweeted
    Returns: dictionary of features with how many times the user tweeted in that interval
    '''

    MINUTE_INTERVAL = 30.0 # The size of the histogram buckets in minutes

    def __init__(self, user):
        self.user = user

    def getKey(self):
        return 'FrequencyOfTweetingFeature'

    def getValue(self):
        time_vector = [0] * int((24*60)/self.MINUTE_INTERVAL) # e.g. 48 for 30 min interval
        for tweet in self.user.tweets:
            time = datetime.datetime.utcfromtimestamp(tweet.timestamp/100)
            time_in_min = time.hour*60 + time.minute
            index_in_time = math.floor(time_in_min/self.MINUTE_INTERVAL) - 1
            time_vector[index_in_time] += 1
        # Convert vector to dictionary
        time_dict = {}
        for x in range(0, len(time_vector)):
            time_dict[self.getKey() + '_{0}'.format(x)] = time_vector[x]
        return time_dict

class NumberOfMultiTweetsFeature(Feature):
    '''
    NumberOfMultiTweetsFeature: Counts the number of multi-tweet tweets for the user
    Note: Counts number of *complete* multi-tweets as 1
    '''
    def __init__(self, user):
        self.user = user

    def getKey(self):
        return 'NumberOfMultiTweetsFeature'

    def getValue(self):
        multi_tweets = []
        num_complete_multi_tweets = 0
        pattern = re.compile('^\(([0-9]+) of ([0-9]+)\)') # Matches: ^(X of Y)

        for tweet in self.user.tweets:
            match = pattern.match(tweet.rawText)
            if match:
                placed = False
                tweet_num = match.group(1)
                of_tweet = match.group(2)

                # Search and see if there is a multi-tweet of the same size missing our number
                for multi_tweet in multi_tweets:
                    if len(multi_tweet) == of_tweet and not multi_tweet[tweet_num]:
                        multi_tweet[tweet_num] = True
                        placed = True
                        break
                # No existing multi-tweet for this tweet, make a new multi-tweet
                if not placed:
                    multi_tweet = [False] * of_tweet
                    multi_tweet[tweet_num] = True
                    multi_tweets.append(multi_tweet)

        for multi_tweet in multi_tweets:
            # If all of the mutli-tweet is True, it is a complete multi-tweet
            if sum(multi_tweet) == len(multi_tweet):
                num_complete_multi_tweets += 1

        return num_complete_multi_tweets

class CountCategoricalWords(Feature):
    '''
    CountCategoricalWords : Counts the number of categorical words in the tweet
    '''
    def __init__(self,tweet):
        self.tweet = tweet;

    def getKey(self):
        return 'CountCategoricalWords';

    def getValue(self):
        categoricalWordsList = ['help','acheive','success','dreams','goals',
                                'career','beer','alcohol','sex','football',
                                'esteem','ego','pride','gym'];
        wordsList = self.tweet.rawText.split(" ");
        count = 0;
        for word in wordsList:
            if word in categoricalWordsList:
                count += 1;
        return count;
