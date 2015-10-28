'''
dataStructures.py
Defines the User, Tweet, and Feature classes we will be using
'''

class User:
    '''
    User: defines a single user
    Attributes:
        tweets: list of tweets
    '''

    def __init__(self, tweets):
        self.tweets = []

class Tweet:
    '''
    Tweet: defines a single tweet by a user
    Attributes:
        text: string containing text of the tweet
        time: time the tweet was sent
    '''

    def __init__(self, text):
        self.text = text

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
    def getKey(self):
        return 'CapitalizationFeature'

    # tweet: tweet to be evaluated
    def getValue(self, tweet):
        return sum(1 for c in tweet.text if c.isupper())

class AverageTweetLengthFeature(Feature):
    '''
    AverageTweetLength: counts the average length of the user's tweets
    '''
    def getKey(self):
        return 'AverageTweetLength'

    # user: user to be evaluated
    def getValue(self, user):
        val = 0
        for tweet in user.tweets:
            val += len(tweet)
        if not val:
            return 0
        else:
            return val/len(user.tweets)
