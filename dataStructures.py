'''
dataStructures.py
Defines the User, Tweet, and Feature classes we will be using
'''

class User:
    '''
    User: defines a single user
    Attributes:
        tweets: list of Tweet objects
        ngrams: dictionary of ngrams
        replacements: dictionary of replacements
        transforms: dictionary of transforms
        userInfo: dictionary of user info
    '''

    def __init__(self, tweets=[], ngrams={}, replacements={}, transforms={}, userInfo={}):
        self.tweets = tweets
        self.ngrams = ngrams
        self.replacements = replacements
        self.transforms = transforms
        self.userInfo = userInfo

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

    def __init__(self, id=0, tokens=[], timestamp='', rawText='', numTokens=0, numPunctuation=0):
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
