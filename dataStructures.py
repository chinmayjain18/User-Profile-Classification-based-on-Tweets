'''
dataStructures.py
Defines the User, Tweet, and Feature classes we will be using
'''

import re

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
            val += len(tweet.rawText)
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
