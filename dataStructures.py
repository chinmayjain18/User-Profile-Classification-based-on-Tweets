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

    def __init__(self):
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
