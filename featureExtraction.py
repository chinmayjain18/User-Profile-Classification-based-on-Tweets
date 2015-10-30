from textblob import TextBlob
import string
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords

# Returns POS tagging of the tweet
def posTagging(tweet):
    return tweet.tags;

# Returns count of personal reference words
def countPersonalReferences(tweet):
    listOfWords = list(tweet.tokens);
    count = 0;
    listOfPR = ['I','he','she','we','y;ou','they'];
    for word in listOfWords:
        if word in listOfPR:
            count += 1;
    return count;

# Returns count of punctuations
def countPunctuations(tweet):
    punctuations = string.punctuation;
    listOfWords = list(tweet.tokens);
    count = 0;
    for word in listOfWords:
        if word in punctuations:
            count += 1;
    return count;

# Returns count of HashTags in the tweet
def countHashTags(tweet):
    pattern = re.compile('#([a-zA-Z0-9]+)');
    count = 0;
    listOfWords = tweet.split();
    for word in listOfWords:
        if pattern.match(word):
            count += 1;
    return count;

# Retunrs count of Emoticons in the tweet
def countEmoticon(tweet):
    pattern = re.compile(':\)|:\(|:D|:\'\)|=\)|:O|:P|B\)');
    count = 0;
    posTagList = tweet.tags;
    for (word,tag) in posTagList:
        if pattern.match(word):
            count += 1;
    return count;
    
# Returns count of Emotional Words in the tweet
def countEmotionalWords(tweet):
    file = open('EmotionalWords.txt','r');
    listOfWords = [word.lower() for word in (file.read()).split(',')];
    count = 0;
    for (word,tag) in tweet.tags:
        if word in listOfWords:
            count += 1;
    return count;

# Returns count of Misspelled Words in the tweet
def countMisspelledWords(tweet):
    count = 0;
    stopwordList = stopwords.words('english');
    for (word,tag) in tweet.tags:
        if not wordnet.synsets(word) and word.lower() not in stopwordList and tag != 'SYM':
            count += 1;
    return count;
    
# Main code
line = "Heyyyy.. My Name is Chinmay!!!! I am so happy!!! That so gay.. :) =) #Name #Me #12";
text = TextBlob(line);
print(text.tokens);
print("\n");
print(posTagging(text));
print(countPersonalReferences(text));
print(countPunctuations(text));
print(countHashTags(line));
print(countEmoticon(text));
print(countEmotionalWords(text));
print(countMisspelledWords(text));