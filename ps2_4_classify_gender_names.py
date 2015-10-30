import nltk
from nltk.corpus import names
import random
from string import ascii_lowercase

import dataStructures

import pickle
import sys, os

# Input:
# files = list of strings of filenames in a directory
# filename = string of file we are looking for.
# Returns:
# None if file is not found, otherwise, returns filename of file that was looked for.
def check_for_file(files, filename):
    if filename in files:
        index = files.index(filename)
        return files[index]
    else:
        return None

# Input:
# root = root directory that file is in.
# filename = file we are going to open
# Returns:
# None if filename is none, otherwise, returns unpickled_data from file.
def unpickle_from_filename(root, filename):
    if filename is None:
        return None
    else:
        unpickled_data = pickle.load(open(os.path.join(root, filename), "rb"))
        return unpickled_data

# Input:
# data_folder = string, foldername we are going to recursively traverse.
# Returns:
# List of User objects. There should be a User object per subfolder.
def load_data(data_folder):
    user_list = []
    for root, sub_folders, files in os.walk(data_folder):
        ngramsFile = check_for_file(files, 'ngrams.pickl')
        replacementsFile = check_for_file(files, 'replacements.pickl')
        transformsFile = check_for_file(files, 'transforms.pickl')
        tweetsFile = check_for_file(files, 'tweets.pickl')
        userFile = check_for_file(files, 'user.pickl')

        ngrams = unpickle_from_filename(root, ngramsFile)
        replacements = unpickle_from_filename(root, replacementsFile)
        transforms = unpickle_from_filename(root, transformsFile)
        tweets = unpickle_from_filename(root, tweetsFile)
        userInfo = unpickle_from_filename(root, userFile)

        # Take tweets dictionary and turn to Tweet objects.
        tweets_list = []
        if tweets is not None:
            for tweetId, value in tweets.items():
                tweet = dataStructures.Tweet(id=tweetId, tokens=value["tokenized"], timestamp=value["time"], rawText=value["text"], numTokens=value["tokens"], numPunctuation=value["punc"])
                tweets_list.append(tweet)

        user = dataStructures.User(id=root, tweets=tweets_list, ngrams=ngrams, replacements=replacements, transforms=transforms)
        if userInfo is not None:
            for key, value in userInfo.items():
                setattr(user, key.lower(), value)
        user_list.append(user)
    return user_list


def gender_features(word):
    features = {'first two letters': word[:2], 'last letter': word[-1], 'first letter': word[0], 'length': len(word), 'last two': word[-2:], 'last three': word[-3:]}
    for letter in ascii_lowercase:
        features["count(%s)" % letter] = word.lower().count(letter)
    return features

def main():
    data_folder = sys.argv[1]
    user_list = load_data(data_folder)

    labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
    random.shuffle(labeled_names)
    featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    train_set, test_set, final_set = featuresets[2382:], featuresets[1191:2382], featuresets[:1191]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    print("Accuracy for devtest_set:")
    print(nltk.classify.accuracy(classifier, test_set))
    print("")
    print(classifier.show_most_informative_features(5))
    print("")

    error = []
    count = 0
    for word, tag in final_set:
        count += 1
        guess = classifier.classify(word)
        if guess != tag:
            #print "Wrong! " + label + " => " + guess
            error.append(word)
        #print error
    print ("error rate in final set:", float(len(error)) / float(count))
    print ("Accuracy for final set:")
    print(nltk.classify.accuracy(classifier, final_set))

if __name__ == '__main__':
    main()
