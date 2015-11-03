import dataStructures
import classifier

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

def main():
    data_folder = sys.argv[1]
    user_list = load_data(data_folder)
    user_gender_list = []
    gender_list = []
    for user in user_list:
        if user.gender == "Male" or user.gender == "Female":
            user_gender_list.append(user)
            gender_list.append(user.gender)
    training_gender_list = gender_list[:30]
    test_gender_list = gender_list[30:]

    # Add features to array
    f_objects = []
    for user in user_gender_list:
        avg_tweet_len = dataStructures.AverageTweetLengthFeature(user)
        freq_tweet = dataStructures.FrequencyOfTweetingFeature(user)
        user_dict = {}
        user_dict[avg_tweet_len.getKey()] = avg_tweet_len.getValue()
        #user_dict[freq_tweet.getKey()] = freq_tweet.getValue()
        f_objects.append(user_dict)
        #cap_list = []
        #for tweet in user.tweets:
        #    cap_list.append(dataStructures.CapitalizationFeature(tweet))
        #f_objects.append(cap_list)
    print(len(f_objects))
    print(f_objects)
    training_feature_objects = f_objects[:30]
    test_feature_objects = f_objects[30:]
    acc = classifier.get_SVM_Acc(training_feature_objects, training_gender_list, test_feature_objects, test_gender_list)
    print (acc)

if __name__ == '__main__':
    main()
