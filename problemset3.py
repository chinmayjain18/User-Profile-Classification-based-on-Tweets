import dataStructures
import classifier

import pickle
import sys, os

from textblob import TextBlob

from enum import Enum

# Possible classes for education
class EDUCATION_CLASS(Enum):
    high_school = 'high_school'
    some_college = 'some_college'
    graduate = 'graduate'

def _getEducationFromString(user_education):
    '''
    Args:
        input: Input string from user response for education level
    Returns:
        EDUCATION_CLASS of user, or None for not sure
    '''
    hs_keywords = ['high']
    sc_keywords = ['bachelor', 'college', 'bs', 'ba']
    g_keywords = ['doctoral', 'phd', 'ma', 'master', 'graduate', 'mba', 'mlis']
    if not user_education:
        return None
    else:
        user_education = user_education.lower()
        print('\n' + user_education)
        for keyword in hs_keywords:
            if keyword in user_education:
                return EDUCATION_CLASS.high_school
        for keyword in sc_keywords:
            if keyword in user_education:
                return EDUCATION_CLASS.some_college
        for keyword in g_keywords:
            if keyword in user_education:
                return EDUCATION_CLASS.graduate
    return None

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
                if key == 'Education':
                    setattr(user, key.lower(), _getEducationFromString(value))
                else:
                    setattr(user, key.lower(), value)
        user_list.append(user)
    return user_list

def calculate_features(user_list):
    # Add features to array
    calculated_features = []
    for user in user_list:
        user_dict = {}
        avg_tweet_len = dataStructures.AverageTweetLengthFeature(user)
        num_user_mention = dataStructures.NumberOfTimesOthersMentionedFeature(user)
        user_dict[avg_tweet_len.getKey()] = avg_tweet_len.getValue()
        user_dict[num_user_mention.getKey()] = num_user_mention.getValue()

        #TODO Ugly.
        count = 0
        count_key = ""
        count_personal_sum = 0
        count_personal_sum_key = ""
        for tweet in user.tweets:
            count_categorical_words = dataStructures.CountCategoricalWords(tweet)
            count += count_categorical_words.getValue()
            count_key = count_categorical_words.getKey()
            tweetTB = TextBlob(tweet.rawText)
            count_personal = dataStructures.CountPersonalReferences(tweetTB)
            count_personal_sum += count_personal.getValue()
            count_personal_sum_key = count_personal.getKey()
            #array.
            #pos_tag = dataStructures.POSTagging(tweetTB)
            #user_dict[pos_tag.getKey()] = pos_tag.getValue()

        user_dict[count_key] = count
        user_dict[count_personal_sum_key] = count_personal_sum

        # Merge in time vectors from that feature
        time_vector_feature = dataStructures.FrequencyOfTweetingFeature(user)
        user_dict.update(time_vector_feature.getValue())

        # Add the user dictionary to the features list.
        calculated_features.append(user_dict)
    return calculated_features

def main():
    data_folder = sys.argv[1]
    user_list = load_data(data_folder)

    calculated_features = calculate_features(user_list)
    print(len(calculated_features))
    print(calculated_features)

    user_genders = []
    gender_features = []
    for user, user_feature in zip(user_list, calculated_features):
        if user.gender == "Male" or user.gender == "Female":
            user_genders.append(user.gender)
            gender_features.append(user_feature)

    training_genders = user_genders[:30]
    test_genders = user_genders[30:]
    training_gender_features = gender_features[:30]
    test_gender_features = gender_features[30:]

    acc = classifier.get_SVM_Acc(training_gender_features, training_genders, test_gender_features, test_genders)
    acc_nb = classifier.get_Naivebayes_Acc(training_gender_features, training_genders, test_gender_features, test_genders)
    print (acc)
    print(acc_nb)

if __name__ == '__main__':
    main()
