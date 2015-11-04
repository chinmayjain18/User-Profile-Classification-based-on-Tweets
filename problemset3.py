import dataStructures
import classifier

import pickle
import sys, os

from textblob import TextBlob

import argparse

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
        for keyword in hs_keywords:
            if keyword in user_education:
                return EDUCATION_CLASS.high_school.value
        for keyword in sc_keywords:
            if keyword in user_education:
                return EDUCATION_CLASS.some_college.value
        for keyword in g_keywords:
            if keyword in user_education:
                return EDUCATION_CLASS.graduate.value
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
        if root == ".DS_Store" or root == data_folder:
            continue
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

        avg_tweet_len = dataStructures.AverageTweetLengthFeature(user)
        num_user_mention = dataStructures.NumberOfTimesOthersMentionedFeature(user)
        num_languages = dataStructures.CountLanguageUsed(user)

        user_dict = {}
        user_dict[avg_tweet_len.getKey()] = avg_tweet_len.getValue()
        user_dict[num_user_mention.getKey()] = num_user_mention.getValue()
        user_dict[num_languages.getKey()] = num_languages.getValue()

        count, count_personal_sum = 0, 0
        count_key, count_personal_sum_key = "", ""

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

    parser = argparse.ArgumentParser(description='Problem Set 3')
    parser.add_argument('data_folder', help='path to data folder')
    parser.add_argument('-v', help='verbose mode')

    args = parser.parse_args()

    verbose_mode = bool(args.v)

    user_list = load_data(args.data_folder)

    calculated_features = calculate_features(user_list)

    if verbose_mode:
        print(len(calculated_features))
        print(calculated_features)

    user_genders = []
    gender_features = []
    user_educations = []
    education_features = []
    for user, user_feature in zip(user_list, calculated_features):
        if user.gender == "Male" or user.gender == "Female":
            user_genders.append(user.gender)
            gender_features.append(user_feature)
        if user.education is not None:
            user_educations.append(user.education)
            education_features.append(user_feature)

    training_genders = user_genders[:30]
    test_genders = user_genders[30:]
    training_gender_features = gender_features[:30]
    test_gender_features = gender_features[30:]
    training_educations = user_educations[:30]
    test_educations = user_educations[30:]
    training_education_features = education_features[:30]
    test_education_features = education_features[30:]

    acc = classifier.get_SVM_Acc(training_gender_features, training_genders, test_gender_features, test_genders)
    acc_nb = classifier.get_Naivebayes_Acc(training_gender_features, training_genders, test_gender_features, test_genders)
    #acc_lr = classifier.get_LinearRegression_Acc(training_gender_features, training_genders, test_gender_features, test_genders)
    print('\t{0} gender accuracy: {1}'.format('SVM', acc))
    print('\t{0} gender accuracy: {1}'.format('Naive Bayes', acc_nb))
    #print('\t{0} gender accuracy: {1}'.format('Linear Regression', acc_lr))

    acc = classifier.get_SVM_Acc(training_education_features, training_educations, test_education_features, test_educations)
    acc_nb = classifier.get_Naivebayes_Acc(training_education_features, training_educations, test_education_features, test_educations)
    #acc_lr = classifier.get_LinearRegression_Acc(training_education_features, training_educations, test_education_features, test_educations)
    print('\t{0} education accuracy: {1}'.format('SVM', acc))
    print('\t{0} education accuracy: {1}'.format('Naive Bayes', acc_nb))
    #print('\t{0} education accuracy: {1}'.format('Linear Regression', acc_lr))

if __name__ == '__main__':
    main()
