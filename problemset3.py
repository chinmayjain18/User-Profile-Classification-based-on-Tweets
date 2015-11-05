import dataStructures
import classifier

import pickle
import os

from textblob import TextBlob

import argparse

from random import shuffle
from enum import Enum

# Possible classes for education
class EDUCATION_CLASS(Enum):
    high_school = 0
    some_college = 1
    graduate = 2

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

    #user_list = completeGenderData(user_list);
    #user_list = completeEducationData(user_list);

    return user_list

def completeEducationData(user_list):
    highSchoolCount = 0;
    bachelorsCount = 0;
    graduateCount = 0;
    n = len(user_list);
    for user in user_list:
        if user.education == 0:
            highSchoolCount += 1;
        elif user.education == 1:
            bachelorsCount += 1;
        elif user.education == 2:
            graduateCount += 1;
    missingHighSchool = (int)(n/3) - highSchoolCount;
    missingBachelors = (int)(n/3) - bachelorsCount;
    missingGarduates = (int)(n/3) - graduateCount;
    randomEducationList = [];
    j = 0;
    for j in range(missingHighSchool):
        randomEducationList.append(0);
    j = 0;
    for j in range(missingBachelors):
        randomEducationList.append(1);
    j = 0;
    for j in range(missingGarduates):
        randomEducationList.append(2);
    shuffle(randomEducationList);
    for x in user_list:
        if x.education is None:
            x.education = randomEducationList.pop();
    return user_list;

def completeGenderData(user_list):
    maleCount = 0;
    femaleCount = 0;
    n = len(user_list);
    for user in user_list:
        if user.gender == 'Male':
            maleCount += 1;
        elif user.gender == 'Female':
            femaleCount += 1;
    missingMale = (int)(n/2) - maleCount;
    missingFemale = (int)(n/2) - femaleCount;
    randomGenderList = [];
    j = 0;
    for j in range(missingMale):
        randomGenderList.append('Male');
    j =0;
    for j in range(missingFemale):
        randomGenderList.append('Female');
    shuffle(randomGenderList);
    user = '';
    for user in user_list:
        if user.gender is None:
            user.gender = randomGenderList.pop();
    return user_list;

def calculate_features(user_list):
    '''
    Calculates the features for each user in user_list
    Args:
        user_list: List of users
    Returns:
        calculated_features: list of dictionaries of features for each user
        in user_list in the same order as user_list
    '''
    calculated_features = []

    # Load emotional words.
    file = open('EmotionalWords.txt','r');
    listOfEmotionalWords = [word.lower() for word in (file.read()).split(',')];

    for user in user_list:

        features = []
        features.append(dataStructures.AverageTweetLengthFeature(user))
        features.append(dataStructures.NumberOfTimesOthersMentionedFeature(user))
        features.append(dataStructures.NumberOfMultiTweetsFeature(user))
        features.append(dataStructures.CountRetweet(user))
        features.append(dataStructures.CountLanguageUsed(user))
        features.append(dataStructures.CountRegions(user))
        features.append(dataStructures.AgeOccupation(user))
        features.append(dataStructures.CountReplacements(user))
        features.append(dataStructures.CountTweets(user))

        user_dict = {}
        for feature in features:
            user_dict[feature.getKey()] = feature.getValue()

        tweet_dict = {}
        for tweet in user.tweets:

            tweet_features = []
            tweetTB = TextBlob(tweet.rawText)
            tweetTB_tags = tweetTB.tags

            tweet_features.append(dataStructures.CapitalizationFeature(tweet))
            tweet_features.append(dataStructures.CountNouns(tweetTB, tweetTB_tags))
            tweet_features.append(dataStructures.CountVerbs(tweetTB, tweetTB_tags))
            tweet_features.append(dataStructures.CountAdjectives(tweetTB, tweetTB_tags))
            tweet_features.append(dataStructures.CountPersonalReferences(tweetTB, tweetTB_tags))
            tweet_features.append(dataStructures.CountPunctuations(tweet))
            tweet_features.append(dataStructures.CountHashTags(tweet))
            tweet_features.append(dataStructures.CountEmoticon(tweetTB, tweetTB_tags))
            tweet_features.append(dataStructures.CountEmotionalWords(tweetTB, listOfEmotionalWords, tweetTB_tags))
            tweet_features.append(dataStructures.CountCategoricalWords(tweet))

            for tweet_feature in tweet_features:
                key = tweet_feature.getKey()
                if not key in tweet_dict.keys():
                    tweet_dict[key] = 0
                tweet_dict[key] += tweet_feature.getValue()

        # Merge tweet-level dic (summed) values into user dic
        user_dict.update(tweet_dict)

        # Merge in time vectors from that feature
        time_vector_feature = dataStructures.FrequencyOfTweetingFeature(user)
        user_dict.update(time_vector_feature.getValue())

        # Add the user dictionary to the features list.
        calculated_features.append(user_dict)

    return calculated_features

def _testAllFeatures(classes, features):
    num_features = len(features[0].keys())
    results = []
    for n in range(1, num_features+1):
        results += _testNFeaturesTogether(n, classes, features)
    results.sort(key=lambda x: x[1])
    results.reverse()
    for x in results[:20]:
        print(x)

def _testNFeaturesTogether(n, classes, features):
    '''
    Test N features together in the classifier and sort's their performance
    '''

    # Split into train and test
    TEST_RATIO = 0.75
    split_index = int(len(classes) * TEST_RATIO)

    train_classes, test_classes = classes[:split_index], classes[split_index:]
    train_features, test_features = features[:split_index], features[split_index:]

    # Test every combination of features
    feature_names = features[0].keys()
    results = []

    import itertools
    feature_combinations = itertools.combinations(feature_names, n)
    for feature_combination in feature_combinations:

        # filter train and test features with this combination
        train_f = _filterFeatures(feature_combination, train_features)
        test_f = _filterFeatures(feature_combination, test_features)

        acc = []
        acc.append(classifier.get_SVM_Acc(train_f, train_classes, test_f, test_classes))
        acc.append(classifier.get_Naivebayes_Acc(train_f, train_classes, test_f, test_classes))
        acc.append(classifier.get_LinearRegression_Acc(train_f, train_classes, test_f, test_classes))

        best_perf = max(acc)
        best_classifier = ' SVM'
        if acc.index(best_perf) == 1:
            best_classifier = ' NB'
        elif acc.index(best_perf) == 2:
            best_classifier = ' LR'

        tmp = ""
        for s in feature_combination:
            tmp = tmp + " " + s
        results.append( (tmp + best_classifier, best_perf) )

    return results
    # Find best results and print top 5
    # results.sort(key=lambda x: x[1])
    # results.reverse()

    # for x in results[:5]:
    #     print(x[0] + ': ' +str(x[1]))
    # print("")

def _testAccuracy(display_type, classes, features):
    '''
    Tests the accuracy, prints results.
    Args:
        display_type: String value of type to be displayed in print message (eg. 'gender')
        train_classes: list of user's classes to train on (eg. genders)
        train_features: corrisponding list of user's computed features
        test_classes: list of user's classes to test on (eg. genders)
        test_features: corrisponding list of user's computed features
    '''

    ACC_STRING = '\t{0} accuracy: {1}'
    TEST_RATIO = 0.75
    split_index = int(len(classes) * TEST_RATIO)

    train_classes, test_classes = classes[:split_index], classes[split_index:]
    train_features, test_features = features[:split_index], features[split_index:]

    # SVM
    acc1 = classifier.get_SVM_Acc(display_type,train_features, train_classes, test_features, test_classes)

    # Naive Bayes
    acc2 = classifier.get_Naivebayes_Acc(display_type,train_features, train_classes, test_features, test_classes)

    # Linear Regression
    acc3 = classifier.get_LinearRegression_Acc(display_type,train_features, train_classes, test_features, test_classes)
    
    acc = max(acc1,acc2,acc3)
    print(ACC_STRING.format(display_type, acc))

    print("")
    
def trainClassifier(display_type, classifier_function, classes, features):
    '''
    Tests the accuracy, prints results.
    Args:
        display_type: String value of type to be displayed in print message (eg. 'gender')
        train_classes: list of user's classes to train on (eg. genders)
        train_features: corrisponding list of user's computed features
        test_classes: list of user's classes to test on (eg. genders)
        test_features: corrisponding list of user's computed features
    '''
    classifier_function(features, classes, features, classes,display_type)

    print("")

def _filterFeatures(whitelist, features_list):
    '''
    Filters out every features that isn't on the whitelist
    Args:
        whitelist: list of strings of the feature keys to keep
        features: list of dictionaries of features
    Returns:
        list of dictionaries of features
    '''
    reduced_list = []
    for user_feature_dict in features_list:
        reduced_user_feature_dict = {}
        for key in whitelist:
            if key in user_feature_dict:
                reduced_user_feature_dict[key] = user_feature_dict[key]
        reduced_list.append(reduced_user_feature_dict)
    return reduced_list

def main():

    parser = argparse.ArgumentParser(description='Problem Set 3')
    parser.add_argument('data_folder', help='path to data folder')
    parser.add_argument('-v', help='verbose mode', action="store_true")

    args = parser.parse_args()

    verbose_mode = bool(args.v)

    user_list = load_data(args.data_folder)

    calculated_features = calculate_features(user_list)

    # Generated list of names of FrequencyOfTweetingFeature's
    FrequencyOfTweetingFeature_NAMES = []
    for x in range(0, 48):
        FrequencyOfTweetingFeature_NAMES.append('FrequencyOfTweetingFeature_' + str(x))

    gender_whitelist = [
        'AverageTweetLength',
        #'NumberOfTimesOthersMentionedFeature',
        #'CountNouns',
        'CountEmotionalWords',
        #'CountNouns',
        'CountTweets',
        #'CountEmoticon',
        'CountLanguageUsed'
        #'Replacements',
        #'CountRegions'
    ] #+ FrequencyOfTweetingFeature_NAMES

    education_whitelist = [
        #'AverageTweetLength',
        #'CapitalizationFeature',
        'CountCategoricalWords',
        #'CountNouns',
        #'CountPunctuations',
        'CountTweets',
        'Occupation',
        #'CountEmotionalWords',
        #'CountHashTags',
        #'CountTweets',
        #'CountReplacements'
    ] #+ FrequencyOfTweetingFeature_NAMES

    age_whitelist = [
        #'AverageTweetLength',
        'CountCategoricalWords',
        'CountNouns',
        #'CapitalizationFeature',
        #'CountHashTags'
    ] #+ FrequencyOfTweetingFeature_NAMES


    age_bucket_whitelist = [
        'AverageTweetLength',
        'CountHashTags',
        'Occupation',
        #'CountCategoricalWords',
        'CountRetweet',
    ]# + FrequencyOfTweetingFeature_NAMES

    user_genders = []
    gender_features = []

    user_educations = []
    education_features = []

    user_ages = []
    age_features =[]

    user_age_buckets = []
    age_bucket_features = []

    for user, user_feature in zip(user_list, calculated_features):
        if user.gender == "Male":
            user_genders.append(0)
            gender_features.append(user_feature)
        elif user.gender == "Female":
            user_genders.append(1)
            gender_features.append(user_feature)
        if user.education != None:
            user_educations.append(user.education)
            education_features.append(user_feature)
        if user.year != None:
            user_ages.append(user.year)
            age_features.append(user_feature)
        if user.year != None:
            if user.year < 2015 and user.year >=1988:
                user_age_buckets.append(0)
                age_bucket_features.append(user_feature)
            elif user.year < 1988 and user.year > 1977:
                user_age_buckets.append(1)
                age_bucket_features.append(user_feature)
            elif user.year <= 1977:
                user_age_buckets.append(2)
                age_bucket_features.append(user_feature)

    if verbose_mode:
        print(len(calculated_features))
        print(len(user_list))
        print(len(user_genders))
        print(len(gender_features))
        print(len(user_educations))
        print(len(education_features))
        print(len(user_ages))
        print(len(age_features))
        print(len(user_age_buckets))
        print(len(age_bucket_features))
        print(user_ages)

    ## Test each feature one at a time for everything
    #feature_keys = gender_features[0].keys()
    #for feature_name in feature_keys:
    #    print('\n' + feature_name)
    #    # Test the accuracy
    #    _testAccuracy('gender', user_genders, _filterFeatures([feature_name], gender_features))
    #    _testAccuracy('education', user_educations, _filterFeatures([feature_name], education_features))
    #    _testAccuracy('age', user_ages, _filterFeatures([feature_name], age_features))
    #    _testAccuracy('age_buckets', user_age_buckets, _filterFeatures([feature_name], age_bucket_features))

    #print("Done with indiv testing.")
    #print("")
    # Filter out non-whitelist features
    gender_features = _filterFeatures(gender_whitelist, gender_features)
    education_features = _filterFeatures(education_whitelist, education_features)
    age_features = _filterFeatures(age_whitelist, age_features)
    age_bucket_features = _filterFeatures(age_bucket_whitelist, age_bucket_features)

    # Test the accuracy
    _testAccuracy('gender', user_genders, gender_features)
    _testAccuracy('education', user_educations, education_features)
    _testAccuracy('age', user_ages, age_features)
    _testAccuracy('age_buckets', user_age_buckets, age_bucket_features)

    # Find the best combinations
    # _testNFeaturesTogether(2, user_genders, gender_features)
    # _testNFeaturesTogether(2, user_educations, education_features)
    # _testNFeaturesTogether(2, user_ages, age_features)
    # _testNFeaturesTogether(2, user_age_buckets, age_bucket_features)

    # Find the best everything
    #_testAllFeatures(user_genders, gender_features)
<<<<<<< HEAD

    trainClassifier('gender', classifier.get_SVM, user_genders, gender_features)
    trainClassifier('education', classifier.get_Naivebayes, user_educations, education_features)
    trainClassifier('age', classifier.get_Naivebayes, user_ages, age_features)
    trainClassifier('age_buckets', classifier.get_LinearRegression, user_age_buckets, age_bucket_features)
=======
>>>>>>> origin/master

if __name__ == '__main__':
    main()
