"""
Testing the classifier
"""
import pickle
import argparse
import nltk
import problemset3

def testAccuracy(classifier, display_type, features):
    '''
    Tests the accuracy, prints results.
    '''

    ACC_STRING = '\t{0} accuracy: {1}'
    
    acc = nltk.classify.accuracy(classifier, features)
    print(ACC_STRING.format(display_type, acc))

    print("")

def main():
    parser = argparse.ArgumentParser(description='Problem Set 3')
    parser.add_argument('data_folder', help='path to data folder')
    parser.add_argument('-v', help='verbose mode', action="store_true")

    args = parser.parse_args()

    user_list = problemset3.load_data(args.data_folder)
    calculated_features = problemset3.calculate_features(user_list)
    
    f = open('gender.pickle', 'rb')
    gender_classifier = pickle.load(f)
    f.close()

    f = open('education.pickle', 'rb')
    education_classifier = pickle.load(f)
    f.close()
    
    f = open('age.pickle', 'rb')
    age_classifier = pickle.load(f)
    f.close()
    
    f = open('age_bracket.pickle', 'rb')
    age_buckets_classifier = pickle.load(f)
    f.close()
    
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

    gender_features = problemset3._filterFeatures(problemset3.gender_whitelist, gender_features)
    education_features = problemset3._filterFeatures(problemset3.education_whitelist, education_features)
    age_features = problemset3._filterFeatures(problemset3.age_whitelist, age_features)
    age_bucket_features = problemset3._filterFeatures(problemset3.age_bucket_whitelist, age_bucket_features)
    
    testAccuracy(gender_classifier,'gender', gender_features)
    testAccuracy(education_classifier,'education', education_features)
    testAccuracy(age_classifier,'age', age_features)
    testAccuracy(age_buckets_classifier,'age_buckets', age_bucket_features)
    
if __name__ == '__main__':
    main()