import nltk
from nltk.corpus import names
import random
from string import ascii_lowercase

def gender_features(word):
    features = {'first two letters': word[:2], 'last letter': word[-1], 'first letter': word[0], 'length': len(word), 'last two': word[-2:], 'last three': word[-3:]}
    for letter in ascii_lowercase:
        features["count(%s)" % letter] = word.lower().count(letter)
    return features

def main():
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
