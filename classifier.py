'''
Generic script for classifier:
Classifier = SVM linear model

It takes four parameters as input:
    a(list) = training data which is dictionary of features
    b(np.array) = ground truth of features(associated class)
    c(list) = test data which is dictionay of features
    d(np.array) = ground truth of test data
'''
import pickle

def saveClassifier(classifier,display_type):
    # Save the classifier in a pickle file
    filename = display_type + '.pickle'
    f = open(filename, 'wb')
    pickle.dump(classifier, f)
    f.close()

def get_SVM(a,b,c,d,display_name):
    # Convert features into vector of numbers
    from sklearn.feature_extraction import DictVectorizer
    v1 = DictVectorizer().fit(a+c)

    #define training data
    X_data_tr = v1.transform(a)
    Y_data_tr = b

    #import linear SVM
    from sklearn.svm import LinearSVC

    #generate model
    svm_Classifier = LinearSVC().fit(X_data_tr, Y_data_tr)
    saveClassifier(svm_Classifier,display_name)


def get_SVM_Acc(a,b,c):

    # Convert features into vector of numbers
    from sklearn.feature_extraction import DictVectorizer
    v1 = DictVectorizer().fit(a+c)

    #define training data
    X_data_tr = v1.transform(a)
    Y_data_tr = b

    #define test data
    X_data_ts = v1.transform(c)

    #import linear SVM
    from sklearn.svm import LinearSVC

    #generate model
    svm_Classifier = LinearSVC().fit(X_data_tr, Y_data_tr)

    #Use trained model to classify test data
    Y_pred = svm_Classifier.predict(X_data_ts)

    return Y_pred

def get_Naivebayes(a,b,c,d,display_name):

    # Convert features into vector of numbers
    from sklearn.feature_extraction import DictVectorizer
    v1 = DictVectorizer().fit(a+c)

    #define training data
    X_data_tr = v1.transform(a)
    Y_data_tr = b

    #import Naive bayes classifier
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_data_tr,Y_data_tr)
    saveClassifier(clf,display_name)

def get_Naivebayes_Acc(a,b,c):

    # Convert features into vector of numbers
    from sklearn.feature_extraction import DictVectorizer
    v1 = DictVectorizer().fit(a+c)

    #define training data
    X_data_tr = v1.transform(a)
    Y_data_tr = b

    #define test data
    X_data_ts = v1.transform(c)

    #import Naive bayes classifier
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_data_tr,Y_data_tr)

    #Use trained model to classify test data
    Y_pred = clf.predict(X_data_ts)

    return Y_pred
    #print(len(clf.classes_))
    #print(clf.classes_)
    #most_informative_feature_for_class(v1,clf, clf.classes_[0])
    #most_informative_feature_for_class(v1,clf, clf.classes_[1])

    #from sklearn.metrics import confusion_matrix
    #print(confusion_matrix(Y_data_ts,Y_pred))


def get_LinearRegression(a,b,c,d,display_name):

    # Convert features into vector of numbers
    from sklearn.feature_extraction import DictVectorizer
    v1 = DictVectorizer().fit(a+c)

    #define training data
    X_data_tr = v1.transform(a)
    Y_data_tr = b

    #import Linear Regression classifier
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(X_data_tr,Y_data_tr)
    saveClassifier(regr,display_name)

def get_LinearRegression_Acc(a,b,c):

    # Convert features into vector of numbers
    from sklearn.feature_extraction import DictVectorizer
    v1 = DictVectorizer().fit(a+c)

    #define training data
    X_data_tr = v1.transform(a)
    Y_data_tr = b

    #define test data
    X_data_ts = v1.transform(c)

    #import Linear Regression classifier
    import numpy as np
    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    regr.fit(X_data_tr,Y_data_tr)

    #Use trained model to classify test data
    Y_pred = regr.predict(X_data_ts)
    # Convert into nearest integer
    Y_pred = np.rint(Y_pred)

    return Y_pred

    #from sklearn.metrics import confusion_matrix
    #print(confusion_matrix(Y_data_ts,Y_pred))


def get_LinearRegression_class(regr, c):

    # Convert features into vector of numbers
    from sklearn.feature_extraction import DictVectorizer
    v1 = DictVectorizer().fit(c)

    #define test data
    X_data_ts = v1.transform(c)

    import numpy as np
    #import Linear Regression classifier
    #Use trained model to classify test data
    Y_pred = regr.predict(X_data_ts)
    # Convert into nearest integer
    Y_pred = np.rint(Y_pred)
    return Y_pred

def get_Naivebayes_class(clf,c):

    # Convert features into vector of numbers
    from sklearn.feature_extraction import DictVectorizer
    v1 = DictVectorizer().fit(c)

    X_data_ts = v1.transform(c)

    #import Naive bayes classifier
    #Use trained model to classify test data
    Y_pred = clf.predict(X_data_ts)
    return Y_pred

    #print(len(clf.classes_))
    #print(clf.classes_)
    #most_informative_feature_for_class(v1,clf, clf.classes_[0])
    #most_informative_feature_for_class(v1,clf, clf.classes_[1])

    #from sklearn.metrics import confusion_matrix
    #print(confusion_matrix(Y_data_ts,Y_pred))


def get_SVM_class(svm_Classifier,c):

    # Convert features into vector of numbers
    from sklearn.feature_extraction import DictVectorizer
    v1 = DictVectorizer().fit(c)

    X_data_ts = v1.transform(c)


    #Use trained model to classify test data
    Y_pred = svm_Classifier.predict(X_data_ts)
    return Y_pred


def most_informative_feature_for_class(vectorizer, classifier, classlabel, n=10):
    labelid = list(classifier.classes_).index(classlabel)
    feature_names = vectorizer.get_feature_names()
    topn = sorted(zip(classifier.coef_[labelid], feature_names))[-n:]

    for coef, feat in topn:
        print(classlabel, feat, coef)


'''
    Function to generate output files(.txt)
        Parameters:
            usrnames(list)- contains usenames of test data.
            Y_pred(list)- Predicted values for test data.
            filename(string) - 'gender' or 'age' or 'education'
'''

def createTextFiles(usrnames,Y_pred,filename, age_bucket_ypred=[]):
    if filename == 'gender':
        file_name = filename+'.txt'
        file1 = open(file_name,'w')
        for i in range(0,len(Y_pred)):
            file1.write(str(usrnames[i]))
            file1.write("\t")
            if Y_pred[i]==0:
                file1.write('Male')
            elif Y_pred[i]==1:
                file1.write('Female')
            file1.write("\n")

        file1.close()

    elif filename == 'education':
        file_name = filename+ '.txt'
        file1 = open(file_name,'w')
        for i in range(0,len(Y_pred)):
            file1.write(str(usrnames[i]))
            file1.write("\t")
            if Y_pred[i]==0:
                file1.write('high_school')
            elif Y_pred[i]==1:
                file1.write('some_college')
            elif Y_pred[i]==2:
                file1.write('graduate')
            file1.write("\n")
        file1.close()

    elif filename == 'age':
        file_name = filename+ '.txt'
        file1 = open(file_name,'w')
        for i in range(0,len(Y_pred)):
            file1.write(str(usrnames[i]))
            file1.write("\t")
            if age_bucket_ypred[i] == 0:
                file1.write('<=25')
            elif age_bucket_ypred[i] == 1:
                file1.write('26-35')
            elif age_bucket_ypred[i] == 2:
                file1.write('>=36')
            file1.write("\t")
            file1.write(str(Y_pred[i]))
            file1.write("\n")
        file1.close()


'''
Example:-

import pickle
feature_dict = pickle.load( open('bass.X.pkl', 'r') )
Y_data = pickle.load( open('bass.y.pkl', 'r') )
a1 = feature_dict[:900]
b1 = Y_data[:900]
c1 = feature_dict[900:]
d1 = Y_data[900:]

acc = get_SVM_Acc(a1,b1,c1,d1)

Example to generate .txt files:

usrname_temp = [12,13,14,15,16,17,18,19,20,21]
Y_pred_gen = [1,0,1,0,1,0,1,1,0,0]
Y_pred_age = [1900,1989,2000,1985,1988,1934,2004,1970,1979,2012]
Y_pred_edu = [1,0,1,2,0,1,2,0,1,2]

createTextFiles(usrname_temp,Y_pred_edu,'education')


'''
