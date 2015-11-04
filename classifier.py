'''
Generic script for classifier:
Classifier = SVM linear model 

It takes four parameters as input:
    a(list) = training data which is dictionary of features
    b(np.array) = ground truth of features(associated class)
    c(list) = test data which is dictionay of features
    d(np.array) = ground truth of test data 
'''

def get_SVM_Acc(a,b,c,d):

   # Convert features into vector of numbers
   from sklearn.feature_extraction import DictVectorizer   
   v1 = DictVectorizer().fit(a+c) 
  
   #define training data
   X_data_tr = v1.transform(a)
   Y_data_tr = b
   
   #define test data
   X_data_ts = v1.transform(c)
   Y_data_ts = d

   #import linear SVM   
   from sklearn.svm import LinearSVC
   
   #generate model 
   svm_Classifier = LinearSVC().fit(X_data_tr, Y_data_tr)
   
   #Use trained model to classify test data
   Y_pred = svm_Classifier.predict(X_data_ts)

   acc = (Y_pred==Y_data_ts).mean()

   return acc


def get_Naivebayes_Acc(a,b,c,d):

   # Convert features into vector of numbers
   from sklearn.feature_extraction import DictVectorizer   
   v1 = DictVectorizer().fit(a+c) 
  
   #define training data
   X_data_tr = v1.transform(a)
   Y_data_tr = b
   
   #define test data
   X_data_ts = v1.transform(c)
   Y_data_ts = d

   #import Naive bayes classifier   
   from sklearn.naive_bayes import MultinomialNB
   clf = MultinomialNB()
   clf.fit(X_data_tr,Y_data_tr)
   
   #Use trained model to classify test data
   Y_pred = clf.predict(X_data_ts)

   acc = (Y_pred==Y_data_ts).mean()
   
   #from sklearn.metrics import confusion_matrix
   #print(confusion_matrix(Y_data_ts,Y_pred))
   
   

   return acc
   
def get_LinearRegression_Acc(a,b,c,d):

   # Convert features into vector of numbers
   from sklearn.feature_extraction import DictVectorizer   
   v1 = DictVectorizer().fit(a+c) 
  
   #define training data
   X_data_tr = v1.transform(a)
   Y_data_tr = b
   
   #define test data
   X_data_ts = v1.transform(c)
   Y_data_ts = d


   #import Linear Regression classifier   
   import numpy as np
   from sklearn import linear_model
   regr = linear_model.LinearRegression()
   regr.fit(X_data_tr,Y_data_tr)
   
   #Use trained model to classify test data
   Y_pred = regr.predict(X_data_ts)
   # Convert into nearest integer 
   Y_pred = np.rint(Y_pred)

   acc = (Y_pred==Y_data_ts).mean()
   
   #from sklearn.metrics import confusion_matrix
   #print(confusion_matrix(Y_data_ts,Y_pred))

   return acc
   
   
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
   
   
'''
   
   
