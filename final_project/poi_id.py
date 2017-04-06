#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

#from feature_format import featureFormat, targetFeatureSplit
from my_feature_format import myFeatureFormat, myTargetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import preprocessing

import matplotlib.pyplot as pl
from sklearn.feature_selection import SelectKBest
from pretty_picture import prettyPicture
from sklearn.grid_search import GridSearchCV
import math
from sklearn.metrics import f1_score, make_scorer, fbeta_score

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """


    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    
    if math.isnan(float(poi_messages)) or math.isnan(float(all_messages)):
        fraction = 0.
    else:
        fraction = (poi_messages * 1.0) / (all_messages * 1.0)



    return fraction


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi', 'total_payments', 'total_stock_value', 'fraction_from_poi', 'fraction_to_poi']
features_list = ['poi', 'salary','bonus', 'fraction_from_poi', 'fraction_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

##print('length of dataset: %d' % len(data_dict))
#print(data_dict['PICKERING MARK R'])
print(data_dict['ALLEN PHILLIP K'])
##poi_count = 0
##for key,value in data_dict.iteritems():
##    if value['poi'] == True:
##        print(key + (' salary %d'%value['salary']))
##print('poi count: %d' % poi_count)
#print(data_dict['LAY KENNETH L'])
#print(data_dict['SKILLING JEFFREY K'])

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
my_dataset.pop('TOTAL')

error_total_stock_count = 0
for name in my_dataset:
    person_data = my_dataset[name]
    if person_data['total_stock_value'] < 0:
        error_total_stock_count += 1
        person_data['total_stock_value'] = 0.0
    person_data['fraction_from_poi'] = computeFraction(person_data['from_poi_to_this_person'], person_data['to_messages'])
    person_data['fraction_to_poi'] = computeFraction(person_data['from_this_person_to_poi'], person_data['from_messages'])
print(my_dataset['BELFER ROBERT'])
print(my_dataset['CARTER REBECCA C'])
print(error_total_stock_count)
    

### Extract features and labels from dataset for local testing
data = myFeatureFormat(my_dataset, features_list, sort_keys = True)
names, labels, features = myTargetFeatureSplit(data)
print('data size: %d' % len(labels))

scaler = preprocessing.MinMaxScaler()
scaled_features = scaler.fit_transform(features)

##selector = SelectKBest(k=2)
##selector.fit(scaled_features, labels)
##features_selected = selector.transform(scaled_features)
#print(features_selected)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
##from sklearn.naive_bayes import GaussianNB
##clf = GaussianNB()

f1_scorer = make_scorer(f1_score)
f2_scorer = make_scorer(fbeta_score, beta=2)

#SVC kernel definition: http://scikit-learn.org/dev/modules/svm.html#kernel-functions
from sklearn.svm import SVC
#parameters = {'C':[0.1,0.3,0.5,0.8,1,3,5,7,10], 'kernel':['poly', 'rbf', 'sigmoid'], 'gamma':[0.1,0.2,0,3,0.5,0.8,1,3,5,7,10], 'degree':[2,3,5,10]}
#parameters = {'C':[0.03125,0.125,0.5,2,8,32], 'kernel':['rbf'], 'gamma':[0.0078125,0.03125,0.125,0.5,2,8]}
#parameters = {'C':[5,6,7], 'kernel':['rbf'], 'gamma':[11,12,13]} # the best parameter for rbf kernel so far is C=6 and gamma=12 with 0.51908 precision and 0.238 recall
parameters = {'C':[0.03125,0.125,0.5,2,8,32], 'kernel':['poly'], 'gamma':[0.0078125,0.03125,0.125,0.5,2,8], 'degree':[2,3,5]} # best parameter for poly kernel so far is C=8 and gamma=8 and degree=5 with 0.48428 precision and 0.323 recall
svr = SVC()
clf = GridSearchCV(svr, parameters, scoring=f2_scorer)
#clf = SVC(C=5, kernel='rbf', gamma=2)

##from sklearn.tree import DecisionTreeClassifier
##parameters = {'min_samples_split':[2,3,4,5,25,30], 'criterion':['entropy']}
##decisionTree = DecisionTreeClassifier()
##clf = GridSearchCV(decisionTree, parameters, scoring=f1_scorer)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#test_classifier(clf, my_dataset, features_list)

# Example starting point. Try investigating other evaluation techniques!
##from sklearn.cross_validation import train_test_split
##features_train, features_test, labels_train, labels_test = \
##    train_test_split(features, labels, test_size=0.3, random_state=42)
##
##clf.fit(features_train, labels_train)
prettyPicture(scaled_features, labels, features_list)
#print(clf.best_params_)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

from feature_to_dataset import featureToDataset
scaled_dataset = featureToDataset(scaled_features, labels, names, features_list)
print(scaled_dataset)
dump_classifier_and_data(clf, scaled_dataset, features_list)
##from my_tester import dump_classifier_and_data
##dump_classifier_and_data(clf, scaled_features, labels)


