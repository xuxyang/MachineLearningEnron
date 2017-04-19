#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

#from feature_format import featureFormat, targetFeatureSplit
from my_feature_format import myFeatureFormat, myTargetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as pl
from sklearn.feature_selection import SelectKBest
from pretty_picture import prettyPicture, featureRelationPicture, classifierBoundary
from sklearn.grid_search import GridSearchCV
import math
from sklearn.metrics import f1_score, make_scorer, fbeta_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from get_support import get_support_features

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
org_features_list = ['poi','fraction_shared', 'fraction_from_poi', 'fraction_to_poi', 'salary','exercised_stock_options','bonus','restricted_stock','expenses','loan_advances','other','director_fees','long_term_incentive','restricted_stock_deferred','deferred_income','deferral_payments']
#org_features_list = ['poi', 'salary','exercised_stock_options','bonus','restricted_stock','expenses','loan_advances','other','director_fees','long_term_incentive','restricted_stock_deferred','deferred_income','deferral_payments']
print(len(org_features_list))

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print('length of dataset: %d' % len(data_dict))
#print(data_dict['PICKERING MARK R'])
#print(data_dict['ALLEN PHILLIP K'])
poi_count = 0
for key,value in data_dict.iteritems():
    if value['poi'] == True:
        poi_count += 1
        #print(key + (' salary %d'%value['salary']))
print('poi count: %d' % poi_count)
print(data_dict['LAY KENNETH L'])
#print(data_dict['SKILLING JEFFREY K'])
print(data_dict['THE TRAVEL AGENCY IN THE PARK'])

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
my_dataset.pop('TOTAL')
my_dataset.pop('THE TRAVEL AGENCY IN THE PARK')

test_dataset = {} 

non_poi_to_poi_ratio = 7
org_keys = my_dataset.keys()

for name in org_keys:
    person_data = my_dataset[name]
    if name == 'BELFER ROBERT':
        person_data['deferred_income'] = person_data['deferral_payments']
        person_data['deferral_payments'] = 0.0
        person_data['total_stock_value'] = 0.0
    if person_data['deferred_income'] < 0:
        person_data['deferred_income'] = abs(person_data['deferred_income'])
    if person_data['deferral_payments'] < 0:
        person_data['deferral_payments'] = abs(person_data['deferral_payments'])
    if person_data['restricted_stock_deferred'] < 0:
        person_data['restricted_stock_deferred'] = abs(person_data['restricted_stock_deferred'])
    person_data['fraction_from_poi'] = computeFraction(person_data['from_poi_to_this_person'], person_data['to_messages'])
    person_data['fraction_to_poi'] = computeFraction(person_data['from_this_person_to_poi'], person_data['from_messages'])
    person_data['fraction_shared'] = computeFraction(person_data['shared_receipt_with_poi'], person_data['to_messages'])
    test_dataset[name] = person_data.copy()
    if person_data['poi'] == True:
        for i in range(0, non_poi_to_poi_ratio - 1):
            my_dataset[name +str(i)] = person_data # my_dataset will be the over-sampled dataset for poi


### Extract features and labels from dataset for local testing
##data = featureFormat(my_dataset, org_features_list, sort_keys = True)
##labels, features = targetFeatureSplit(data)
data = myFeatureFormat(test_dataset, org_features_list, random_keys = False)
#data = myFeatureFormat(my_dataset, org_features_list, random_keys = False)
names, labels, features = myTargetFeatureSplit(data)
print('data size: %d' % len(labels))

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
kbest = SelectKBest(k=4)
kbest.fit(scaled_features,labels)
print(kbest.get_support())
print(kbest.scores_)
##featureRelationPicture(features, [0,1], features_list)

std_features = StandardScaler().fit_transform(features)
pca = PCA(n_components=12)
pca.fit(std_features)
print(pca.components_)
print(pca.explained_variance_ratio_) # 5 components > 0.7

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

#pipe = make_pipeline(MinMaxScaler(), SelectKBest(k=4), SVC())
pipe = make_pipeline(MinMaxScaler(), SelectKBest(k=4), DecisionTreeClassifier(random_state=42))
#pipe = make_pipeline(StandardScaler(), PCA(n_components=5), SVC())
#pipe = make_pipeline(StandardScaler(), PCA(n_components=5), DecisionTreeClassifier(random_state=42))

#parameters = [dict(svc__C=[0.03125,0.125,0.5,2,6,8,32,36], svc__kernel=['poly'], svc__gamma=[0.0078125,0.03125,0.125,0.5,2,8,10,12,16], svc__degree=[2,3,4,5]),
#              dict(svc__C=[0.03125,0.125,0.5,2,6,8,32,36], svc__kernel=['rbf'], svc__gamma=[0.0078125,0.03125,0.125,0.5,2,8,10,12,16])]

##final parameters for decision tree 
parameters = dict(decisiontreeclassifier__min_samples_leaf=[2,3,4,5,10,20,30], decisiontreeclassifier__criterion=['entropy','gini'])

####final parameters for svc 
##parameters = [dict(svc__C=[0.5,2,6,8,32,36,40], svc__kernel=['poly'], svc__gamma=[0.5,2,8,10,12,16,20], svc__degree=[2,3,4,5]),
##              dict(svc__C=[0.5,2,6,8,32,36,40], svc__kernel=['rbf'], svc__gamma=[0.5,2,8,10,12,16,20])]

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
##    train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
##print(sum(labels_train))
##print(len(labels_train))
##print(sum(labels_test))
##print(len(labels_test))

#grid_cv = StratifiedKFold(labels, n_folds=8, shuffle=True, random_state=42) # 4 folds for the original dataset and 8 folds for over-sampled dataset
grid_cv = StratifiedShuffleSplit(labels, 100, random_state = 123, test_size=0.3)
#grid_cv = StratifiedShuffleSplit(labels_train, 100, random_state = 42, test_size=0.3)
grid = GridSearchCV(pipe, parameters, n_jobs=1, scoring=f2_scorer, cv=grid_cv)
#grid.fit(features_train, labels_train)
grid.fit(features, labels)
support_list = grid.best_estimator_.named_steps['selectkbest'].get_support(indices=True)
support_features = get_support_features(support_list, org_features_list)
best_params = grid.best_params_
print(grid.best_score_)
print(best_params)
print(support_features)
print(grid.best_estimator_.named_steps['selectkbest'].scores_)
print(grid.best_estimator_.named_steps['decisiontreeclassifier'].feature_importances_)
##print(grid.best_estimator_.named_steps['pca'].explained_variance_ratio_)

##predict = grid.predict(features_test)
##print(accuracy_score(labels_test, predict))
##print(precision_score(labels_test, predict))
##print(recall_score(labels_test, predict))


features_list = support_features 
##if best_params['svc__kernel'] == 'rbf':
##    svc = SVC(C=best_params['svc__C'],kernel=best_params['svc__kernel'],gamma=best_params['svc__gamma'])
##elif best_params['svc__kernel'] == 'poly':
##    svc = SVC(C=best_params['svc__C'],kernel=best_params['svc__kernel'],gamma=best_params['svc__gamma'],degree=best_params['svc__degree'])
##clf = make_pipeline(MinMaxScaler(), svc) #100 folds and 0.3 test size shuffle split; fraction_to_poi, salary, bonus and exercised_stock_options features; ploy kernel with c=40, gamma=20, degree=4; accuracy 0.83369, precision 0.44788 and recall 0.348
##features_list = ['poi','fraction_from_poi', 'fraction_to_poi', 'salary','bonus'] #use 8 folds and random_state 42 and svc for GridSearchCV with precision 0.45987 and recall 0.4355 and accuracy 0.80436
##clf = make_pipeline(MinMaxScaler(), SVC(C=32,kernel='poly',gamma=16,degree=3))
clf = DecisionTreeClassifier(min_samples_leaf=best_params['decisiontreeclassifier__min_samples_leaf'], criterion=best_params['decisiontreeclassifier__criterion']) #100 folds and 0.3 test size shuffle split;fraction_to_poi, salary, bonus and exercised_stock_options features; min_samples_leaf=10, entropy criteria;0.84277 accuracy, 0.48497 precision and 0.355 recall

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

##plot classifier boundary
features_list = ['poi', 'fraction_to_poi', 'bonus'] # reduce features from 4 to the top 2 important features (fraction_to_poi > 0.5 and bonus > 0.3): accuracy 0.8177, precision 0.55713 and recall 0.4315
data = myFeatureFormat(test_dataset, features_list, random_keys = False)
names, labels, features = myTargetFeatureSplit(data)
scaled_features = MinMaxScaler().fit_transform(features)
clf.fit(scaled_features, labels)
classifierBoundary(clf, features_list)


dump_classifier_and_data(clf, test_dataset, features_list)


