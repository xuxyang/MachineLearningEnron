#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import math

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
#print len(enron_data)
keys = enron_data.keys()
##print keys[0]
##print enron_data[keys[1]]["email_address"]
##print len(enron_data[keys[1]].keys())
#print len(keys)

poi_count = 0
for key in keys:
    if enron_data[key]["poi"] == 1:
        poi_count += 1


print "poi count: ", poi_count

##name_count = 0
##names_file = open("../final_project/poi_names.txt", "r")
##for line in names_file:
##    if line.startswith("(y)") or line.startswith("(n)"):
##        name_count += 1
##
##
##print name_count

##for key in keys:
##    if key.startswith("PRENTICE JAMES"):
##        print enron_data[key]["total_stock_value"]

##for key in keys:
##    if key.startswith("COLWELL WESLEY"):
##        print enron_data[key]["from_this_person_to_poi"]

##for key in keys:
##    if key.startswith("SKILLING JEFFREY"):
##        print enron_data[key]["exercised_stock_options"]

##print enron_data["LAY KENNETH L"]["total_payments"]
##print enron_data["SKILLING JEFFREY K"]["total_payments"]
##print enron_data["FASTOW ANDREW S"]["total_payments"]

##has_salary_count = 0
##for key in keys:
##    if not math.isnan(float(enron_data[key]["salary"])):
##        has_salary_count += 1
##
##print has_salary_count

##has_email_count = 0
##for key in keys:
##    if enron_data[key]["email_address"] != "NaN":
##        has_email_count += 1
##
##print has_email_count

no_total_payments_count = 0
for key in keys:
    if enron_data[key]["total_payments"] == "NaN":
        no_total_payments_count += 1

print no_total_payments_count
print len(keys)
print (no_total_payments_count * 1.0) / (len(keys) * 1.0)

no_total_payments_poi_count = 0
for key in keys:
    if enron_data[key]["total_payments"] == "NaN" and enron_data[key]["poi"] == 1:
        no_total_payments_poi_count += 1

print no_total_payments_poi_count
print (no_total_payments_poi_count * 1.0) / (poi_count * 1.0)
    


