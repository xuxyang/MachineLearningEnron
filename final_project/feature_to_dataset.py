#!/usr/bin/python

def featureToDataset(features, labels, names, features_list):
    dataset_dict = {}
    for  i in range(len(names)):
        features_dict = {}
        features_dict['poi'] = labels[i]
        for j in range(len(features_list) - 1):
            features_dict[features_list[j + 1]] = features[i][j]
        dataset_dict[names[i]] = features_dict

    return dataset_dict
