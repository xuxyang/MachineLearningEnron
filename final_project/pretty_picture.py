#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def prettyPicture(X_test, y_test, features_list):
    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    plt.clf()
    plt.scatter(grade_sig, bumpy_sig, color = "b", label="non-poi")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="poi")
    plt.legend()
    plt.xlabel(features_list[1])
    plt.ylabel(features_list[2])

    plt.savefig("test.png")

def featureRelationPicture(X_test, comparingFeaturesIndex_list,  features_list):
    sorted_X = sorted(X_test, key=lambda x: x[comparingFeaturesIndex_list[0]])
    feature_one_value = [sorted_X[i][comparingFeaturesIndex_list[0]] for i in range(0, len(X_test))]
    feature_two_value = [sorted_X[i][comparingFeaturesIndex_list[1]] for i in range(0, len(X_test))]
    x_axis_value = range(0, len(X_test))

    plt.clf()
    plt.plot(x_axis_value, feature_one_value, color = "b", label=features_list[comparingFeaturesIndex_list[0]+1])
    plt.plot(x_axis_value, feature_two_value, color = "r", label=features_list[comparingFeaturesIndex_list[1]+1])
    plt.legend()
    plt.xlabel("Person")
    plt.ylabel("dollar")

    plt.savefig("featureRelation.png")

def classifierBoundary(clf, features_list):
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

##    # Plot also the test points
##    featureOne_nonpoi = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
##    featureTwo_nonpoi = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
##    featureOne_poi = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
##    featureTwo_poi = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]
##
##    plt.scatter(featureOne_nonpoi, featureTwo_nonpoi, color = "b", label="ono-poi")
##    plt.scatter(featureOne_poi, featureTwo_poi, color = "r", label="poi")
    plt.legend()
    plt.xlabel(features_list[1])
    plt.ylabel(features_list[2])

    plt.savefig("classifierBoundary.png")
    
import base64
import json
import subprocess

def output_image(name, format, bytes):
    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {}
    data['name'] = name
    data['format'] = format
    data['bytes'] = base64.encodestring(bytes)
    print image_start+json.dumps(data)+image_end
                                    
