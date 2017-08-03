import pandas as pd
#import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
#from matplotlib import pyplot as plt
#from sklearn import model_selection
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import tree
#from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.externals import joblib
#from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
#from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import RandomizedLasso


cols_dynamic = ['Filename','Datum','Zeit','Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_Nm_min','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N','Tor_SW','Tor_SO','Tor_N','Digital_4','Digital_5','Digital_6','Schallpegel','Schlackenauswurf','Slopping','Time']
cols_dynamicAttr = ['Zeit','Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_Nm_min','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N', 'Tor_SW','Tor_SO','Tor_N','Digital_4','Digital_5','Digital_6','Schallpegel','Schallpegel','Schlackenauswurf']
cols_dynamicRes = ['Slopping']



df1_dynamic = pd.read_csv("Training_1.txt", sep="	", header=1, names=cols_dynamic)
#df1_dynamic = pd.read_csv("Dynamic_data_500.csv",  header=1, names=cols_dynamic)
#df2 = pd.read_csv("ConvC_data_binary.csv", nrows=269, header=1, names=cols_dynamicRes)
df1_dynamic.head()


#np.nan_to_num(df1_dynamic)
#print (np.where(np.isnan(df1_dynamic)))

dfArr1_dynamic = pd.DataFrame(df1_dynamic, columns=cols_dynamicAttr) #training array
dfRes1_dynamic = pd.DataFrame(df1_dynamic, columns=cols_dynamicRes) # training results

#X_train1_dynamic, X_test1_dynamic, y_train1_dynamic, y_test1_dynamic = train_test_split(dfArr1_dynamic, dfRes1_dynamic, test_size=0.2)

#print (X_train1_dynamic.shape, y_train1_dynamic.shape)
#print (X_test1_dynamic.shape, y_test1_dynamic.shape)


#feat_extr = SelectKBest(k=7)
#fitter = feat_extr.fit(dfArr1_dynamic, ravel(dfRes1_dynamic))

#scores1 = fitter.scores_

#scores = pd.DataFrame(fitter.scores_, index=cols_dynamicAttr)

#model = ExtraTreesClassifier()
#model = model.fit(dfArr1_dynamic, ravel(dfRes1_dynamic))

#model_scores = pd.DataFrame(model.feature_importances_, index=cols_dynamicAttr)


#rlasso = RandomizedLasso()
#lasso = rlasso.fit(dfArr1_dynamic, ravel(dfRes1_dynamic))

#lasso_scores = pd.DataFrame(lasso.scores_, index=cols_dynamicAttr)


ard = linear_model.ARDRegression(compute_score=True)
autorelevdet = ard.fit(dfArr1_dynamic, ravel(dfRes1_dynamic))

ard_scores = pd.DataFrame(autorelevdet.scores_, index=cols_dynamicAttr)

ard_coef = pd.DataFrame(autorelevdet.coef_, index=cols_dynamicAttr)


