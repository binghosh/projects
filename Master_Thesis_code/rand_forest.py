import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score

cols = ['ALTER_KONVERTER','LANZENALTER','ID_PROBE','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','KALKZUGABE','SLOPPING','GEWICHTE_1','GEWICHTE_2','GEWICHTE_3','GEWICHTE_4','GEWICHTE_5','GEWICHTE_6']
colsAttr = ['ALTER_KONVERTER','LANZENALTER','ID_PROBE','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','KALKZUGABE','GEWICHTE_1','GEWICHTE_2','GEWICHTE_3','GEWICHTE_4','GEWICHTE_5','GEWICHTE_6']
colsRes = ['SLOPPING']


df1 = pd.read_csv("ConvC_data_binary.csv",  nrows=269, header=1, names=cols)
#df2 = pd.read_csv("ConvC_data_binary.csv", nrows=269, header=1, names=colsRes)
df1.head()


np.nan_to_num(df1)
print (np.where(np.isnan(df1)))

dfArr1 = pd.DataFrame(df1, columns=colsAttr) #training array
dfRes1 = pd.DataFrame(df1, columns=colsRes) # training results

X_train1, X_test1, y_train1, y_test1 = train_test_split(dfArr1, dfRes1, test_size=0.2)

print (X_train1.shape, y_train1.shape)
print (X_test1.shape, y_test1.shape)

#y_trained = y_train.ravel()

rf1 = RandomForestClassifier(n_estimators=100) # initialize
rf1.fit(X_train1, ravel(y_train1)) # fit the data to the algorithm

predictions1 = rf1.predict(X_test1)
print (predictions1)
#print ("Score:", model.score(X_test, y_test))
score_rf = accuracy_score(y_test1, predictions1)
print (score_rf)

score_rf_cv = cross_val_score(rf1, X_train1, ravel(y_train1), cv=50)

print ("Accuracy_rf_cv: %0.2f (+/- %0.2f)" % (score_rf_cv.mean(), score_rf_cv.std() * 2))

accuracy_rf_cv = score_rf_cv.mean()