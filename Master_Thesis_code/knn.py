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
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

cols = ['ALTER_KONVERTER','LANZENALTER','ID_PROBE','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','KALKZUGABE','SLOPPING','GEWICHTE_1','GEWICHTE_2','GEWICHTE_3','GEWICHTE_4','GEWICHTE_5','GEWICHTE_6']
colsAttr = ['ALTER_KONVERTER','LANZENALTER','ID_PROBE','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','KALKZUGABE','GEWICHTE_1','GEWICHTE_2','GEWICHTE_3','GEWICHTE_4','GEWICHTE_5','GEWICHTE_6']
colsRes = ['SLOPPING']


df1 = pd.read_csv("ConvC_data_binary.csv",  nrows=269, header=1, names=cols)
#df2 = pd.read_csv("ConvC_data_binary.csv", nrows=269, header=1, names=colsRes)
df1.head()


#np.nan_to_num(df1)
#print (np.where(np.isnan(df1)))

dfArr1 = pd.DataFrame(df1, columns=colsAttr) #training array
dfRes1 = pd.DataFrame(df1, columns=colsRes) # training results

X_train1, X_test1, y_train1, y_test1 = train_test_split(dfArr1, dfRes1, test_size=0.2)

print (X_train1.shape, y_train1.shape)
print (X_test1.shape, y_test1.shape)

#y_trained = y_train.ravel()

knn = KNeighborsClassifier() # initialize
knn.fit(X_train1, ravel(y_train1)) # fit the data to the algorithm

predictions_knn = knn.predict(X_test1)
print (predictions_knn)
#print ("Score:", model.score(X_test, y_test))
score_knn = accuracy_score(y_test1, predictions_knn)
print (score_knn)

score_knn_cv = cross_val_score(knn, X_train1, ravel(y_train1), cv=50)

print ("Accuracy_cv: %0.2f (+/- %0.2f)" % (score_knn_cv.mean(), score_knn_cv.std() * 2))

accuracy_knn_cv = score_knn_cv.mean()

prob_static_knn = knn.predict_proba(X_test1)

print(knn.predict_proba(X_test1))