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
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit


cols_static = ['SID', 'BEZ_SCHMELZE_ORG', 'SLS_SCHMELZEID', 'DT_BEGINN_IST', 'BEGINN_HBL', 'BEZ_KNV_IST', 'ALTER_KONVERTER', 'LANZENALTER', 'ID_PROBE', 'C', 'SI', 'TI', 'V', 'EINGELEERTES_RE', 'EINLEERGEWICHT_SC_GESAMT', 'AG', 'AW', 'NS', 'FB', 'SN', 'RS', 'FS', 'RE', 'MV', 'MUENDUNGSBAER', 'KALKZUGABE', 'GEWICHTE', 'KALK_ZUGABE_ZEITPUNKTE', 'MAX_INTENSITÄT_AUSWURF', 'ROUND(AUSWURF.DAUER)']
cols_staticAttr = ['ALTER_KONVERTER','LANZENALTER', 'C', 'SI', 'TI', 'V', 'EINGELEERTES_RE', 'EINLEERGEWICHT_SC_GESAMT', 'AG', 'AW', 'NS', 'FB', 'SN', 'RS', 'FS', 'RE', 'MV', 'KALKZUGABE']
#cols_staticAttr = ['ALTER_KONVERTER', 'LANZENALTER', 'C', 'SI', 'TI', 'V', 'EINGELEERTES_RE', 'EINLEERGEWICHT_SC_GESAMT', 'MV']
cols_staticRes = ['MAX_INTENSITÄT_AUSWURF']
#cols_staticRes = ['ROUND(AUSWURF.DAUER)']



df_static = pd.read_excel("Kopie_von_OAS_Datenauswertung_20170713.xlsx", sheetname="Tabelle1", header=1, names=cols_static)
#df1_static = pd.read_csv("static_data_500.csv",  header=1, names=cols_static)
#df2 = pd.read_csv("ConvC_data_binary.csv", nrows=269, header=1, names=cols_staticRes)
df_static.head()


#np.nan_to_num(df1_sv)
#print (np.where(np.isnan(df1_sv)))

dfArr1_sv = pd.DataFrame(df_static, columns=cols_staticAttr) #training array
dfRes1_sv = pd.DataFrame(df_static, columns=cols_staticRes) # training results

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=0) 

for train_index, test_index in sss.split(dfArr1_sv, dfArr1_sv):
    X_train1_sv, X_test1_sv = dfArr1_sv.iloc[train_index], dfArr1_sv.iloc[test_index]
    y_train1_sv, y_test1_sv = dfRes1_sv.iloc[train_index], dfRes1_sv.iloc[test_index]                  

print (X_train1_sv.shape, y_train1_sv.shape)
print (X_test1_sv.shape, y_test1_sv.shape)

#y_trained = y_train.ravel()


sv = SVC(probability=True) # initialize
sv.fit(X_train1_sv, np.ravel(y_train1_sv)) # fit the data to the algorithm

predictions_sv = sv.predict(X_test1_sv)
predictions_sv_prob = sv.predict_proba(X_test1_sv)
print (predictions_sv)
print (predictions_sv_prob) #No error
#print ("Score:", model.score(X_test, y_test))
score_sv = accuracy_score(y_test1_sv, predictions_sv)
print (score_sv)



#score_knn_cv = cross_val_score(knn, X_train1_sv, ravel(y_train1_sv), cv=50)

#print ("Accuracy_cv: %0.2f (+/- %0.2f)" % (score_knn_cv.mean(), score_knn_cv.std() * 2))

#accuracy_knn_cv = score_knn_cv.mean()