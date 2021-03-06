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
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import RandomizedLasso

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier


cols_static = ['SID', 'BEZ_SCHMELZE_ORG', 'SLS_SCHMELZEID', 'DT_BEGINN_IST', 'BEGINN_HBL', 'BEZ_KNV_IST', 'ALTER_KONVERTER', 'LANZENALTER', 'ID_PROBE', 'C', 'SI', 'TI', 'V', 'EINGELEERTES_RE', 'EINLEERGEWICHT_SC_GESAMT', 'AG', 'AW', 'NS', 'FB', 'SN', 'RS', 'FS', 'RE', 'MV', 'MUENDUNGSBAER', 'KALKZUGABE', 'GEWICHTE', 'KALK_ZUGABE_ZEITPUNKTE', 'MAX_INTENSITÄT_AUSWURF', 'ROUND(AUSWURF.DAUER)']
cols_staticAttr = ['ALTER_KONVERTER','LANZENALTER', 'C', 'SI', 'TI', 'V', 'EINGELEERTES_RE', 'EINLEERGEWICHT_SC_GESAMT', 'AG', 'AW', 'NS', 'FB', 'SN', 'RS', 'FS', 'RE', 'MV', 'KALKZUGABE']
#cols_staticAttr = ['ALTER_KONVERTER', 'LANZENALTER', 'C', 'SI', 'TI', 'V', 'EINGELEERTES_RE', 'EINLEERGEWICHT_SC_GESAMT', 'MV']
cols_staticRes = ['MAX_INTENSITÄT_AUSWURF']
#cols_staticRes = ['ROUND(AUSWURF.DAUER)']



df_static = pd.read_excel("Kopie_von_OAS_Datenauswertung_20170713_binary.xlsx", sheetname="Tabelle1", header=1, names=cols_static)
#df1_static = pd.read_csv("static_data_500.csv",  header=1, names=cols_static)
#df2 = pd.read_csv("ConvC_data_binary.csv", nrows=269, header=1, names=cols_staticRes)
df_static.head()


#np.nan_to_num(df1_static)
#print (np.where(np.isnan(df1_static)))

dfAttr_static = pd.DataFrame(df_static, columns=cols_staticAttr) #training array
dfRes_static = pd.DataFrame(df_static, columns=cols_staticRes) # training results


sss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=0) 

for train_index, test_index in sss.split(dfAttr_static, dfRes_static):
    X_train, X_test = dfAttr_static.iloc[train_index], dfAttr_static.iloc[test_index]
    y_train, y_test = dfRes_static.iloc[train_index], dfRes_static.iloc[test_index]                  

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


rf_static = RandomForestClassifier(n_estimators=100) # initialize
#rf_static = AdaBoostClassifier(n_estimators=100)

rf_static = rf_static.fit(X_train, np.ravel(y_train))



predictions_rf_static = rf_static.predict(X_test)
print (predictions_rf_static)

score_rf_static = accuracy_score(y_test, predictions_rf_static)
print (score_rf_static)

matrix = confusion_matrix(y_test, predictions_rf_static)
print(matrix)


TN = matrix[0][0]
FN = matrix[1][0]
TP = matrix[1][1]
FP = matrix[0][1]

#Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ("true positive rate: ", TPR)
print ("Specificity or true negative rate: ", TNR)
print ("Precision or positive predictive value: ", PPV)
print ("Negative predictive value: ", NPV)
print ("Fall out or false positive rate: ", FPR)
print ("False negative rate: ", FNR)
print ("False discovery rate: ", FDR)
print ("Accuracy: ", ACC)


prob_rf_static = rf_static.predict_proba(X_test)

#print(prob_knn)

plt.plot(prob_rf_static[:,1])

plt.show()


score_rf_cv = cross_val_score(rf_static, dfAttr_static, np.ravel(dfRes_static), cv=20)

print(score_rf_cv)