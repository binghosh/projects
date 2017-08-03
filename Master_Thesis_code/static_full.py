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


cols_static = ['SID', 'BEZ_SCHMELZE_ORG', 'SLS_SCHMELZEID', 'DT_BEGINN_IST', 'BEGINN_HBL', 'BEZ_KNV_IST', 'ALTER_KONVERTER', 'LANZENALTER', 'ID_PROBE', 'C', 'SI', 'TI', 'V', 'EINGELEERTES_RE', 'EINLEERGEWICHT_SC_GESAMT', 'AG', 'AW', 'NS', 'FB', 'SN', 'RS', 'FS', 'RE', 'MV', 'MUENDUNGSBAER', 'KALKZUGABE', 'GEWICHTE', 'KALK_ZUGABE_ZEITPUNKTE', 'MAX_INTENSITÄT_AUSWURF', 'ROUND(AUSWURF.DAUER)']
cols_staticAttr = ['ALTER_KONVERTER','LANZENALTER', 'C', 'SI', 'TI', 'V', 'EINGELEERTES_RE', 'EINLEERGEWICHT_SC_GESAMT', 'AG', 'AW', 'NS', 'FB', 'SN', 'RS', 'FS', 'RE', 'MV', 'KALKZUGABE']
#cols_staticAttr = ['ALTER_KONVERTER', 'LANZENALTER', 'C', 'SI', 'TI', 'V', 'EINGELEERTES_RE', 'EINLEERGEWICHT_SC_GESAMT', 'MV']
cols_staticRes = ['MAX_INTENSITÄT_AUSWURF']
#cols_staticRes = ['ROUND(AUSWURF.DAUER)']



df_static = pd.read_excel("Kopie_von_OAS_Datenauswertung_20170713.xlsx", sheetname="Tabelle1", header=1, names=cols_static)
#df1_static = pd.read_csv("static_data_500.csv",  header=1, names=cols_static)
#df2 = pd.read_csv("ConvC_data_binary.csv", nrows=269, header=1, names=cols_staticRes)
df_static.head()


#np.nan_to_num(df1_static)
#print (np.where(np.isnan(df1_static)))

dfArr_static = pd.DataFrame(df_static, columns=cols_staticAttr) #training array
dfRes_static = pd.DataFrame(df_static, columns=cols_staticRes) # training results

X_train1_static, X_test1_static, y_train1_static, y_test1_static = train_test_split(dfArr_static, dfRes_static, test_size=0.2)

print (X_train1_static.shape, y_train1_static.shape)
print (X_test1_static.shape, y_test1_static.shape)

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=0) 

for train_index, test_index in sss.split(dfAttr, dfRes):
    X_train, X_test = dfAttr_static.iloc[train_index], dfAttr_static.iloc[test_index]
    y_train, y_test = dfRes_static.iloc[train_index], dfRes_static.iloc[test_index]                  

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


#feat_extr = SelectKBest(k=7)
#fitter = feat_extr.fit(dfArr_static, ravel(dfRes_static))

#scores1 = fitter.scores_

#scores = pd.DataFrame(fitter.scores_, index=cols_staticAttr)

#model = ExtraTreesClassifier()
#model = model.fit(dfArr_static, ravel(dfRes_static))

#model_scores = pd.DataFrame(model.feature_importances_, index=cols_staticAttr)


#rlasso = RandomizedLasso()
#lasso = rlasso.fit(dfArr_static, ravel(dfRes_static))

#lasso_scores = pd.DataFrame(lasso.scores_, index=cols_staticAttr)


#ard = linear_model.ARDRegression(compute_score=True)
#autorelevdet = ard.fit(dfArr_static, ravel(dfRes_static))

#ard_coef = pd.DataFrame(autorelevdet.coef_, index=cols_staticAttr)




clf = svm.SVC(decision_function_shape='ovo')

clf_static = clf.fit(X_train1_static, ravel(y_train1_static))



predictions_clf_static = clf_static.predict(X_test1_static)
print (predictions_clf_static)

score_clf_static = accuracy_score(y_test1_static, predictions_clf_static)
print (score_clf_static)

matrix = confusion_matrix(y_test1_static, predictions_clf_static)
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



#prob_static_clf = clf_static.predict_proba(X_test1_static)

#print (clf_static.predict_proba(X_test1_static))