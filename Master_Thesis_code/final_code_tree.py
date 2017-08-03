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
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import seaborn as sn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier


columns = ['SLS_SCHMELZEID','Datum','Zeit','Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_[Nm_min]','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N','Tor_SW','Tor_SO','Tor_N','Digital_4','Digital_5','Digital_6','Schallpegel','Schlackenauswurf','SID','BEZ_SCHMELZE_ORG','DT_BEGINN_IST','BEGINN_HBL','BEZ_KNV_IST','ALTER_KONVERTER','LANZENALTER','ID_PROBE','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','MUENDUNGSBAER','KALKZUGABE','GEWICHTE','KALK_ZUGABE_ZEITPUNKTE','MAX_INTENSITÄT_AUSWURF','ROUND(AUSWURF.DAUER)']
cols_Attr = ['Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_[Nm_min]','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N','Tor_SW','Tor_SO','Tor_N','Digital_5','Digital_6','Schallpegel','Schlackenauswurf','LANZENALTER','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','KALKZUGABE']
cols_Res = ['MAX_INTENSITÄT_AUSWURF']

df = pd.read_excel("Full_Data.xlsx", sheetname="Sheet1", header=1, names=columns)

dfAttr = pd.DataFrame(df, columns=cols_Attr) #training array

dfRes = pd.DataFrame(df, columns=cols_Res)  # training results

#X_train, X_test, y_train, y_test = train_test_split(dfAttr, dfRes, test_size=0.7 )    

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=0) 

for train_index, test_index in sss.split(dfAttr, dfRes):
    X_train, X_test = dfAttr.iloc[train_index], dfAttr.iloc[test_index]
    y_train, y_test = dfRes.iloc[train_index], dfRes.iloc[test_index]                

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

clf = tree.DecisionTreeClassifier() # initialize
clf = clf.fit(X_train, np.ravel(y_train)) # fit the data to the algorithm

               
#filename = 'final_model_clf.sav'   # filename for model
#joblib.dump(clf, filename)  #save model as file

predictions_clf = clf.predict(X_test)


score_clf = accuracy_score(y_test, predictions_clf)
print ("Score = ", score_clf)

mse = mean_squared_error(y_test, predictions_clf)

print("MSE = ", mse)


matrix = confusion_matrix(y_test, predictions_clf)

print(matrix)

plt.figure()
sn.heatmap(matrix, annot=True)
plt.show()

#prob_clf = clf.predict_proba(X_test)

#print(prob_clf)

#plt.plot(prob_clf[:,1])

#plt.show()


score_tree_cv = cross_val_score(clf, dfAttr, np.ravel(dfRes), cv=20)

print("Score_cv = ", score_tree_cv)





adaboost = AdaBoostClassifier(n_estimators=100)

score_adaboost_cv = cross_val_score(adaboost, dfAttr, np.ravel(dfRes), cv=10)

print("Score_cv = ", score_adaboost_cv)