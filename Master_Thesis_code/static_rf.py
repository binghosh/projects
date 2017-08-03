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




rf_static = RandomForestClassifier(n_estimators=100) # initialize
#rf_static = AdaBoostClassifier(n_estimators=100)

rf_static = rf_static.fit(dfAttr_static, np.ravel(dfRes_static))

filename = 'static_rf_model.sav'   # filename for model
joblib.dump(rf_static, filename)  #save model as file

          
loaded_model = joblib.load(filename)    #call model from file 



test_static = pd.DataFrame({'ALTER_KONVERTER':[315],
                     'LANZENALTER':[204],
                     'C':[4.62],
                     'SI':[0.557],
                     'TI':[0.054],
                     'V':[0.1],
                     'EINGELEERTES_RE':[175],
                     'EINLEERGEWICHT_SC_GESAMT':[46.25],
                     'AG':[0],
                     'AW':[25310],
                     'NS':[0],
                     'FB':[0],
                     'SN':[13410],
                     'RS':[0],
                     'FS':[0],
                     'RE':[7525],
                     'MV':[0],
                     'KALKZUGABE':[12.905]})
                    # 'MAX_INTENSITÄT_AUSWURF':[0]})





predictions_rf_static = rf_static.predict(test_static)
print (predictions_rf_static)