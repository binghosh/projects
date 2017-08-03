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

filename = 'static_rf_model.sav'   # filename for model

rf_static = joblib.load(filename)    #call model from file 



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