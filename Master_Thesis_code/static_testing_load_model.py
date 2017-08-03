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
from sklearn.naive_bayes import GaussianNB
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

h2o.init()










################################################Test####################################################################################################################################################################################################                             

model_path = 'C:\\Users\\GHBI\\Documents\\Thesis\\static_dl_model\\DeepLearning_model_python_1501759947364_91'

model = h2o.load_model(model_path)

test_cols = ['ALTER_KONVERTER','LANZENALTER','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','KALKZUGABE','MAX_INTENSITÄT_AUSWURF']

test_pd1 = pd.DataFrame({'ALTER_KONVERTER':[315],
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
                     #'MAX_INTENSITÄT_AUSWURF':[0]})

test_pd = test_pd1[['ALTER_KONVERTER','LANZENALTER','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','KALKZUGABE']]

test_data = h2o.H2OFrame(test_pd)

test_data.set_names(['ALTER_KONVERTER','LANZENALTER','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','KALKZUGABE'])






#list = ['SLS_SCHMELZEID','SID','BEZ_SCHMELZE_ORG','DT_BEGINN_IST','BEGINN_HBL','BEZ_KNV_IST','ID_PROBE','MUENDUNGSBAER','GEWICHTE','KALK_ZUGABE_ZEITPUNKTE','ROUND(AUSWURF.DAUER)']
#test_data = test_static.drop(list)



#test_data["MAX_INTENSITÄT_AUSWURF"] = test_data["MAX_INTENSITÄT_AUSWURF"].asfactor()






      



predictions = model.predict(test_data[[1]])

predict = predictions.as_data_frame(use_pandas=True)

print (predict)




