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

columns = ['SLS_SCHMELZEID','Datum','Zeit','Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_[Nm_min]','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N','Tor_SW','Tor_SO','Tor_N','Digital_4','Digital_5','Digital_6','Schallpegel','Schlackenauswurf','SID','BEZ_SCHMELZE_ORG','DT_BEGINN_IST','BEGINN_HBL','BEZ_KNV_IST','ALTER_KONVERTER','LANZENALTER','ID_PROBE','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','MUENDUNGSBAER','KALKZUGABE','GEWICHTE','KALK_ZUGABE_ZEITPUNKTE','MAX_INTENSITÄT_AUSWURF','ROUND(AUSWURF.DAUER)']
cols_Attr = ['Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_[Nm_min]','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N','Tor_SW','Tor_SO','Tor_N','Digital_5','Digital_6','Schallpegel','Schlackenauswurf','ALTER_KONVERTER','LANZENALTER','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','KALKZUGABE']
cols_Res = ['MAX_INTENSITÄT_AUSWURF']

h2o.upload_file(path="Full_Data_csv.csv")

datadata = h2o.import_file(path="Full_Data_csv.csv", header=1, col_names=columns)

datadata.names

list = ['SLS_SCHMELZEID','Datum','Zeit','Digital_4','SID','BEZ_SCHMELZE_ORG','DT_BEGINN_IST','BEGINN_HBL','BEZ_KNV_IST','ID_PROBE','GEWICHTE','KALK_ZUGABE_ZEITPUNKTE','ROUND(AUSWURF.DAUER)']
data_1 = datadata.drop(list)

mask = data_1["Blaszeit_s"] < 250
data = data_1[mask,:]


data["MAX_INTENSITÄT_AUSWURF"] = data["MAX_INTENSITÄT_AUSWURF"].asfactor()

train, valid, test = data.split_frame([0.6, 0.2], seed=1234)


      
model = H2ODeepLearningEstimator(activation="RectifierWithDropout", hidden=[50,50,50], l1=1e-5, epochs=500)

model.train( x=data.col_names[:-1], y=data.col_names[-1], training_frame=train, validation_frame=valid)

predictions = model.predict(test[:-1])

accuracy_dl = (predictions['predict']==test['MAX_INTENSITÄT_AUSWURF']).as_data_frame(use_pandas=True).mean()

print (accuracy_dl)

model_cv = H2ODeepLearningEstimator(activation="RectifierWithDropout", hidden=[50,50,50], l1=1e-5, epochs=500, nfolds=5)

model_cv.train( x=data.col_names[:-1], y=data.col_names[-1], training_frame=data)

#model_path = h2o.save_model(model=model, path="deep_learning_model", force=True)

#model_cv_path = h2o.save_model(model=model_cv, path="deep_learning_model_cv", force=True)

predict = predictions['predict'].as_data_frame(use_pandas=True)
plt.plot(predict[0:10000])

plt.xticks(predict['predict'], predict.index.values ) # location, labels
plt.plot(predict['predict'] )
plt.show()