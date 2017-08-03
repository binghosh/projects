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
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn import datasets

cols_dynamic = ['Filename','Datum','Zeit','Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_Nm_min','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N','Tor_SW','Tor_SO','Tor_N','Digital_4','Digital_5','Digital_6','Schallpegel','Schlackenauswurf','Slopping','Time']
cols_dynamicAttr = ['Zeit','Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_Nm_min','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N','Tor_SW','Tor_SO','Tor_N','Digital_4','Digital_5','Digital_6','Schallpegel','Schlackenauswurf']
cols_dynamicRes = ['Slopping']



df1_dynamic = pd.read_csv("Dynamic_data_500.csv",  header=1, names=cols_dynamic)
#df2 = pd.read_csv("ConvC_data_binary.csv", nrows=269, header=1, names=cols_dynamicRes)
df1_dynamic.head()


#np.nan_to_num(df1_dynamic)
#print (np.where(np.isnan(df1_dynamic)))

dfArr1_dynamic = pd.DataFrame(df1_dynamic, columns=cols_dynamicAttr) #training array
dfRes1_dynamic = pd.DataFrame(df1_dynamic, columns=cols_dynamicRes) # training results

pca = decomposition.PCA(n_components=10)
pca.fit(dfArr1_dynamic)
dfArr1_dynamic = pca.transform(dfArr1_dynamic)

X_train1_dynamic, X_test1_dynamic, y_train1_dynamic, y_test1_dynamic = train_test_split(dfArr1_dynamic, dfRes1_dynamic, test_size=0.2)

print (X_train1_dynamic.shape, y_train1_dynamic.shape)
print (X_test1_dynamic.shape, y_test1_dynamic.shape)




print (X_train1_dynamic)

#y_trained = y_train.ravel()

#knn_dynamic = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1) # initialize
knn_dynamic = KNeighborsClassifier()
knn_dynamic.fit(X_train1_dynamic, ravel(y_train1_dynamic)) # fit the data to the algorithm

filename = 'dynamic_knn_model.sav'   # filename for model
joblib.dump(knn_dynamic, filename)  #save model as file

          
#loaded_model = joblib.load(filename)    #call model from file    

predictions_knn_dynamic = knn_dynamic.predict(X_test1_dynamic)
print (predictions_knn_dynamic)
#print ("Score:", model.score(X_test, y_test))
score_knn_dynamic = accuracy_score(y_test1_dynamic, predictions_knn_dynamic)
print ("Accuracy:", score_knn_dynamic)

matrix = confusion_matrix(y_test1_dynamic, predictions_knn_dynamic)
print(matrix)

TN = matrix[0][0]
FN = matrix[1][0]
TP = matrix[1][1]
FP = matrix[0][1]

# Sensitivity, hit rate, recall, or true positive rate
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



prob_dynamic_knn = knn_dynamic.predict_proba(X_test1_dynamic)

print(knn_dynamic.predict_proba(X_test1_dynamic))

#score_knn_dynamic_cv = cross_val_score(knn_dynamic, X_train1_dynamic, ravel(y_train1_dynamic), cv=50)

#print ("Accuracy_knn_dynamic_cv: %0.2f (+/- %0.2f)" % (score_knn_dynamic_cv.mean(), score_knn_dynamic_cv.std() * 2))

#accuracy_knn_dynamic_cv = score_knn_dynamic_cv.mean()

plt.plot(prob_dynamic_knn[:,1])

plt.show()