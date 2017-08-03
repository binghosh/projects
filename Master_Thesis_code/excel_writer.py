


from sklearn import preprocessing

import pandas as pd



cols_dynamic = ['Datum','Zeit','Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_Nm_min','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N','Tor_SW','Tor_SO','Tor_N','Digital_4','Digital_5','Digital_6','Schallpegel','Schlackenauswurf']

cols_X = ['Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_Nm_min','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N','Tor_SW','Tor_SO','Tor_N','Digital_4','Digital_5','Digital_6','Schallpegel','Schlackenauswurf']

Y = pd.read_csv("wH_2017-04-01_85950.txt", sep="	", header=1, names=cols_dynamic)

X = pd.DataFrame(Y, columns=cols_X)

Xnorm = pd.DataFrame(X, columns=cols_X)

#scaler = preprocessing.StandardScaler().fit(X)
#Xnorm = scaler.transform(X)

mean_Sound_unten_f1 = X.Sound_unten_f1.dropna().mean()
max_Sound_unten_f1 =  X.Sound_unten_f1.dropna().max()
min_Sound_unten_f1 =  X.Sound_unten_f1.dropna().min()

Xnorm['Sound_unten_f1'] = X['Sound_unten_f1'].apply(lambda x: (x - mean_Sound_unten_f1 ) / (max_Sound_unten_f1 -min_Sound_unten_f1 ))


mean_Sound_unten_f2 = X.Sound_unten_f2.dropna().mean()
max_Sound_unten_f2 =  X.Sound_unten_f2.dropna().max()
min_Sound_unten_f2 =  X.Sound_unten_f2.dropna().min()

Xnorm['Sound_unten_f2'] = X['Sound_unten_f2'].apply(lambda x: (x - mean_Sound_unten_f2 ) / (max_Sound_unten_f2 -min_Sound_unten_f2 ))


mean_Sound_unten_f3 = X.Sound_unten_f3.dropna().mean()
max_Sound_unten_f3 =  X.Sound_unten_f3.dropna().max()
min_Sound_unten_f3 =  X.Sound_unten_f3.dropna().min()

Xnorm['Sound_unten_f3'] = X['Sound_unten_f3'].apply(lambda x: (x - mean_Sound_unten_f3 ) / (max_Sound_unten_f3 -min_Sound_unten_f3 ))

mean_Sound_oben_f1 = X.Sound_oben_f1.dropna().mean()
max_Sound_oben_f1 =  X.Sound_oben_f1.dropna().max()
min_Sound_oben_f1 =  X.Sound_oben_f1.dropna().min()

Xnorm['Sound_oben_f1'] = X['Sound_oben_f1'].apply(lambda x: (x - mean_Sound_oben_f1 ) / (max_Sound_oben_f1 -min_Sound_oben_f1 ))


mean_Sound_oben_f2 = X.Sound_oben_f2.dropna().mean()
max_Sound_oben_f2 =  X.Sound_oben_f2.dropna().max()
min_Sound_oben_f2 =  X.Sound_oben_f2.dropna().min()

Xnorm['Sound_oben_f2'] = X['Sound_oben_f2'].apply(lambda x: (x - mean_Sound_oben_f2 ) / (max_Sound_oben_f2 -min_Sound_oben_f2 ))


mean_Sound_oben_f3 = X.Sound_oben_f3.dropna().mean()
max_Sound_oben_f3 =  X.Sound_oben_f3.dropna().max()
min_Sound_oben_f3 =  X.Sound_oben_f3.dropna().min()

Xnorm['Sound_oben_f3'] = X['Sound_oben_f3'].apply(lambda x: (x - mean_Sound_oben_f3 ) / (max_Sound_oben_f3 -min_Sound_oben_f3 ))


#Xnorm = pd.DataFrame(X, columns=cols_X)

writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')

Xnorm.to_excel(writer, sheet_name='Sheet1')

workbook  = writer.book
workbook.filename = 'test.xlsx'
#workbook.add_vba_project('./vbaProject.bin')

writer.save()


