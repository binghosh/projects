import pandas as pd

import CHAID



columns = ['SLS_SCHMELZEID','Datum','Zeit','Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_[Nm_min]','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N','Tor_SW','Tor_SO','Tor_N','Digital_4','Digital_5','Digital_6','Schallpegel','Schlackenauswurf','SID','BEZ_SCHMELZE_ORG','DT_BEGINN_IST','BEGINN_HBL','BEZ_KNV_IST','ALTER_KONVERTER','LANZENALTER','ID_PROBE','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','MUENDUNGSBAER','KALKZUGABE','GEWICHTE','KALK_ZUGABE_ZEITPUNKTE','MAX_INTENSITÄT_AUSWURF','ROUND(AUSWURF.DAUER)']
cols_Attr = ['Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_[Nm_min]','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N','Tor_SW','Tor_SO','Tor_N','Digital_5','Digital_6','Schallpegel','Schlackenauswurf','LANZENALTER','C','SI','TI','V','EINGELEERTES_RE','EINLEERGEWICHT_SC_GESAMT','AG','AW','NS','FB','SN','RS','FS','RE','MV','KALKZUGABE']
cols_Res = ['MAX_INTENSITÄT_AUSWURF']

df = pd.read_csv("Full_Data_csv.csv", header=1, names=columns)

tree = CHAID.from_pandas_df(df, cols_Attr, cols_Res)


