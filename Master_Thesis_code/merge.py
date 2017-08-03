import pandas as pd

#cols_dynamic = ['SLS_SCHMELZEID','Datum','Zeit','Blaszeit_s','O2Menge_Nm','Sound_unten_f1','Sound_unten_f2','Sound_unten_f3','Sound_oben_f1','Sound_oben_f2','Sound_oben_f3','O2Rate_[Nm_min]','Abgasrate_Nm_h','Lanzenheight_m','Stellring_','CO_Konzentration_','Unterdruck_Stellring_kPa','helle_Pixel_N','Tor_SW','Tor_SO','Tor_N','Digital_4','Digital_5','Digital_6','Schallpegel','Schlackenauswurf']
cols_static = ['SID', 'BEZ_SCHMELZE_ORG', 'SLS_SCHMELZEID', 'DT_BEGINN_IST', 'BEGINN_HBL', 'BEZ_KNV_IST', 'ALTER_KONVERTER', 'LANZENALTER', 'ID_PROBE', 'C', 'SI', 'TI', 'V', 'EINGELEERTES_RE', 'EINLEERGEWICHT_SC_GESAMT', 'AG', 'AW', 'NS', 'FB', 'SN', 'RS', 'FS', 'RE', 'MV', 'MUENDUNGSBAER', 'KALKZUGABE', 'GEWICHTE', 'KALK_ZUGABE_ZEITPUNKTE', 'MAX_INTENSITÄT_AUSWURF', 'ROUND(AUSWURF.DAUER)']

cols = ['SLS_SCHMELZEID', 'MAX_INTENSITÄT_AUSWURF']

cols_slopping = ['SLS_SCHMELZEID', 'Slopping']


Tab = pd.read_csv("Slopping_list.csv", header=1, names=cols_slopping)
Tab_1 = pd.read_excel("Kopie_von_OAS_Datenauswertung_20170713.xlsx", sheetname="Tabelle1", header=1, names=cols_static)

dfStatic = pd.DataFrame(Tab_1, columns=cols)

result = pd.merge(dfStatic, Tab, on = ['SLS_SCHMELZEID'], right_index=False, how='left', sort=False, copy=False);
                 
#result_sort = result.sort(['Blaszeit_s'])

#result_500 = result_sort[0:500]
                 
                 
writer = pd.ExcelWriter('Compare.xlsx', engine='xlsxwriter')

result.to_excel(writer, sheet_name='Sheet1')

workbook  = writer.book
workbook.filename = 'Compare.xlsx'
#workbook.add_vba_project('./vbaProject.bin')

writer.save()