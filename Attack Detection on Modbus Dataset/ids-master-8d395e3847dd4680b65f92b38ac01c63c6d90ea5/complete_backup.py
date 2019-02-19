#!/usr/bin/env python
# coding: utf-8

# ## **This is the jupyter notebook running our implementation of the ISA project.**
# 
# It is divided in two parts, first the preprocessing then the training of the classifier, and finally the classification of testing data.
# 
# The first part, the pre processing depends on a unix like system with the command line utilities wireshark and tshark installed. Further dependencies are the python packets pandas, jupyter, pyspark and findspark
# 
# The setup is describe in more detail in the Readme file which is also included in the next cell:

# # ISA Project 2019 Group 2
# These are the project results of Group 2.
# 
# ## This Folder
# The directory structure is as follows:
# ```
# ./data/
#     |->.csv/
#     |   |-> extracted csv files
#     |
#     |->.tshark_csv/
#             |-> files extracted by tshark
#     |
#     |->labels/
#     |   |-> label files from the public modbus dataset
#     |
#     |-> complete.ipynb
#     |
#     |-> backup_complete.py
#     |
#     |-> spark_dataframe/
#     |    |-> this is a preprocessed version of the dataset. If wireshark is not available, this can be used.
#     |     
#     |-> testData.pq/
#          |-> this is stored testdata to only test the classifier
#     |-> logRegModel
# ``` 
# 
# Some of these directories might not be present before the project ran for the first time, they are created during the execution.
# 
# ## Running the Classifier
# The final result we consolidated in the complete.ipynb notebook consist of 4 major parts:
#  1. data preprocessing
#  2. loading the data into spark
#  3. training the classifier
#  4. testing the classifier
#  
#  Part 1 depends on a bash environment, any unix machine should work, however we could only test it on linux.
#  This is the case because we used the experimental "info column" feature, which can not be generated without the wireshark command line utility "tshark". The preprocessing part can be skipped if a spark dataframe we already generated is used, the dataframe is included in this folder. This makes it possible to test the classifier on hosts without a bash environment or wireshark.
#  
# ### Dependencies for Running the complete notebook
# To run the complete project, please install wireshark. On Ubuntu, this can be done with `sudo apt install wireshark`, for other distributions or Mac OS other package managers are available.
# 
# To check if the installation of tshark was successfull, run the following command: `tshark --v`. The output should look similar to this:
# 
# ```
# viktor@yp /m/v/d/U/i/i/final> tshark --v
# TShark (Wireshark) 2.6.5 (Git v2.6.5 packaged as 2.6.5-1~ubuntu16.04.0)
# 
# Copyright 1998-2018 Gerald Combs <gerald@wireshark.org> and contributors.
# License GPLv2+: GNU GPL version 2 or later <http://www.gnu.org/licenses/old-licenses/gpl-2.0.html>
# This is free software; see the source for copying conditions. There is NO
# warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# 
# Compiled (64-bit) with libpcap, with POSIX capabilities (Linux), with libnl 3,
# with GLib 2.48.2, with zlib 1.2.8, with SMI 0.4.8, with c-ares 1.10.0, with Lua
# 5.2.4, with GnuTLS 3.4.10, with Gcrypt 1.6.5, with MIT Kerberos, with MaxMind DB
# resolver, with nghttp2 1.7.1, with LZ4, with Snappy, with libxml2 2.9.3.
# 
# Running on Linux 4.4.0-141-generic, with Intel(R) Core(TM) i7-4500U CPU @
# 1.80GHz (with SSE4.2), with 7895 MB of physical memory, with locale
# LC_CTYPE=en_US.UTF-8, LC_NUMERIC=de_DE.UTF-8, LC_TIME=de_DE.UTF-8,
# LC_COLLATE=en_US.UTF-8, LC_MONETARY=de_DE.UTF-8, LC_MESSAGES=en_US.UTF-8,
# LC_PAPER=de_DE.UTF-8, LC_NAME=de_DE.UTF-8, LC_ADDRESS=de_DE.UTF-8,
# LC_TELEPHONE=de_DE.UTF-8, LC_MEASUREMENT=de_DE.UTF-8,
# LC_IDENTIFICATION=de_DE.UTF-8, with libpcap version 1.7.4, with GnuTLS 3.4.10,
# with Gcrypt 1.6.5, with zlib 1.2.8, binary plugins supported (13 loaded).
# 
# Built using gcc 5.4.0 20160609.
# ```
# ### Dependencies for training and testing the classifier
# For all other parts of the project some python packages are required. They can be installed with the following command on most systems: `pip3 install numpy pandas pyspark findspark jupyter`
# 
# Finally spark has to be installed.
# For that please follow the instructions on this website: https://spark.apache.org/docs/latest/
# 
# In the installation process, the archived is unpacked to some folder on the filesystem where from then spark is located. The path to this top level directory is needed for our jupyter notebook to connect to spark and to access the spark api from python. If spark is not installed to `/usr/local/spark`, then this path in the first cell of the  notebook has to be adjusted.
# 
# In case that the notebook can't connect to spark have included backup scripts called `backup_complete.py` and `backup_complete.py` which can be run by the `./bin/spark-submit` binary in the spark folder.
# 
# 
# ### The complete Jupyter Notebook
# Our programm is consolidated in a jupyter notebook. To open this, please enter `jupyter notebook  complete.ipynb` into a terminal while beeing in our project folder.
# 
# To run all the parts of the programm, select Cell -> Run All.
# Cells can be run individually by placing the cursor in them and pressing CTRL + Enter.
# 
# ### Only running the Classifier
# The classification process is a part of the complete notebook. It loads the Testdata from the 'testData.pq' file in the project folder which is saved there by the Complete notebook.
# It also loads a stored Logistic Regression Model saved there by the Complete Notebook and then predicts attacks on that test data and prints the performance results.
# To only start the Classification Process, please scroll down in the Notebook to the big 'Classification' headline and run the remaining cell from there.

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import pandas as pd
pd.options.display.max_columns = None

import findspark
findspark.init("/usr/local/spark/")
from pyspark import SparkContext
from pyspark import SQLContext

from pyspark.context import SparkContext
sc = SparkContext.getOrCreate()


spark = SQLContext(sc)


# In[4]:


import os
#extracting the pcap files to csv
os.system("./wireshark_extraction.sh")
os.system("./tshark_extraction.sh")


# In[62]:


#loading wireshark extracted CSVs
csv_path = './data/csv'
label_path = './data/labels'
tshark_path = './data/tshark_csv'
print("Looking up files in path", os.path.abspath(label_path))
label_files = os.listdir(label_path)
csv_files = os.listdir(csv_path)

l = []
for file in sorted(label_files):
    print("found:",file)
    if 'labeled' in file:
        csv_file = file.replace('_labeled', '.pcap_extracted')
        print("found:",csv_file)
        tshark_file = file.replace('_labeled', '.pcap_tshark')
        print("found:",tshark_file)
        t = os.path.join(label_path,file),os.path.join(csv_path,csv_file),os.path.join(tshark_path,tshark_file)
        l.append(t)


# In[ ]:


#these are helper functions for the data preprocessing
def merge(traffic,tshark,labeled):
    attacks = labeled.iloc[:,1]
    merged = pd.concat([extracted,tshark,attacks], axis=1)
    return merged

def add(all_extracted,merged):
    added = pd.concat([all_extracted,merged],axis=0)
    return added

#this function generates the sport and dport features.
#they described source and destination port regardlerss whether udp or tcp was used.
#the tcp and udp features are set if the corresponding transport protocol was used
def extract_ports(df):
    tshark = df
    import math

    for i in tshark.index:
        if  ((not math.isnan(tshark.iloc[i,:]["tcp.srcport"])) or (not math.isnan(tshark.iloc[i,:]["tcp.dstport"]))):
            #print("tcp")
            tshark["sport"] = int(tshark.iloc[i,:]["tcp.srcport"])
            tshark["dport"] = int(tshark.iloc[i,:]["tcp.dstport"])
            tshark["tcp"] = 1
            tshark["udp"] = 0

            pass
        elif ((not math.isnan(tshark.iloc[i,:]["udp.srcport"])) or (not math.isnan(tshark.iloc[i,:]["udp.dstport"]))):
            tshark["sport"] = int(tshark.iloc[i,:]["udp.srcport"])
            tshark["dport"] = int(tshark.iloc[i,:]["udp.dstport"])
            tshark["udp"] = 1
            tshark["tcp"] = 0
        
        else:
            tshark["sport"] = 0
            tshark["dport"] = 0
            tshark["udp"] = 0
            tshark["tcp"] = 0


# In[95]:


#now we load all csv files, to the port processing on the tshark file and merge them.
#the merged files are appended together to one big dataframe containing all the features.


#this takes a while, depending on the harddrive speed and CPU of the computer where it is executed.

all_extracted = pd.DataFrame()

for p in l:
    print("reading" + p[0])
    extracted = pd.read_csv(p[1], sep=';;',engine="python")
    
    tshark_extracted = pd.read_csv(p[2], sep=',')
    extract_ports(tshark_extracted)
    tshark_extracted = tshark_extracted[["sport","dport","udp","tcp","eth.src","eth.dst"]]
    print(tshark_extracted.shape)
    
    labeled = pd.read_csv(p[0], sep=';',names=["nr","attack"])
    print(extracted.shape)
    merged = merge(extracted,tshark_extracted,labeled)
    
    print(labeled.shape)
    
    
    if all_extracted.empty:
        all_extracted = merged
    else:
         all_extracted = add(all_extracted,merged)
    print("new size of complete dataframe:",all_extracted.shape)
all_extracted


# In[96]:


all_extracted.to_csv("./data/preprocessed.csv")


# In[97]:


data = pd.read_csv("./data/preprocessed.csv")


# In[98]:


all_extracted


# In[ ]:





# In[101]:


#rename the wrongly labeled attack column (pandas par)
df = all_extracted
df.rename({"nr":"attack"}, axis="columns", inplace=True)
df.rename({"eth.src":"eth_src"}, axis="columns", inplace=True)
df.rename({"eth.dst":"eth_dst"}, axis="columns", inplace=True)
#make shure we have only the columns we want
cols = ['Source', 'Destination', 'Protocol', 'Length', 'Info',  "sport","dport", "tcp","udp","eth_src","eth_dst",'attack']
df = df[cols]

#print the dataframe in the current form
df


# In[106]:


#now we generate the spark dataframe and save it to disk. This can take a while
data = spark.createDataFrame(df)
#The below command will fail if the spark dataframe was already written earlier. 
#Then you can just ontinue with the next cell!
try:
    data.write.parquet("spark_dataframe.pq")
except:
    print("spark dataframe not writte, it probably already exists?")
    pass


# In[107]:


data = spark.read.parquet('spark_dataframe.pq')


# In[109]:


#import the necessary pakages for the data processing
from pyspark.ml.feature import RegexTokenizer,StringIndexer, OneHotEncoder, VectorAssembler,StopWordsRemover, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression


# In[110]:


############INFO COLUMN #########
# regular expression tokenizer
regexTokenizer = [RegexTokenizer(inputCol="Info", outputCol="words", pattern="\\W")]
# stop words
add_stopwords = ["->"] 
stopwordsRemover = [StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)]
# bag of words count
countVectors = [CountVectorizer(inputCol="filtered", outputCol="info_features", vocabSize=10000, minDF=5)]


# In[111]:


##########Categorical Columns################

categorical_columns = ["Source", "Destination", "Protocol", "eth_dst", "eth_src","sport","dport",]
#categorical_columns.remove('Info')
print("Categorical columns:", categorical_columns)

####Build StringIndexe stages


stringindexer_stages = [StringIndexer(inputCol=c, outputCol='stringindexed_' + c) for c in categorical_columns]
# encode label column and add it to stringindexer stages
stringindexer_stages += [StringIndexer(inputCol='attack', outputCol='label')]

print("stringindexer_stages:",stringindexer_stages)


#####Build OneHotEncoder stages


onehotencoder_stages = [OneHotEncoder(inputCol='stringindexed_' + c, outputCol='onehot_'+c) for c in categorical_columns]

print("onehotencoder_stages", onehotencoder_stages)

#####Build VectorAssembler stage

feature_columns = ['onehot_' + c for c in categorical_columns]
feature_columns += ["info_features", "Length", "tcp", "udp"]
vectorassembler_stage = VectorAssembler(inputCols=feature_columns, outputCol='features')

print("vectorassembler_stage", vectorassembler_stage)


# In[112]:


#####Build pipeline model

all_stages = regexTokenizer + stopwordsRemover + countVectors + stringindexer_stages + onehotencoder_stages + [vectorassembler_stage]

print(all_stages)

pipeline = Pipeline(stages=all_stages)


#######Fit pipeline model

pipeline_model = pipeline.fit(data)


#######Transform data
print("Pipeline assembled, now starting transformation")
final_columns = ['features', 'label']
cuse_df = pipeline_model.transform(data).select(final_columns)
print("data preprocessing is finished")
cuse_df.show(5)


# In[113]:


###############Train & Test ###################################


from pyspark.sql.functions import lit


print("length of dataset:",cuse_df.count())
                                                           

(trainingData, testData) = cuse_df.randomSplit([0.7, 0.3], seed = 100)



print("length of training data:",trainingData.count())
print("length of test data:",testData.count())


from pyspark.sql.functions import desc

attacks = trainingData.filter(cuse_df.label == 1)
no_attacks = trainingData.filter(cuse_df.label == 0)
training_attacks = attacks.count()
print("how many attacks in training dataset? ",training_attacks)
#these are fixed lengths, only fitting to the dataset of length 902754!
(taken, not_taken) = no_attacks.randomSplit([(training_attacks/637997),1-(training_attacks/637997)], seed = 100)
print("how many of the non attack packets do we use for the traiing dataset?",str(taken.count()))


trainingData = attacks.union(taken)

testattacks = testData.filter("label == 1").count()
testratio = testattacks/testData.count()
print("length of test data set: ",str(testData.count()))
print("test ratio:",testratio)

trainattacks = trainingData.filter("label == 1").count()
print(trainattacks)
trainratio = trainattacks/trainingData.count()
print("length of training data set: ",trainingData.count())
print("train ratio:",trainratio)


# In[152]:


try:
    testData.write.parquet("testData.pq")
except:
    print("testData already saved!")
    pass


# # Classification starts here
# ## If you only want to try out the classification, run the remaining cells from here

# In[154]:


testData = spark.read.parquet('testData.pq')
    


# In[155]:


#train it!
lr = LogisticRegression(labelCol="label", maxIter=20)

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

paramGrid = ParamGridBuilder()     .addGrid(lr.regParam, [0.01, 0.3, 0.5])     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])    .build()
    
    
crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=3)  # use 3+ folds in practice

# Run cross-validation, and choose the best set of parameters.
lrModel = crossval.fit(trainingData)


# In[156]:


#this code block extracts the best threshold value for a maximum recall value
trainingSummary = lrModel.bestModel.summary
recall = trainingSummary.recallByThreshold
maxRecall = recall.groupBy().max('recall').select('max(recall)').head()
bestThreshold = recall.where(recall['recall'] == maxRecall['max(recall)'])     .select('threshold').head()['threshold']
print("setting the optimal threshold for maximum recall: ",bestThreshold)
_ = lr.setThreshold(bestThreshold)


# In[159]:


#this model is now saved so it can be quickly acessed in future runs
try:
    lrModel.save("LogRegModel") 
except:
    print("model was already saved")


# In[160]:


from pyspark.ml.tuning import CrossValidatorModel
CrossValidatorModel.read().load("LogRegModel")
#now the model ist loaded


# In[161]:


#Now we use our Model and predict the test data with it
pred = lrModel.transform(testData)


# In[165]:


TP = pred.filter("label == 1 and prediction == 1").count()
TN = pred.filter("label == 0 and prediction == 0").count()
FP = pred.filter("label == 0 and prediction == 1").count()
FN = pred.filter("label == 1 and prediction == 0").count()


print("we have some results!\n")
print("True Positive", TP)

print("True Negative", TN)

print("False Positive", FP)

print("False Negative", FN)

print("\nrecall",TP/(TP+FN))
print("precision:",(TP/(TP+FP)))


# In[163]:


import os
try:
    os.system('spd-say -i -10 "dear human, the classification is finished"')
except:
    pass

