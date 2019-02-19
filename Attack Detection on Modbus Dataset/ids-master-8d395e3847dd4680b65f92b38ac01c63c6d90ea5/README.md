SA Project 2019 Group 2
These are the project results of Group 2.

## This Folder
The directory structure is as follows:
```
./data/
    |->.csv/
    |   |-> extracted csv files
    |
    |->.tshark_csv/
            |-> files extracted by tshark
    |
    |->labels/
    |   |-> label files from the public modbus dataset
    |
    |-> complete.ipynb
    |
    |-> backup_complete.py
    |
    |-> spark_dataframe/
    |    |-> this is a preprocessed version of the dataset. If wireshark is not available, this can be used.
    |     
    |-> testData.pq/
         |-> this is stored testdata to only test the classifier
    |-> logRegModel
``` 

Some of these directories might not be present before the project ran for the first time, they are created during the execution.

## Running the Classifier
The final result we consolidated in the complete.ipynb notebook consist of 4 major parts:
 1. data preprocessing
 2. loading the data into spark
 3. training the classifier
 4. testing the classifier
 
 Part 1 depends on a bash environment, any unix machine should work, however we could only test it on linux.
 This is the case because we used the experimental "info column" feature, which can not be generated without the wireshark command line utility "tshark". The preprocessing part can be skipped if a spark dataframe we already generated is used, the dataframe is included in this folder. This makes it possible to test the classifier on hosts without a bash environment or wireshark.
 
### Dependencies for Running the complete notebook
To run the complete project, please install wireshark. On Ubuntu, this can be done with `sudo apt install wireshark`, for other distributions or Mac OS other package managers are available.

To check if the installation of tshark was successfull, run the following command: `tshark --v`. The output should look similar to this:

```
viktor@yp /m/v/d/U/i/i/final> tshark --v
TShark (Wireshark) 2.6.5 (Git v2.6.5 packaged as 2.6.5-1~ubuntu16.04.0)

Copyright 1998-2018 Gerald Combs <gerald@wireshark.org> and contributors.
License GPLv2+: GNU GPL version 2 or later <http://www.gnu.org/licenses/old-licenses/gpl-2.0.html>
This is free software; see the source for copying conditions. There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Compiled (64-bit) with libpcap, with POSIX capabilities (Linux), with libnl 3,
with GLib 2.48.2, with zlib 1.2.8, with SMI 0.4.8, with c-ares 1.10.0, with Lua
5.2.4, with GnuTLS 3.4.10, with Gcrypt 1.6.5, with MIT Kerberos, with MaxMind DB
resolver, with nghttp2 1.7.1, with LZ4, with Snappy, with libxml2 2.9.3.

Running on Linux 4.4.0-141-generic, with Intel(R) Core(TM) i7-4500U CPU @
1.80GHz (with SSE4.2), with 7895 MB of physical memory, with locale
LC_CTYPE=en_US.UTF-8, LC_NUMERIC=de_DE.UTF-8, LC_TIME=de_DE.UTF-8,
LC_COLLATE=en_US.UTF-8, LC_MONETARY=de_DE.UTF-8, LC_MESSAGES=en_US.UTF-8,
LC_PAPER=de_DE.UTF-8, LC_NAME=de_DE.UTF-8, LC_ADDRESS=de_DE.UTF-8,
LC_TELEPHONE=de_DE.UTF-8, LC_MEASUREMENT=de_DE.UTF-8,
LC_IDENTIFICATION=de_DE.UTF-8, with libpcap version 1.7.4, with GnuTLS 3.4.10,
with Gcrypt 1.6.5, with zlib 1.2.8, binary plugins supported (13 loaded).

Built using gcc 5.4.0 20160609.
```
### Dependencies for training and testing the classifier
For all other parts of the project some python packages are required. They can be installed with the following command on most systems: `pip3 install numpy pandas pyspark findspark jupyter`

Finally spark has to be installed.
For that please follow the instructions on this website: https://spark.apache.org/docs/latest/

In the installation process, the archive is unpacked to some folder on the filesystem where from then spark is located. The path to this top level directory is needed for our jupyter notebook to connect to spark and to access the spark api from python. If spark is not installed to `/usr/local/spark`, then this path in the first cell of the  notebook has to be adjusted.

In case that the notebook can't connect to spark have included backup scripts called `backup_complete.py` and `backup_complete.py` which can be run by the `./bin/spark-submit` binary in the spark folder. The complete command would then be `/usr/local/spark/bin/spark-submit complete_backup.py`.


### The complete Jupyter Notebook
Our programm is consolidated in a jupyter notebook. To open this, please enter `jupyter notebook  complete.ipynb` into a terminal while beeing in our project folder.

To run all the parts of the programm, select Cell -> Run All.
Cells can be run individually by placing the cursor in them and pressing CTRL + Enter.

### Only running the Classifier
The classification process is a part of the complete notebook. It loads the Testdata from the 'testData.pq' file in the project folder which is saved there by the Complete notebook.
It also loads a stored Logistic Regression Model saved there by the Complete Notebook and then predicts attacks on that test data and prints the performance results.
To only start the Classification Process, please scroll down in the Notebook to the big 'Classification' headline and run the remaining cell from there.
