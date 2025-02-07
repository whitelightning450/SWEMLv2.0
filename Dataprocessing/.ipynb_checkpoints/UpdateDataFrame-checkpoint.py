import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook

#HOME = os.getcwd()
HOME = os.chdir('..')
HOME = os.getcwd()
#HOME = os.path.expanduser('~')


def updateTrainingDF(region, output_res, training_cat, fSCA, updatefile):
#    dfpath = f"{HOME}/SWEMLv2.0/data/TrainingDFs/{region}/{output_res}M_Resolution"
    dfpath = f"{HOME}/data/TrainingDFs/{region}/{output_res}M_Resolution"

    #load files
    updatefile = pd.read_parquet(f"{dfpath}/{region}_metadata.parquet")
    trainpath = f"{dfpath}/{training_cat}/{fSCA}"

    #get list of files in trainingpath
    trainfiles = [f for f in listdir(trainpath) if isfile(join(trainpath, f))]

    for file in tqdm_notebook(trainfiles):
        #read file
        trainfile = pd.read_parquet(f"{trainpath}/{file}")

        #set index
        if 'cell_id' in trainfile.columns:
            trainfile.set_index('cell_id', inplace=True)

        #update trainfile with correct elevation, slope, aspect
        trainfile.update(updatefile)

        #save file
        table = pa.Table.from_pandas(trainfile)

        # Parquet with Brotli compression
        print(f"{trainpath}/{trainfile}")

        pq.write_table(table, f"{trainpath}/{file}", compression='BROTLI')