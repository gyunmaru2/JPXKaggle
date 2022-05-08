# %%
import os
import numpy as np
import pandas as pd
import datetime 

class storage_file(object) :

    train_files = "/kaggle/input/"\
        "jpx-tokyo-stock-exchange-prediction/train_files/"
    supplemental_files = "/kaggle/input/"\
        "jpx-tokyo-stock-exchange-prediction/supplemental_files/"

    storage_dir = "/kaggel/working/JPXKaggle/"
    

    def __init__(self) :

        if not os.path.exists(self.storage_dir) :
            os.makedirs(self.storage_dir)

    def save_df(self,df,name_base) :

        hiduke = datetime.datetime.today().strftime("%Y%m%d")
        df.to_csv(name_base+f"_{hiduke}.csv",index=False)

    def add_online_data(self,newdf,filename) :

        if filename[-4:] != '.csv' :
            filename += ".csv"

        if not os.path.exists(self.storage_dir+filename) :
            self.create_base_file(filename)

        newdf.to_csv(self.storage_dir+filename,mode="a",
                header = False, index = False 
            )

    def create_base_file(self,filename) :

        if filename in ("stock_fin.csv") :
            querystr = "Date >= '2019-12-31'"
        else :
            querystr = "Date >= '2021-06-30"

        tmp1 = pd.read_csv(
            self.train_files+filename, dtype=str
        )
        tmp1 = tmp1.query(querystr)

        tmp2 = pd.read_csv(
            self.supplemental_files+filename, dtype=str
        )

        tmp1 = pd.concat([tmp1,tmp2])\
                .sort_values(["Date","SecuritiesCode"])\
                .reset_index(drop=True)

        tmp1.to_csv(
            self.storage_dir+filename,index=False
        )





    