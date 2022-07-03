# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# %%

class label_etl(object) :

    ID = ['RowId','Date','SecuritiesCode','Target']

    def __init__(self,data_dir) :

        self.data_dir = data_dir
        self.price = pd.read_csv(data_dir+'stock_prices.csv',
                    dtype=str
                )

        for c in ['Open','High','Low','Close','Volume','AdjustmentFactor','Target'] :
            self.price.loc[:,c] = pd.to_numeric(self.price.loc[:,c],
                errors = "coerce")
        

    def select_stock(self,code) :

        price = self.price
        
        #修正株価
        tmp = price.query(f"SecuritiesCode == '{code}'")\
            .sort_values('Date').reset_index(drop=True)
        tmp = tmp.assign(cadj=tmp.AdjustmentFactor[::-1].cumprod())
        for c in ['Open','High','Low','Close'] :
            tmp.loc[:,c] = tmp.loc[:,c] * tmp.cadj
        tmp.loc[:,"PrevClose"] = pd.concat([pd.Series(np.nan),tmp.Close.iloc[:-1]])\
            .reset_index(drop=True)

        return tmp

    def add_label(self,df):

        df.loc[:,'rtn_day'] = df.Close.pct_change()
        df.loc[:,'label'] = df.rtn_day.shift(-2)

        return df


    def run_etl(self,debug=False) :

        self.debug=debug
        cols = self.ID.copy()
        cols.append('label')

        price = self.price
        codes = price.SecuritiesCode.dropna().unique()
        dfs = []
        for i,code in enumerate(codes):

            df = self.select_stock(code)
            df = self.add_label(df)
            df = df.loc[:,cols]
            dfs.append(df)

            if self.debug & (i==2) :
                break

        return pd.concat(dfs)
 
        




