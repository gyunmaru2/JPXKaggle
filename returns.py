
import numpy as np
import pandas as pd
import os
import sys 
import json

class return_calcurator:

    def __init__(self):

        with open("./local_settings.json","r") as f :
            config_ = json.load(f)

        self.train_files = config_['train_files']
        self.storage = config_['storage']

    def read_price(self):

        self.price = pd.read_csv(self.train_files+"stock_prices.csv",
                    dtype = str
                )
        for c in ['Open','High','Low','Close','Volume','AdjustmentFactor'] :
            self.price.loc[:,c] = pd.to_numeric(self.price.loc[:,c],
                errors = "coerce")

    def return_on_adjusted_close(self,query=None) :

        if not (query is None):
            self.price = self.price.query(query).reset_index(drop=True)

        codes = self.price.SecuritiesCode.unique()

        # return on adjusted price
        aps = []
        price = self.price
        for code in codes :
            tmp = price.query(f"SecuritiesCode == '{code}'")\
                    .sort_values('Date')\
                    .loc[:,['Date','SecuritiesCode','Close','AdjustmentFactor']]\
                    .reset_index(drop=True)
            tmp = tmp.assign(cadj=tmp.loc[:,'AdjustmentFactor'][::-1].cumprod())
            for c in ['Close'] :
                tmp.loc[:,c] = tmp.loc[:,c] * tmp.cadj

            tmp.loc[:,'rtn_p1d'] = tmp.Close.pct_change(1)
            aps.append(tmp.loc[:,['Date','SecuritiesCode','rtn_p1d']])

        aps = pd.concat(aps).reset_index(drop=True)

        return aps