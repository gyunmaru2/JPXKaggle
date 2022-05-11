
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
        self.kaggle_data = config_['kaggle_data']

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

    def index_return(self,pdf,thrs=(0.01,0.99)) :

        b,u = thrs
        if pdf.shape[0]== 0 :
            return np.nan
        assert ('rtn_p1d' in pdf.columns), """
            no column named rtn_p1d
        """
        qu = pdf.loc[:,"rtn_p1d"].quantile(u)
        qb = pdf.loc[:,"rtn_p1d"].quantile(b)

        return pdf.rtn_p1d.where(pdf.rtn_p1d<=qu,np.nan)\
                .where(pdf.rtn_p1d>=qb,np.nan)\
                .mean()

    def market_return(self,stock_returns=None):

        return_stock_returns=False
        if stock_returns is None :
            return_stock_returns=True
            stock_returns = self.return_on_adjusted_close()

            mkt_rtn = stock_returns.loc[:,['Date','rtn_p1d']]\
                .groupby('Date')\
                .apply(self.index_return)
        if return_stock_returns :
            return mkt_rtn, stock_returns
        else :
            return mkt_rtn

    def sector_return(self,stock_returns=None,
        mkt_return=None) :

        return_stock_returns=False; return_mkt_return=False
        if (stock_returns is None) & (mkt_return is None):
            return_stock_returns=True;return_mkt_return=True
            mkt_return, stock_returns = self.market_return()
        elif stock_returns is None :
            return_stock_returns=True
            stock_returns = self.sector_return()
        elif mkt_return is None :
            return_mkt_return=True
            mkt_return = self.market_return(stock_returns)

        sl = pd.read_csv(self.kaggle_data+"stock_list.csv",
            dtype=str)
        inds = stock_returns.merge(sl.loc[:,['SecuritiesCode','33SectorCode']],
            on="SecuritiesCode",how="left")\
            .drop(columns=['SecuritiesCode'])\
            .groupby(['Date','33SectorCode'])\
            .apply(self.index_return)

        inds = inds.reset_index().rename(columns={0:"rtn_p1d"})

        if return_mkt_return & return_stock_returns :
            return inds, mkt_return, stock_returns
        elif return_mkt_return :
            return inds, mkt_return
        else :
            return inds