

# %%%

import numpy as np
import pandas as pd
from scipy.stats import norm

from tech_etl import technicals_etl
# from fin_etl import financials_etl

class prepare_dataset(object) :

    storage_dir = "/kaggle/working/JPXKaggle/"

    def __init__(self,private_board=False) :

        if private_board :
            self.private_board = private_board
            self.storage_dir += "private_board/"
        else :
            self.private_board = private_board
            self.storage_dir += "public_board/"


    def etl_prepare_features(self) :

        # technicals
        tech = technicals_etl()
        tech.read_file(self.storage_dir+"stock_prices.csv")
        df = tech.run_etl()
        df.to_csv(self.storage_dir+"tech_feature.csv",
            index=False)
        del(df)

        # financials
        # fin = financials_etl()
        # fin.read_file(self.storage_dir+"financials.csv")
        # df =  fin.run_etl()
        # df.to_csv(self.storage_dir+"fin_feature.csv",
        #     index=False)
        # del(df)

    def etl_prepare_dataset(self,idx,calc_date,):

        tech = pd.read_csv(
            self.storage_dir+"tech_feature.csv"
        )
        # fin = pd.read_csv(
        #     self.storage_dir+"fin_feature.csv"
        # )
        # fin = fin.drop_duplicates(subset=['SecuritiesCode','Date'])

        # drop_cols_fin = ['DisclosureNumber', 'DateCode', 'Date',
        # 'DisclosedTime', 'DisclosedUnixTime', 'TypeOfDocument',
        # 'CurrentPeriodEndDate', 'TypeOfCurrentPeriod',
        # 'CurrentFiscalYearStartDate', 'CurrentFiscalYearEndDate',]

        drop_cols_tech = ['RowId','Date']


        # fin = fin.loc[fin.Date <= calc_date, :]
        # fin = fin.sort_values(['SecuritiesCode','Date'])
        # fin = fin.drop_duplicates(subset=['SecuritiesCode'],keep='last')\
        #     .reset_index(drop=True)
        tech = tech.loc[tech.Date==calc_date,:].reset_index(drop=True)
        
        # fin = fin.drop(columns=drop_cols_fin)
        tech = tech.drop(columns=drop_cols_tech)
        tech_fnames = tech.drop(columns=['SecuritiesCode']).columns
        for _tg in tech_fnames:
            tech.loc[:,_tg] = self._normalize_blom(
                tech.loc[:,_tg].values
            )

        
        # idx = idx.merge(fin,on=['SecuritiesCode'],how="left")
        idx = idx.merge(tech,on=['SecuritiesCode'],how='left')

        return idx

    def _normalize_blom(self,x):
        # https://www.statsdirect.com/help/data_preparation/transform_normal_scores.htm
        # filling nan values with mean
        if np.prod(np.isnan(x)) == 0:
            x = np.nan_to_num(x, nan=np.nanmean(x))
            r = np.argsort(np.argsort(x))
            n = len(x)
            return( norm.ppf((r-3/8)/(n+1/4)) )
        else :
            return(np.zeros(len(x)))
