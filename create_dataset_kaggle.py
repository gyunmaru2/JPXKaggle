

# %%%

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer
import gc


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
        idx.iloc[:,1:] = idx.iloc[:,1:].fillna(0.)


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


# %%

class prepare_dataset_ver2(prepare_dataset):

    def etl_prepare_dataset(self, idx, calc_date):

        tech = pd.read_csv(
            self.storage_dir+"tech_feature.csv"
        )
        drop_cols_tech = ['RowId','Date']
        tech = tech.loc[tech.Date==calc_date,:].reset_index(drop=True)

        tech = tech.drop(columns=drop_cols_tech)
        tech_fnames = tech.drop(columns=['SecuritiesCode']).columns

        #Rank Gauss
        #https://tsumit.hatenablog.com/entry/2020/06/20/044835
        #https://scikit-learn.org/stable/modules/preprocessing.html
        qt = QuantileTransformer(random_state=0, output_distribution='normal')
        qt.fit(tech[tech_fnames])
        # RankGaussによる変換を行い、num_colsの値を置換
        tech[tech_fnames] = qt.transform(tech[tech_fnames])

        idx = idx.merge(tech,on=['SecuritiesCode'],how="left")
        idx.iloc[:,1:] = idx.iloc[:,1:].fillna(0.)

        return idx


# %%
class prepare_dataset_ver3(object):

    prices : pd.DataFrame = None
    fin : pd.DataFrame = None
    train_files = "/kaggle/input/"\
        "jpx-tokyo-stock-exchange-prediction/train_files/"
    supplemental_files = "/kaggle/input/"\
        "jpx-tokyo-stock-exchange-prediction/supplemental_files/"

    storage_dir = "/kaggle/working/JPXKaggle/"

    def __init__(self,private_board=False):

        self.private_board = private_board

    def read_prices(self) :

        if self.private_board :
            self.prices = pd.read_csv( 
                self.train_files+"stock_prices.csv",
                skiprows=list(
                    range(1,int(2_000*250*3))
                    ),
                dtype={"SecuritiesCode":str}
            )
            
            dates = self.prices.Date.unique()
            thrs = dates[-120]
            self.prices = self.prices.loc[
                self.prices.Date>=thrs,:]\
                    .reset_index(drop=True)

            for c in ['Open','High','Low','Close','Volume','AdjustmentFactor'] :
                self.prices.loc[:,c] = pd.to_numeric(
                    self.prices.loc[:,c],errors = "coerce")
                
            del(self.prices['Target'])
        else :
            pass

    def add_online_data(self,newdf:pd.DataFrame,key:str):

        if (key == 'price') & (self.prices is not None ):

            #prepare datatypes
            newdf['SecuritiesCode'] = newdf['SecuritiesCode'].astype(str)

            # concat
            self.prices = pd.concat(
                [self.prices,newdf[self.prices.columns]]
            )

            # delete old data
            min_date = self.prices.Date.min()
            print("min_date: ",min_date)
            self.prices = self.prices.loc[self.prices.Date != min_date,:]\
                .reset_index(drop=True)

        if (key == 'fin') & (self.fin is not None):
            self.fin = pd.concat(
                [self.fin,newdf[self.fin.columns]]
            )


    def etl_prepare_features_tech(self,calc_date):

        tech = technicals_etl()
        tech.price = self.prices
        df = tech.run_etl_day()
        df = df.loc[df.Date==calc_date,:].reset_index(drop=True)
        del tech.price; del tech; gc.collect()
        return df

    def etl_prepare_dataset(self,idx,calc_date):

        tech = self.etl_prepare_features_tech(calc_date)
#         drop_cols_tech = ['RowId','Date']
        drop_cols_tech = ['Date']
        tech = tech.drop(columns=drop_cols_tech)

        tech_fnames = tech.drop(columns=['SecuritiesCode']).columns

        #Rank Gauss
        #https://tsumit.hatenablog.com/entry/2020/06/20/044835
        #https://scikit-learn.org/stable/modules/preprocessing.html
        qt = QuantileTransformer(random_state=0, output_distribution='normal')
        qt.fit(tech[tech_fnames])
        # RankGaussによる変換を行い、num_colsの値を置換
        tech[tech_fnames] = qt.transform(tech[tech_fnames])

        idx = idx.merge(tech,on=['SecuritiesCode'],how="left")
#         idx.iloc[:,1:] = idx.iloc[:,1:].fillna(0.)

        return idx






