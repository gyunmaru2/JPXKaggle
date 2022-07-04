# %%
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer
import gc
from multiprocessing import Pool

from tech_etl_online import technicals_etl
from fin_etl_online import financials_etl

# %%

def _normalize_rank_gauss(pdf):

    use_cols = pdf.drop(columns=['SecuritiesCode']).columns
    n_quantiles = pdf.shape[0]
    qt = QuantileTransformer(n_quantiles=n_quantiles,
        random_state=0,output_distribution='normal')
    qt.fit(pdf[use_cols])
    pdf[use_cols] = qt.transform(pdf[use_cols])

    return pdf


# %%

class prepare_dataset_ver3(object):

    prices : pd.DataFrame = None
    fin : pd.DataFrame = None
    train_files = "/kaggle/input/"\
        "jpx-tokyo-stock-exchange-prediction/train_files/"
    supplemental_files = "/kaggle/input/"\
        "jpx-tokyo-stock-exchange-prediction/supplemental_files/"
    stock_list_file =  "/kaggle/input/"\
        "jpx-tokyo-stock-exchange-prediction/stock_list.csv"

    storage_dir = "/kaggle/working/JPXKaggle/"
    model_dir = "/kaggle/input/jpx-production1/"

    def __init__(self,private_board=False):

        self.private_board = private_board
        self.stock_list = pd.read_csv(
            self.stock_list_file, dtype = {"SecuritiesCode":str},
            usecols=["SecuritiesCode","17SectorCode"]
        )
        self.read_prices()
        self.read_fin()
        self.read_base_feat()

        return

    def read_prices(self) :

        if self.private_board :
            # self.prices = pd.read_csv( 
            #     self.train_files+"stock_prices.csv",
            #     skiprows=list(
            #         range(1,int(2_000*250*3))
            #         ),
            #     dtype={"SecuritiesCode":str,'Date':str}
            # )
            self.prices = pd.read_csv( 
                self.train_files+"stock_prices.csv",
                dtype={"SecuritiesCode":str,'Date':str}
            )
            
            dates = self.prices.Date.unique()
            thrs = dates[-120]
            self.prices = self.prices.loc[self.prices.Date>=thrs,:]

            tmp = pd.read_csv(self.supplemental_files+"stock_prices.csv",
                dtype={"SecuritiesCode":str,"Date":str}
            )
            self.prices = pd.concat([self.prices,tmp])\
                    .reset_index(drop=True)

            for c in ['Open','High','Low','Close','Volume','AdjustmentFactor'] :
                self.prices.loc[:,c] = pd.to_numeric(
                    self.prices.loc[:,c],errors = "coerce")
                
            if 'Target' in self.prices.columns :
                del(self.prices['Target'])
        else :
            pass

        return

    def read_fin(self) :

        if self.private_board :
            self.fin = pd.read_csv( 
                self.train_files+"financials.csv",
                dtype={"SecuritiesCode":str,"Date":str}
            )
            self.fin = self.fin.loc[self.fin.Date >= "2019-09-30", :]
            tmp = pd.read_csv(self.supplemental_files+"financials.csv",
                dtype={"SecuritiesCode":str,"Date":str}
            )
            self.fin = pd.concat([self.fin,tmp])
            self.fin = self.fin.sort_values(['SecuritiesCode','Date'])\
                    .reset_index(drop=True)

        else :
            pass

        return

    def read_base_feat(self) :

        if self.private_board :
            self.feat_fin = pd.read_csv(
                self.model_dir+"feat_fin_base.csv", 
                dtype={"SecuritiesCode":str,"Date":str}
            )
            self.feat_fin = self.feat_fin.loc[
                self.feat_fin.Date >= "2019-09-30", :]\
                .reset_index(drop=True)

            self.feat_rev = pd.read_csv(
                self.model_dir+"feat_rev_base.csv",
                dtype={"SecuritiesCode":str,"Date":str}
            )
            self.feat_rev = self.feat_rev.loc[
                self.feat_rev.Date >= "2019-09-30", :]\
                .reset_index(drop=True)
        else :
            pass

        return

    def add_online_data(self,newdf:pd.DataFrame,key:str):

        if (key == 'price') & (self.prices is not None ):

            # original num rows
            norg = len(self.prices)

            #prepare datatypes
            newdf['SecuritiesCode'] = newdf['SecuritiesCode'].astype(str)
            for c in ['Open','High','Low','Close','Volume','AdjustmentFactor'] :
                newdf.loc[:,c] = pd.to_numeric(
                    newdf.loc[:,c],errors = "coerce")


            # concat
            self.prices = pd.concat(
                [self.prices,newdf[self.prices.columns]]
            )

            # drop duplicates
            self.prices = self.prices\
                .drop_duplicates(subset=['SecuritiesCode','Date'])\
                .reset_index(drop=True)

            nnew = len(self.prices)

            if nnew > norg :
                # delete old data
                min_date = self.prices.Date.min()
                self.prices = self.prices.loc[self.prices.Date != min_date,:]\
                    .reset_index(drop=True)

        if (key == 'fin') & (self.fin is not None):
            self.fin = pd.concat(
                [self.fin,newdf[self.fin.columns]]
            )

            # drop duplicates
            self.fin = self.fin\
                .drop_duplicates(subset=['SecuritiesCode','Date'])\
                .reset_index(drop=True)

        return


    def etl_prepare_features_tech(self,calc_date):

        tech = technicals_etl()
        tech.price = self.prices\
            .loc[self.prices.Date <= calc_date, :]\
            .reset_index(drop=True)

        df = tech.run_etl_groupby_mp()
        df = df.loc[df.Date==calc_date,:].reset_index(drop=True)
        del tech.price; del tech; gc.collect()

        cols = df.drop(columns=['Date','SecuritiesCode']).columns
        for c in cols :
            df.loc[:,c] = pd.to_numeric(df.loc[:,c],errors='coerce')
        return df

    def etl_prepare_fatures_fin(self,calc_date,new_codes) :

        ff = financials_etl()
        ff.fin = self.fin\
                .loc[self.fin.Date <= calc_date,:]\
                .reset_index(drop=True)

        feat_fin_new = ff.run_etl_groupby_mp_online(calc_date,new_codes)

        self.feat_fin = pd.concat([self.feat_fin,feat_fin_new])\
                .sort_values(['SecuritiesCode','Date'])\
                .drop_duplicates(subset=['SecuritiesCode','Date'])\
                .reset_index(drop=True)

        feat_rev_new = ff.run_fr_groupby_mp_online(calc_date,new_codes)
        self.feat_rev = pd.concat([self.feat_rev,feat_rev_new])\
                .sort_values(['SecuritiesCode','Date'])\
                .drop_duplicates(subset=['SecuritiesCode','Date'])\
                .reset_index(drop=True)

        return


    def etl_prepare_dataset(self,idx,calc_date,new_codes=None,debug=False):

        # add 17 sector code
        idx = idx.merge(self.stock_list,on=['SecuritiesCode'],how="left")

        # technical features 
        tech = self.etl_prepare_features_tech(calc_date)
        drop_cols_tech = ['Date']
        tech = tech.drop(columns=drop_cols_tech)
        # fnames1 = tech.drop(columns=['SecuritiesCode']).columns

        #financial features
        if new_codes is not None :
            self.etl_prepare_fatures_fin(calc_date,new_codes)
        
        fin1 = self.feat_fin.loc[self.feat_fin.Date <= calc_date, :]\
                .copy()
        fin1 = fin1.sort_values(['SecuritiesCode','Date'])\
                .drop_duplicates(subset=['SecuritiesCode'],keep='last')\
                .drop(columns=['Date'])
        fin2 = self.feat_rev.loc[self.feat_rev.Date <= calc_date, :]\
                .copy()
        fin2 = fin2.sort_values(['SecuritiesCode','Date'])\
                .drop_duplicates(subset=['SecuritiesCode'],keep='last')\
                .drop(columns=['Date'])
        
        # fnames2 = fin1.drop(columns=['SecuritiesCode']).columns
        # fnames3 = fin2.drop(columns=['SecuritiesCode']).columns

        # merge frames
        idx = idx.merge(tech,on=['SecuritiesCode'],how="left")
        idx = idx.merge(fin1,on=['SecuritiesCode'],how="left")
        idx = idx.merge(fin2,on=['SecuritiesCode'],how="left")
        idx = idx.drop_duplicates(['SecuritiesCode'],keep='last')

        use_cols = idx.drop(columns=['SecuritiesCode']).columns

        idx[use_cols] = idx[use_cols].replace([np.inf, -np.inf], np.nan)
        idx[use_cols] =  idx[use_cols].fillna(idx[use_cols].mean())

        if debug :
            print(idx)

        #Rank Gauss
        #make this sector neutral
        #https://tsumit.hatenablog.com/entry/2020/06/20/044835
        #https://scikit-learn.org/stable/modules/preprocessing.html

        dfg = idx.groupby(['17SectorCode'])
        with Pool(3) as p:
            res_list = p.map(
                _normalize_rank_gauss,[group for _,group in dfg])
        dfg = pd.concat(res_list)
        dfg = dfg.drop(columns=['17SectorCode'])

        return idx



