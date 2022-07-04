# %%

import numpy as np
import pandas as pd
import os, sys, gc
from multiprocessing import Pool
import multiprocessing

# %%

id = ["Date", "SecuritiesCode"]

r_cn = dict(
    r_sales = 'NetSales', r_op = 'OperatingProfit', 
    r_ordp = "OrdinaryProfit",r_ni = "Profit"
)

f_cn = dict(

    f_sales = 'ForecastNetSales', f_op = 'ForecastOperatingProfit',
    f_ordp = 'ForecastOrdinaryProfit', f_ni = "ForecastProfit"
)

b_cn = dict(
    r_asset = 'TotalAssets', r_equity = "Equity"
)

feat1 = ['annual']

feat2 = ["r_sales", "r_op", "r_ordp", "r_ni", 
                "r_expense1", "r_expense2", "r_expense3",
                "f_sales", "f_op", "f_ordp", "f_ni", 
                "f_expense1", "f_expense2", "f_expense3",
                "r_assets", "r_equity"
                ]
feat3 = [
    "r_pm1", "r_roe1", "r_roa1", 
    "r_pm2", "r_roe2", "r_roa2", 
    "r_pm3", "r_roe3", "r_roa3", 
    "r_cost1", "r_cost2", "r_cost3", 
    "r_turn",  
    "f_pm1", "f_roe1", "f_roa1", 
    "f_pm2", "f_roe2", "f_roa2", 
    "f_pm3", "f_roe3", "f_roa3", 
    "f_cost1", "f_cost2", "f_cost3", 
    "f_turn",  
    "equity_ratio"
    ]

feat4 = ["c_sales","c_op","c_ordp","c_ni"]


# %%
def _pct_change_abs(x:pd.Series) :
    # x = pd.to_numeric(x)
    x_ = x.shift(1)
    rtn = (x-x_)/x_.abs()
    return rtn

def _forecast_revision(df) :

    df = df.reset_index(drop=True)
    ind = df.loc[df.TypeOfDocument=="ForecastRevision",:].index.values
    ind = ind[ind!=0]
    if len(ind) == 0 :
        return(pd.DataFrame({},columns=['Date','SecuritiesCode',
            'frev_sales','frev_op','frev_ordp','frev_ni']))
    cols = ['ForecastNetSales','ForecastOperatingProfit',
        'ForecastOrdinaryProfit','ForecastProfit']
    df[cols] = df[cols].apply(_pct_change_abs)
    df = df.loc[:,['Date','SecuritiesCode',*cols]]
    df.columns = ['Date','SecuritiesCode_',
        'frev_sales','frev_op','frev_ordp','frev_ni'
    ]
    df = df.iloc[ind,:].reset_index(drop=True)

    return df


def _fin2qtr(df):

    for k,v in r_cn.items() :
        df.loc[:,k] = np.nan
        df.loc[df.TypeOfCurrentPeriod == "1Q",k]=df.loc[:,v]
        df.loc[df.TypeOfCurrentPeriod != "1Q",k]=df.loc[:,v].diff(1)
    df.loc[:,'r_assets']=df.loc[:,'TotalAssets']
    df.loc[:,'r_equity']=df.loc[:,'Equity']

    return df 

def _finqtr2ltm(df) :
    for k,v in r_cn.items() :
        df.loc[:,k] = df.loc[:,k].rolling(4).sum()
    return df 

def _fin_results_add(df):

    #コスト
    df.loc[:,'r_expense1']=df.loc[:,'r_sales']-df.loc[:,'r_op']
    df.loc[:,'r_expense2']=df.loc[:,'r_op']-df.loc[:,'r_ordp']
    df.loc[:,'r_expense3']=df.loc[:,'r_ordp']-df.loc[:,'r_ni']

    #複合指標　原系列
    #利益系
    for i, profit in enumerate(['r_ni','r_ordp','r_op']) :
        df.loc[:,f'r_pm{i+1}'] = df.loc[:,profit] / df.loc[:,'r_sales']
        df.loc[:,f'r_roe{i+1}'] = df.loc[:,profit] / df.loc[:,'r_equity']
        df.loc[:,f'r_roa{i+1}'] = df.loc[:,profit] / df.loc[:,'r_assets']

    #コスト
    for i in range(1,4) :
        df.loc[:,f"r_cost{i}"]=df.loc[:,f'r_expense{i}']/\
            df.loc[:,"r_sales"]

    #売上高回転率
    df.loc[:,'r_turn'] = df.loc[:,'r_assets'] / df.loc[:,'r_sales']

    #財務健全性
    df.loc[:,'equity_ratio']=df.loc[:,'r_equity']/df.loc[:,'r_assets']

    return df

def _fin_forcast_add(df) :

    chng_name = {v:k for k,v in f_cn.items()}
    df = df.rename(columns=chng_name)

    df.loc[:,"f_expense1"] = df.loc[:,'f_sales'] - df.loc[:,'f_op']
    df.loc[:,'f_expense2'] = df.loc[:,'f_op'] - df.loc[:,'f_ordp']
    df.loc[:,'f_expense3'] = df.loc[:,'f_ordp'] - df.loc[:,'f_ni']

    for i, profit in enumerate(['f_ni','f_ordp','f_op']) :
        df.loc[:,f"f_pm{i+1}"] = df.loc[:,profit]/df.loc[:,'f_sales']
        df.loc[:,f"f_roe{i+1}"] = df.loc[:,profit]/df.loc[:,'r_equity']
        df.loc[:,f"f_roa{i+1}"] = df.loc[:,profit]/df.loc[:,'r_assets']
    
    #コスト
    for i in range(1,4) :
        df.loc[:,f"f_cost{i}"]=df.loc[:,f'f_expense{i}']/\
            df.loc[:,"f_sales"]

    #売上高回転率
    df.loc[:,'f_turn'] = df.loc[:,'r_assets'] / df.loc[:,'f_sales']

    return df

def _fin_milestone(df):

    df['c_sales'] = df['r_sales'] / df['f_sales'].shift(1)
    df['c_op'] = df['r_op'] / df['f_op'].shift(1)
    df['c_ordp'] = df['r_ordp'] / df['f_ordp'].shift(1)
    df['c_ni'] = df['r_ni'] / df['f_op'].shift(1)

    return df


def _fin_diff_add(df) :

    #Inf値をNan値化
    df = df.replace([np.inf,-np.inf],np.nan)

    #差分系列
    for f in feat2 :
        df.loc[:,f"d_{f}"] = df.loc[:,f].pct_change(1)

    for f in feat3 :
        df.loc[:,f"d_{f}"] = df.loc[:,f].diff(1)

    for f in feat4 :
        df.loc[:,f"d_{f}"] = df.loc[:,f].diff(1)

    return df

def _choose_use_column(df) :

    cols = id + feat2 + feat3 + feat4
    col1 = ["d_"+f for f in feat2]
    col2 = ["d_"+f for f in feat3]
    col3 = ["d_"+f for f in feat4]
    cols.extend(col1)
    cols.extend(col2)
    cols.extend(col3)

    df = df.loc[:,cols]
    return df


def _pipeline(pdf) :

    pdf = _fin2qtr(pdf)
    pdf = _finqtr2ltm(pdf)
    pdf = _fin_results_add(pdf)
    pdf = _fin_forcast_add(pdf)
    pdf = _fin_milestone(pdf)
    pdf = _fin_diff_add(pdf)
    pdf = _choose_use_column(pdf)

    return pdf

def _pipeline_fr(pdf) :

    pdf = _forecast_revision(pdf)

    return pdf


# %%

class financials_etl(object) :

    """

    Caution !!

        do not foreget to 
            - read financial files from both train and supplmental files
                - and just keep for 2years, Date >= 2020-03-31
            - prepare base feature files for fin and rev
    
    """


    PATH = None
    id = ['DisclosureNumber','DateCode','Date','SecuritiesCode',
            'DisclosedDate','DisclosedTime','DisclosedUnixTime',
            'TypeOfDocument','CurrentPeriodEndDate','TypeOfCurrentPeriod',
            'CurrentFiscalYearStartDate','CurrentFiscalYearEndDate']
    id_fr = ['SecuritiesCode','Date']
    
    r_cn = dict(
        r_sales = 'NetSales', r_op = 'OperatingProfit', 
        r_ordp = "OrdinaryProfit",r_ni = "Profit"
    )

    f_cn = dict(

        f_sales = 'ForecastNetSales', f_op = 'ForecastOperatingProfit',
        f_ordp = 'ForecastOrdinaryProfit', f_ni = "ForecastProfit"
    )

    b_cn = dict(
        r_asset = 'TotalAssets', r_equity = "Equity"
    )

    feat1 = ['annual']

    feat2 = ["r_sales", "r_op", "r_ordp", "r_ni", 
                 "r_expense1", "r_expense2", "r_expense3",
                    "f_sales", "f_op", "f_ordp", "f_ni", 
                 "f_expense1", "f_expense2", "f_expense3",
                 "r_assets", "r_equity"
                 ]
    feat3 = [
        "r_pm1", "r_roe1", "r_roa1", 
        "r_pm2", "r_roe2", "r_roa2", 
        "r_pm3", "r_roe3", "r_roa3", 
        "r_cost1", "r_cost2", "r_cost3", 
        "r_turn",  
        "f_pm1", "f_roe1", "f_roa1", 
        "f_pm2", "f_roe2", "f_roa2", 
        "f_pm3", "f_roe3", "f_roa3", 
        "f_cost1", "f_cost2", "f_cost3", 
        "f_turn",  
        "equity_ratio"
        ]

    def __init__(self) :

        self.fin = None

    def run_etl_groupby_mp_online(self,calc_date,new_,debug=False) :

        """
        
        create fiancial features for newly given codes for a day

        Parameters
        ------------------------
            
        new_:iterable[str]
            new SecuritiesCodes on a day
        
        """

        self.debug = debug

        fin = self.fin.copy()
        fin = fin.loc[fin.Date <= calc_date , :]

        tmp = []
        for code in new_ :
            tmp.append(fin.loc[fin.SecuritiesCode==code,:])
        fin = pd.concat(tmp)
        del tmp

        fin.TypeOfDocument = fin.TypeOfDocument.fillna("notype")

        fin = fin.loc[
            fin.TypeOfDocument.str.contains('FinancialStatements'),:
        ]
        fin = fin.sort_values(['SecuritiesCode','DisclosedDate'])\
                .reset_index(drop=True)

        cols = self.id +list(self.r_cn.values()) +list(self.b_cn.values())\
                 + list(self.f_cn.values())
        fin = fin.loc[:,cols]

        cols2 = list(self.r_cn.values()) +list(self.b_cn.values())\
                 + list(self.f_cn.values())

        for c in cols2 :
            fin.loc[:,c] = pd.to_numeric(fin.loc[:,c],errors='coerce')

        dfg = fin.groupby('SecuritiesCode')
        with Pool(3) as p:
            res_list = p.map(_pipeline,[group for name,group in dfg])

        res = pd.concat(res_list)\
                .sort_values(["SecuritiesCode","Date"])\
                .drop_duplicates(subset=['SecuritiesCode'],keep="last")

        del fin, dfg, res_list
        gc.collect()

        return res

    def run_fr_groupby_mp_online(self,calc_date,new_,debug=False) :
        self.debug = debug

        fin = self.fin.copy()
        fin = fin.loc[fin.Date <= calc_date , :]

        tmp = []
        for code in new_ :
            tmp.append(fin.loc[fin.SecuritiesCode==code,:])
        fin = pd.concat(tmp)
        del tmp

        fin.TypeOfDocument = fin.TypeOfDocument.fillna("notype")

        fin = fin.sort_values(['SecuritiesCode','DisclosedDate'])\
                .reset_index(drop=True)

        cols = self.id_fr + ['TypeOfDocument'] + list(self.f_cn.values())
        fin = fin.loc[:,cols]

        cols2 = list(self.f_cn.values())

        for c in cols2 :
            fin.loc[:,c] = pd.to_numeric(fin.loc[:,c],errors='coerce')

        fin = fin.dropna(how='all').reset_index(drop=True)

        dfg = fin.groupby('SecuritiesCode')
        with Pool(3) as p:
            res_list = p.map(_pipeline_fr,[group for name,group in dfg])

        res = pd.concat(res_list)\
            .sort_values(["SecuritiesCode","Date"])\
            .drop_duplicates(subset=['SecuritiesCode'],keep="last")

        del fin, dfg, res_list
        gc.collect()

        return res

