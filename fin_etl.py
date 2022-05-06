# %%

from ast import Pass
import numpy as np
import pandas as pd
import os 
import sys

# %%

class financials_etl(object) :


    PATH = None
    id = ['DisclosureNumber','DateCode','Date','SecuritiesCode',
            'DisclosedDate','DisclosedTime','DisclosedUnixTime',
            'TypeOfDocument','CurrentPeriodEndDate','TypeOfCurrentPeriod',
            'CurrentFiscalYearStartDate','CurrentFiscalYearEndDate']
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

    def __init__(self,data_dir) :

        self.data_dir = data_dir
        self.fin = pd.read_csv(data_dir+'financials.csv',
                    dtype=str
                )

    def fin2qtr(self,df) :

        for k,v in self.r_cn.items() :
            df.loc[:,k] = np.nan
            df.loc[df.TypeOfCurrentPeriod == "1Q",k]=df.loc[:,v]
            df.loc[df.TypeOfCurrentPeriod != "1Q",k]=df.loc[:,v].diff(1)
        df.loc[:,'r_assets']=df.loc[:,'TotalAssets']
        df.loc[:,'r_equity']=df.loc[:,'Equity']

        return df 

    def finqtr2ltm(self,df) :

        for k,v in self.r_cn.items() :
            df.loc[:,k] = df.loc[:,k].rolling(4).sum()

        return df 

    def screen_stock(self,df,code) :

        out = df.loc[(df.SecuritiesCode==code)&
            (df.TypeOfDocument.str.contains('FinancialStatements')),:]\
                .sort_values('DisclosedDate')\
                .reset_index(drop=True)

        return out

    def screen_column(self,df) :

        cols = self.id +list(self.r_cn.values()) +list(self.b_cn.values())\
                 + list(self.f_cn.values())
        df = df.loc[:,cols]

        cols2 = list(self.r_cn.values()) +list(self.b_cn.values())\
                 + list(self.f_cn.values())

        for c in cols2 :
            df.loc[:,c] = pd.to_numeric(df.loc[:,c],errors='coerce')

        return df

    def choose_use_column(self,df) :

        cols = self.id + self.feat2 + self.feat3
        col1 = ["d_"+f for f in self.feat2]
        col2 = ["d_"+f for f in self.feat3]
        cols.extend(col1)
        cols.extend(col2)

        df = df.loc[:,cols]
        return df

    def fin_results_add(self,df) :


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

    def fin_forcast_add(self,df) :

        chng_name = {v:k for k,v in self.f_cn.items()}
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

    def fin_diff_add(self,df) :

        #Inf値をNan値化
        df = df.replace([np.inf,-np.inf],np.nan)

        #差分系列
        for f in self.feat2 :
            df.loc[:,f"d_{f}"] = df.loc[:,f].diff(1)

        for f in self.feat3 :
            df.loc[:,f"d_{f}"] = df.loc[:,f].diff(1)

        return df


    def run_etl(self,debug=False) :
        self.debug = debug

        fin = self.fin
        codes = fin.SecuritiesCode.dropna().unique()

        dfs = []
        for count,code in enumerate(codes) :

            print(code)

            tmp = self.screen_stock(fin,code)
            if tmp.shape[0] == 0 :
                continue
            tmp = self.screen_column(tmp)
            tmp = self.fin2qtr(tmp)
            tmp = self.finqtr2ltm(tmp)

            tmp = self.fin_results_add(tmp)
            tmp = self.fin_forcast_add(tmp)
            tmp = self.fin_diff_add(tmp)

            tmp = self.choose_use_column(tmp)

            dfs.append(tmp)

            if self.debug & (count==2) :
                break

        return pd.concat(dfs)

        






