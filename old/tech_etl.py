# %%

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import os, gc
from multiprocessing import Pool
import multiprocessing

# %%

def _uki_tech_day(df):

    tmp = df
    calc_date = df.tail(1).Date.iloc[0]
    code = df.SecuritiesCode.iloc[0]
    cols=["Date",'SecuritiesCode']; vals = [calc_date,code]
    
    #騰落率
    for i in [1,5,10,20,40,60,100] :
        vals.append(tmp['Close'].iloc[-1]/tmp['Close'].iloc[-1-i]-1.)
        cols.append(f"ror_{i}")

    #売買代金
    tmp.loc[:,'vol'] = tmp.loc[:,'Volume'] * tmp.loc[:,'Close']
    tmp = tmp.replace([np.inf,-np.inf],np.nan)
    for i in [1,5,10,20,40,60,100] :
        vals.append(tmp['vol'].iloc[-i:].mean())
        cols.append(f"vol_{i}")
    vals.append(vals[cols.index('vol_1')]/vals[cols.index('vol_20')])
    cols.append('d_vol')

    #レンジ
    tmp['range'] = (tmp.loc[:,['PrevClose','High']].max(axis=1)-
                        tmp.loc[:,['PrevClose','Low']].min(axis=1))/tmp.loc[:,'PrevClose']
    tmp = tmp.replace([np.inf,-np.inf],np.nan)

    for i in [1,5,10,20,40,60,100] :
        vals.append(tmp['range'].iloc[-i:].mean())
        cols.append(f"atr_{i}")
    vals.append(vals[cols.index('atr_1')]/vals[cols.index('atr_20')])
    cols.append('d_atr')

    #ギャップレンジ
    tmp.loc[:,'gap_range'] = (np.abs(tmp.loc[:,'Open']-tmp.loc[:,'PrevClose']))\
        /tmp.loc[:,'PrevClose']
    for i in [1,5,10,20,40,60,100] :
        vals.append(tmp['gap_range'].iloc[-i:].mean())
        cols.append(f"g_atr_{i}")

    #デイレンジ
    tmp.loc[:,'day_range'] = (tmp.loc[:,'High']-tmp.loc[:,'Low'])/tmp.loc[:,'PrevClose']
    for i in [1,5,10,20,40,60,100] :
        vals.append(tmp['day_range'].iloc[-i:].mean())
        cols.append(f"d_atr_{i}")

    #ヒゲレンジ
    tmp.loc[:,'hige_range'] = ((tmp.loc[:,'High']-tmp.loc[:,'Low']
                                -np.abs(tmp.loc[:,'Open']-tmp.loc[:,'Close'])))\
        / tmp.loc[:,'PrevClose']
    for i in [1,5,10,20,40,60,100] :
        vals.append(tmp['hige_range'].iloc[-i:].mean())
        cols.append(f"h_atr_{i}")

    #ボラティリティ
    tmp.loc[:,'rtn'] = tmp.Close.pct_change()
    for i in [5,10,20,40,60,100] :
        vals.append(tmp['rtn'].iloc[-i:].mean())
        cols.append(f"vola_{i}")

        #HLバンド
    for i in [5,10,20,40,60,100]:
        vals.append( 
            tmp.loc[:,'High'].iloc[-i:].max() - 
            tmp.loc[:,'Low'].iloc[-i:].min() )
        cols.append(f"hl_{i}")

    del tmp ; gc.collect()

#         print(vals)
    return( pd.DataFrame(np.array(vals).reshape(1,-1),index=[0],columns=cols))
    

# %%
class technicals_etl(object) :

    def __init__(self) :

        self.price = None

    def read_file(self,paths,queries=None) :

        if isinstance(paths,str) :
            paths = [paths]

        if isinstance(queries,str) | (queries is None ) :
            queries = [queries]

        assert len(paths) == len(queries) , f"""
            length of paths({len(paths)}) and queries({len(queries)}) is not same.
        """

        for path,query in zip(paths,queries) :
            tmp = pd.read_csv(path,dtype="str")
            if query is not None :
                tmp = tmp.query(query)
                tmp = tmp.reset_index(drop=True)

            if self.price is None :
                self.price = tmp
            else :
                self.price = pd.concat([self.price,tmp])

        self.price = self.price.reset_index(drop=True)
            
        for c in ['Open','High','Low','Close','Volume','AdjustmentFactor'] :
            self.price.loc[:,c] = pd.to_numeric(self.price.loc[:,c],
                errors = "coerce")

    def add_input_dataframe(self,df) :

        for c in ['Open','High','Low','Close','Volume','AdjustmentFactor'] :
            df.loc[:,c] = pd.to_numeric(df.loc[:,c],
                errors = "coerce")

        self.price = pd.concat([self.price,df])

        self.price = self.price.reset_index(drop=True)

        

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
        
    def technicals_uki_add(self,df) :
        
        tmp=df

        out_ = tmp.loc[:,['RowId','Date','SecuritiesCode']].copy()

        #騰落率
        for i in [1,5,10,20,40,60,100] :
            out_.loc[:,f"ror_{i}"] = tmp['Close'].pct_change(i)
            
        #売買代金
        tmp.loc[:,'vol'] = tmp.loc[:,'Volume'] * tmp.loc[:,'Close']
        tmp = tmp.replace([np.inf,-np.inf],np.nan)
        for i in [1,5,10,20,40,60,100] :
            out_.loc[:,f"vol_{i}"] = tmp.loc[:,"vol"].rolling(i).mean()
        out_.loc[:,'d_vol'] = tmp.loc[:,'vol'] / out_.loc[:,'vol_20']
            
        #レンジ
        tmp['range'] = (tmp.loc[:,['PrevClose','High']].max(axis=1)-
                           tmp.loc[:,['PrevClose','Low']].min(axis=1))/tmp.loc[:,'PrevClose']
        tmp = tmp.replace([np.inf,-np.inf],np.nan)
        
        out_.loc[:,'atr_1'] = tmp['range']
        for i in [5,10,20,40,60,100] :
            out_.loc[:,f'atr_{i}'] = tmp.loc[:,'range'].rolling(i).mean()
        out_.loc[:,'d_atr'] = tmp.loc[:,'range'] / out_.loc[:,'atr_20']
        
        #ギャップレンジ
        tmp.loc[:,'gap_range'] = (np.abs(tmp.loc[:,'Open']-tmp.loc[:,'PrevClose']))\
            /tmp.loc[:,'PrevClose']
        for i in [1,5,10,20,40,60,100] :
            out_.loc[:,f'g_atr_{i}'] = tmp.loc[:,'gap_range'].rolling(i).mean()
            
        #デイレンジ
        tmp.loc[:,'day_range'] = (tmp.loc[:,'High']-tmp.loc[:,'Low'])/tmp.loc[:,'PrevClose']
        for i in [1,5,10,20,40,60,100] :
            out_.loc[:,f'd_atr_{i}'] = tmp.loc[:,'day_range'].rolling(i).mean()
            
        #ヒゲレンジ
        tmp.loc[:,'hige_range'] = ((tmp.loc[:,'High']-tmp.loc[:,'Low']
                                  -np.abs(tmp.loc[:,'Open']-tmp.loc[:,'Close'])))\
            / tmp.loc[:,'PrevClose']
        for i in [1,5,10,20,40,60,100] :
            out_.loc[:,f'h_atr_{i}'] = tmp.loc[:,'hige_range'].rolling(i).mean()
            
        #ボラティリティ
        for i in [5,10,20,40,60,100]:
            out_.loc[:,f'vola_{i}'] = out_.loc[:,'ror_1'].rolling(i).std()
            
        #HLバンド
        for i in [5,10,20,40,60,100]:
            out_.loc[:,f"hl_{i}"] = tmp.loc[:,'High'].rolling(i).max() - \
                tmp.loc[:,'Low'].rolling(i).min()

        del tmp ; gc.collect()

        return out_
        
    def technicals_uki_add_day(self,df):

        tmp = df
        calc_date = df.tail(1).Date.iloc[0]
        code = df.SecuritiesCode.iloc[0]
        cols=["Date",'SecuritiesCode']; vals = [calc_date,code]
        
        #騰落率
        for i in [1,5,10,20,40,60,100] :
            vals.append(tmp['Close'].iloc[-1]/tmp['Close'].iloc[-1-i]-1.)
            cols.append(f"ror_{i}")

        #売買代金
        tmp.loc[:,'vol'] = tmp.loc[:,'Volume'] * tmp.loc[:,'Close']
        tmp = tmp.replace([np.inf,-np.inf],np.nan)
        for i in [1,5,10,20,40,60,100] :
            vals.append(tmp['vol'].iloc[-i:].mean())
            cols.append(f"vol_{i}")
        vals.append(vals[cols.index('vol_1')]/vals[cols.index('vol_20')])
        cols.append('d_vol')

        #レンジ
        tmp['range'] = (tmp.loc[:,['PrevClose','High']].max(axis=1)-
                           tmp.loc[:,['PrevClose','Low']].min(axis=1))/tmp.loc[:,'PrevClose']
        tmp = tmp.replace([np.inf,-np.inf],np.nan)

        for i in [1,5,10,20,40,60,100] :
            vals.append(tmp['range'].iloc[-i:].mean())
            cols.append(f"atr_{i}")
        vals.append(vals[cols.index('atr_1')]/vals[cols.index('atr_20')])
        cols.append('d_atr')

        #ギャップレンジ
        tmp.loc[:,'gap_range'] = (np.abs(tmp.loc[:,'Open']-tmp.loc[:,'PrevClose']))\
            /tmp.loc[:,'PrevClose']
        for i in [1,5,10,20,40,60,100] :
            vals.append(tmp['gap_range'].iloc[-i:].mean())
            cols.append(f"g_atr_{i}")

        #デイレンジ
        tmp.loc[:,'day_range'] = (tmp.loc[:,'High']-tmp.loc[:,'Low'])/tmp.loc[:,'PrevClose']
        for i in [1,5,10,20,40,60,100] :
            vals.append(tmp['day_range'].iloc[-i:].mean())
            cols.append(f"d_atr_{i}")

        #ヒゲレンジ
        tmp.loc[:,'hige_range'] = ((tmp.loc[:,'High']-tmp.loc[:,'Low']
                                  -np.abs(tmp.loc[:,'Open']-tmp.loc[:,'Close'])))\
            / tmp.loc[:,'PrevClose']
        for i in [1,5,10,20,40,60,100] :
            vals.append(tmp['hige_range'].iloc[-i:].mean())
            cols.append(f"h_atr_{i}")

        #ボラティリティ
        tmp.loc[:,'rtn'] = tmp.Close.pct_change()
        for i in [5,10,20,40,60,100] :
            vals.append(tmp['rtn'].iloc[-i:].mean())
            cols.append(f"vola_{i}")

         #HLバンド
        for i in [5,10,20,40,60,100]:
            vals.append( 
                tmp.loc[:,'High'].iloc[-i:].max() - 
                tmp.loc[:,'Low'].iloc[-i:].min() )
            cols.append(f"hl_{i}")

        del tmp ; gc.collect()

#         print(vals)
        return( pd.DataFrame(np.array(vals).reshape(1,-1),index=[0],columns=cols))
       
    def run_etl(self,debug=False) :

        self.debug=debug

        self.price is not None , """
            attr set is not set 
        """

        price = self.price
        codes = price.SecuritiesCode.unique()
        dfs = []
        for i,code in enumerate(codes):

            df = self.select_stock(code)
            df = self.technicals_uki_add(df)
            dfs.append(df)

            if self.debug & (i==2) :
                break

        return pd.concat(dfs)

    def run_etl_day(self,debug=False) :

        price = self.price
        codes = price.SecuritiesCode.unique()
        dfs = []
        for i,code in enumerate(codes):

            df = self.select_stock(code)
            df = self.technicals_uki_add_day(df)
            dfs.append(df)

            if debug & (i==2) :
                break

        dfs = pd.concat(dfs)
        dfs = dfs.reset_index(drop=True)
        dfs.SecuritiesCode = dfs.SecuritiesCode.astype(int)

        return dfs

    def run_etl_day_mp(self,debug=False):

        price = self.price
        codes = price.SecuritiesCode.unique()
        dfs = []
        p = Pool(2)
        for i, code in enumerate(codes):

            df = self.select_stock(code)
            res = p.apply_async(
                _uki_tech_day,args=(df)
            )
            dfs.append(res.get(60*60))

        p.close(); p.terminate(); p.join()

        dfs = pd.concat(dfs)
        dfs = dfs.reset_index(drop=True)
        dfs.SecuritiesCode = dfs.SecuritiesCode.astype(int)

        return dfs

 
        




