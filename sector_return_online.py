# %%

import numpy as np
import pandas as pd
import os, gc
from multiprocessing import Pool
import multiprocessing

# %%

def _pipeline (pdf) :

    tmp = pdf.copy()
    tmp = tmp.sort_values('Date').reset_index(drop=True)
    tmp = tmp.assign(cadj=tmp.AdjustmentFactor[::-1].cumprod())
    for c in ['Open','High','Low','Close'] :
        tmp.loc[:,c] = tmp.loc[:,c] * tmp.cadj
    tmp.loc[:,"PrevClose"] = pd.concat([pd.Series(np.nan),tmp.Close.iloc[:-1]])\
        .reset_index(drop=True)

    tmp = _uki_tech_day(tmp)

    return tmp

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
class sector_return_etl(object) :

    def __init__(self) :

        self.price = None
        self.stock_list = None

    def run_etl_groupby_mp(self,debug=False):
        
        price = self.price
        sl = self.stock_list
        
        dfg = price.groupby('SecuritiesCode')
        with Pool(3) as p:
            res_list = p.map(
                _pipeline, [group for name, group in dfg]
            )
        dfg = pd.concat(res_list).reset_index(drop=True)
        # dfg.SecuritiesCode = dfg.SecuritiesCode.astype(int)

        return dfg

 
        