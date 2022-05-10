# Databricks notebook source
# MAGIC %md # calc market beta for each stock on stock list

# COMMAND ----------

import numpy as np
import pandas as pd
import os 
import json

# COMMAND ----------

with open("./local_settings.json") as f:

    config_ = json.load(f)

# COMMAND ----------

df = pd.read_csv(config_['train_files']+"stock_prices.csv",
    dtype=str)
df = df.loc[:,['Date','SecuritiesCode','Close','AdjustmentFactor']]
for c in ["Close",'AdjustmentFactor'] :
    df.loc[:,c] = pd.to_numeric(df.loc[:,c],errors="coerce")

dates = df.Date.unique()
codes = df.SecuritiesCode.unique()

# COMMAND ----------

# MAGIC %md ## return on adjusted Close price

# COMMAND ----------

aps = []
for code in codes :
    tmp = df.loc[df.SecuritiesCode==code,:]
    if len(tmp) == 0 :
        continue
    # adjusted price

    tmp = df.query(f"SecuritiesCode == '{code}'")\
        .sort_values('Date').reset_index(drop=True)
    if len(tmp) == 0 :
        continue
    tmp = tmp.assign(cadj=tmp.AdjustmentFactor[::-1].cumprod())
    for c in ['Close'] :
        tmp.loc[:,c] = tmp.loc[:,c] * tmp.cadj
    tmp.loc[:,'rtn_p1d'] = tmp.Close.pct_change(1)

    aps.append(tmp.loc[:,['Date','SecuritiesCode','rtn_p1d']])

aps = pd.concat(aps).reset_index(drop=True)

aps.head()

# COMMAND ----------

def market_return(pdf) :

    if pdf.shape[0]== 0 :
        return np.nan

    assert ('rtn_p1d' in pdf.columns), """
        no column named rtn_p1d
    """

    q99 = pdf.loc[:,"rtn_p1d"].quantile(0.99)
    q1 = pdf.loc[:,"rtn_p1d"].quantile(0.01)

    return pdf.rtn_p1d.where(pdf.rtn_p1d<=q99,np.nan)\
            .where(pdf.rtn_p1d>=q1,np.nan)\
            .mean()

mkt_rtn = aps.loc[:,['Date','rtn_p1d']]\
            .groupby('Date')\
            .apply(market_return)





# COMMAND ----------

# MAGIC %md # Sector returns

# COMMAND ----------

sl = pd.read_csv(config_['kaggle_data']+'stock_list.csv',
                    dtype=str)

# COMMAND ----------

# MAGIC %md #### simple sector return

# COMMAND ----------

inds = aps.merge(sl.loc[:,['SecuritiesCode','33SectorCode']],
          on="SecuritiesCode",how="left")\
    .drop(columns=['SecuritiesCode'])\
    .groupby(['Date','33SectorCode'])\
    .apply(market_return)

# COMMAND ----------

inds = inds.reset_index().rename(columns={0:"rtn_p1d"})\
    .pivot(index="Date",columns="33SectorCode",values='rtn_p1d')

# COMMAND ----------

# MAGIC %md #### ex market beta

# COMMAND ----------

from sklearn.linear_model import LinearRegression

inds_xb = []
for sec in inds.columns :
    
    tmp = inds.loc[:,sec].reset_index()\
        .merge(mkt_rtn.reset_index().rename(columns={0:'mkt'})
                  ,on="Date",how="left"
              )\
        .dropna()
    y = tmp.loc[:,sec].values
    X = tmp.mkt.values.reshape(-1,1)
    reg = LinearRegression().fit(X,y)
    res = y - reg.predict(X)
    
    res = pd.DataFrame(res,index=tmp.Date,columns=[sec])
    inds_xb.append(res)
    
inds_xb = pd.concat(inds_xb,axis=1)

# COMMAND ----------

inds_xb

# COMMAND ----------

# MAGIC %md # calc beta

# COMMAND ----------

from tqdm import tqdm
from sklearn.linear_model import LinearRegression

res = []
for code in tqdm(codes) :
    
    tmp = aps.loc[aps.SecuritiesCode==code,:]
    X = mkt_rtn.reset_index()\
            .rename(columns={0:"mkt"})
    tmp = tmp.merge(X,on='Date',how="left")
    sector = sl.query(f"SecuritiesCode=='{code}'").loc[:,"33SectorCode"].iloc[0]
    tmp = tmp.merge(
        inds_xb.loc[:,[sector]].rename(columns={sector:"sector"}),
        left_on="Date",right_index=True,how="left")
    tmp = tmp.dropna()
    y = tmp.rtn_p1d.values
    X = tmp.loc[:,['mkt','sector']].values
    
    row = [code,sector]
    reg = LinearRegression().fit(X,y)
    
    row.extend(reg.coef_)
    row.append(reg.intercept_)
    
    res.append(row)

# COMMAND ----------

pd.DataFrame(res,columns=['SecuritiesCode','33SectorCode','mkt','sec','alpha'])

# COMMAND ----------

pd.DataFrame(res,columns=['SecuritiesCode','33SectorCode','mkt','sec','alpha'])\
    .to_csv(config_['storage']+"betas.csv",index=False)
