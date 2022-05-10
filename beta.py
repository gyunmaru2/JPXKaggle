# Databricks notebook source

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
    df.loc[:,c] = pd.to_numeric(df.loc[:,c],erros="coerce")

dates = df.Dates.unique()
codes = df.SecuritiesCode.unique()

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
    tmp.loc[:,'rtn_p1d'] = tmp.pct_change(1)

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

    q95 = pdf.loc[:,"rtn_p1d"].quantile(0.95)
    q5 = pdf.loc[:,"rtn_p1d"].quantile(0.05)

    return pdf.rtn_p1d.where(pdf.rtn_p1d<=q95,np.nan)\
            .where(pdf.rtn_p1d>=q5,np.nan)\
            .mean()

mkt_rtn = df.loc[:,['Dates','rtn_p1d']]\
            .groupby('Dates')\
            .apply(market_return)




