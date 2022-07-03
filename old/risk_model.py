# %%

import numpy as np
import pandas as pd
import os
import json

class risk_model(object) :

    covmat = None
    spesific_risk = None
    betas = None

    def __init__(self,local=False) :

        self.local = local

        if local :
            with open("./local_settings.json") as f:
                config_ = json.load(f)
            self.storage = config_['storage']

    def predict(self,w:pd.DataFrame):

        r"""
            calculates portfolio estimated risk

            Parameters 
            ----------------------------------

            w: pandas.DataFrame
                dataframe with column of SecuritiesCode,weight

            Returns
            --------------------------------
            port_risk:float
        
        """

        if self.spesific_risk is None :
            self.read_spesific_risk()
        if self.covmat is None :
            self.read_covmat()
        if self.betas is None :
            self.read_betas()

        tmps = w.merge(self.spesific_risk,on='SecuritiesCode',
                    how="left")\
                .sort_values('SecuritiesCode')
        tmpb = w.merge(self.betas,on=['SecuritiesCode'],how='left')

        wgt = tmpb.loc[:,'weight'].values.reshape(-1,1)
        sp_risk = tmps.loc[:,'sp_risk'].values
        betas = tmps.drop(columns=['SecuritiesCode','weight']).values
        port_beta = wgt.T.dot(betas)

        risk = self.calc_risk(
            factor_cov=self.covmat,
            port_beta=port_beta,
            port_weight=wgt,
            spesific_risk=sp_risk
        )

        return risk

    def calc_risk(self,
            factor_cov:np.ndarray,
            port_beta:np.ndarray,
            port_weight:np.ndarray,
            spesific_risk:np.ndarray
        ) :

        r"""
            Calcurates portflio estimated risk

            Parameters 
            ---------------------------
            factor_cov: numpy.ndarray
                covariance matrix of factor returns
            port_beta: numpy.ndarray
                must be shape of (M,1), M as number of factors
                portfolio risk exposure to factors
            port_weight: numpy.ndarray
                shape of (N,)
                portfolio weights
            spesific_risk: numpy.ndarray
                Shape of (N,)
                spesific risks (standard deviasion) of each stock, s

            Returns
            ---------------------------
            risk : float
                risk of portfolio
                TBA should this be annualized?
        
        """

        risk_from_factors = port_beta.T.dot(factor_cov).dot(port_beta)
        risk_from_spesific = (port_weight**2 * spesific_risk**2).sum()

        return np.sqrt(risk_from_factors+risk_from_spesific)

    def calc_factor_covariance_matrix(self,factor_returns:np.ndarray,
        save_matrix=False
    ):

        T,N = factor_returns.shape

        covmat = factor_returns.T.dot(factor_returns)
        covmat /= T

        if save_matrix :
            pd.DataFrame(covmat)\
                .to_csv(self.storage+"factor_cov.max",
                    index=False
                )

        return covmat

    def read_covmat(self):
        covmat = pd.read_csv(self.storage+"factor_covmat.csv")
        self.covmat = covmat


    def calc_spesific_risk(self,spesific_returns:pd.DataFrame,
        save_matrix=False
    ):
        spr = np.std(spesific_returns,axis=0).reshape(-1,1)

        spr = spesific_returns\
                .loc[:,['SecuritiesCode','sp_rtn']]\
                .groupby('SecuritiesCode')\
                .std()\
                .reset_index()

        if save_matrix:
            spr = pd.rename(columns={'sp_rtn':'sp_risk'})\
                .to_csv(self.storage+"spesific_risk.csv",
                    index=False)

        self.spesific_risk = spr

        return spr

    def read_spesific_risk(self):

        spesific_risk = pd.read_csv(self.storage+"spesific_risk.csv")
        self.spesific_risk = spesific_risk

    def read_betas(self):

        betas = pd.read_csv(self.storage+"betas.csv")
        self.betas = betas


    def _prepare_files_on_local(self):

        assert self.local, """
            this function is intended be used on local environment
        """

        factor_returns = pd.read_csv(self.storage+"factor_returns.csv")
        self.calc_factor_covariance_matrix(factor_returns,save_matrix=True)
        del(factor_returns)

        spesific_returns = pd.read_csv(self.storage+"spesific_returns.csv")
        self.calc_spesific_risk(spesific_returns,save_matrix=True)
        del(spesific_returns)






# %%
