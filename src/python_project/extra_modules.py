#%%
import yfinance as yf
import pandas as pd 
from sec_cik_mapper import StockMapper
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging 
from scipy.optimize import minimize
import numpy as np
from pybacktestchain.data_module import Information

# Setup logging
logging.basicConfig(level=logging.INFO)

#---------------------------------------------------------
# Constants
#---------------------------------------------------------

UNIVERSE_SEC = list(StockMapper().ticker_to_cik.keys())

#---------------------------------------------------------
# Classes 
#---------------------------------------------------------

#Extend the possibilities for the user. The user could now chose between different asset allocation strategies:
#   - Two First Moments from pybacktestchain
#   - Risk Parity from python_project
#   - Minimum Variance POrtfolio from python_project
# The user choice is asked by the function strategy_choice in user_function.py file.


# Class that represents the data used in the backtest.   

class RiskParity(Information):
    def compute_portfolio(self, t: datetime, information_set):
        try:
            # Ensure that information_set is valid and contains the expected keys
            if not information_set or 'covariance_matrix' not in information_set or 'companies' not in information_set:
                logging.warning("Incomplete information set provided. Returning equal weight portfolio.")
                return self.equal_weight_portfolio(information_set)

            Sigma = information_set['covariance_matrix']
            n = Sigma.shape[0]  # number of assets
            # Diagonal terms of the covariance matrix:
            inv_vol = 1 / np.sqrt(np.diag(Sigma))
            # Calculate the raw weights, inversely proportional to volatility
            raw_weights = inv_vol / np.sum(inv_vol)
            # Create the portfolio as a dictionary mapping companies to their respective weights            
            portfolio = {k: raw_weights[i] for i, k in enumerate(information_set['companies'])}
            return portfolio
        except Exception as e:
            logging.warning("Error computing portfolio, returning equal weight portfolio.")
            logging.warning(e)
            return self.equal_weight_portfolio(information_set)
    
    def equal_weight_portfolio(self, information_set):
        if not information_set or 'companies' not in information_set:
            logging.warning("Information set is missing companies. Defaulting to empty portfolio.")
            return {}

        num_companies = len(information_set['companies'])
        return {k: 1/num_companies for k in information_set['companies']}

    def compute_information(self, t: datetime):
        data = self.slice_data(t)
        information_set = {}
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()
        # Compute expected returns as the average of daily returns
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()
        data = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column)
        data = data.dropna(axis=0)

        if data.empty:
            logging.warning("Data for computing covariance is empty. Returning empty information set.")
            return information_set  # Return an empty information set if there's no data
        
        # Calculate the covariance matrix and store it in the information set
        information_set['covariance_matrix'] = data.cov().to_numpy()
        information_set['companies'] = data.columns.to_numpy()

        # Return the populated information set
        return information_set



class MinimumVariancePortfolio(Information):
    def compute_portfolio(self, t: datetime, information_set):
        try:
            Sigma = information_set['covariance_matrix']
            n = Sigma.shape[0]
            # Define the objective function: minimize portfolio variance
            obj = lambda w: w.T @ Sigma @ w
            # Constraints: The sum of weights must equal 1
            cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1)] * n  # No short selling
            x0 = np.ones(n) / n # Initial weights
            res = minimize(obj, x0, constraints=cons, bounds=bounds)
            portfolio = {k: None for k in information_set['companies']}
            if res.success:
                for i, company in enumerate(information_set['companies']):
                    portfolio[company] = res.x[i]
            else:
                raise Exception("Optimization did not converge")
            return portfolio
        except Exception as e:
            logging.warning("Error computing portfolio, returning equal weight portfolio")
            logging.warning(e)
            return {k: 1/len(information_set['companies']) for k in information_set['companies']}
        
    def compute_information(self, t: datetime):
        data = self.slice_data(t)
        information_set = {}
        data = data.sort_values(by=[self.company_column, self.time_column])
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()
        data = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column)
        data = data.dropna(axis=0)
        # Calculate covariance matrix
        covariance_matrix = data.pct_change().cov().to_numpy()  # Calculate covariance from percentage changes
        
        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data.columns.to_numpy()
        
        return information_set
        
    
        
    
