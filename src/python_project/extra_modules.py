#The amin idea here is to add other class that give the user an alternative approach to portfolio construction

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

# Class that represents the data used in the backtest.   

class RiskParity(Information):
    def compute_portfolio(self, t: datetime, information_set):
        try:
            Sigma = information_set['covariance_matrix']
            n = Sigma.shape[0]  # number of assets
            
            # Calculate the inverse of the variance (diagonal of the covariance matrix)
            inv_vol = 1 / np.sqrt(np.diag(Sigma))

            # Calculate the raw weights, inversely proportional to volatility
            raw_weights = inv_vol / np.sum(inv_vol)

            # Prepare the portfolio dictionary
            portfolio = {k: None for k in information_set['companies']}

            # Assign values to the portfolio, normalized weights
            for i, company in enumerate(information_set['companies']):
                portfolio[company] = raw_weights[i]

            return portfolio
        except Exception as e:
            logging.warning("Error computing portfolio, returning equal weight portfolio")
            logging.warning(e)
            return {k: 1/len(information_set['companies']) for k in information_set['companies']}
    
def compute_information(self, t: datetime):
    data = self.slice_data(t)
    information_set = {}
    data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()
    information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()
    
    # Covariance Calculation could involve more detailed filtering depending on the strategy
    data = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column)
    data = data.dropna(axis=0)
    
    # Might include additional filtering based on volatility or other factors
    information_set['covariance_matrix'] = data.cov().to_numpy()
    information_set['companies'] = data.columns.to_numpy()
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
            
            # Initial weights
            x0 = np.ones(n) / n
            
            # Minimize the objective function
            res = minimize(obj, x0, constraints=cons, bounds=bounds)

            # Prepare output portfolio
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
        # Get the sliced data for time t
        data = self.slice_data(t)
        information_set = {}

        # Sort data by ticker and date
        data = data.sort_values(by=[self.company_column, self.time_column])
        
        # Calculate daily returns using percentage change
        data['return'] = data.groupby(self.company_column)[self.adj_close_column].pct_change()
        
        # Compute expected return for each company: mean of daily returns
        information_set['expected_return'] = data.groupby(self.company_column)['return'].mean().to_numpy()

        # Creating the covariance matrix
        data = data.pivot(index=self.time_column, columns=self.company_column, values=self.adj_close_column)
        
        # Drop any rows with missing values
        data = data.dropna(axis=0)

        # Compute the covariance matrix from the adjusted close prices
        covariance_matrix = data.pct_change().cov().to_numpy()  # Calculate covariance from percentage changes
        
        # Add the computed covariance matrix to the information set
        information_set['covariance_matrix'] = covariance_matrix
        information_set['companies'] = data.columns.to_numpy()
        
        return information_set
        
    
        
    
