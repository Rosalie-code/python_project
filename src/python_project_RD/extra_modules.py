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
#   - Risk Parity from python_project_RD
#   - Minimum Variance POrtfolio from python_project_RD
# The user choice is asked by the function strategy_choice in user_function.py file.


# Class that represents the data used in the backtest.   

class RiskParity(Information):
    def compute_portfolio(self, t: datetime, information_set):
        """Calculate asset allocation based on volatility.

        Parameters
        ----------
        t : datetime
            The timestamp for portfolio computation.
        information_set : dict
            Dictionary containing 'covariance_matrix' (2D array-like) and 'companies' (list of company names).

        Returns
        -------
        dict
            A portfolio dictionary with company names as keys and their weights as values.
            Returns equal weight portfolio if information is invalid.

        Examples
        --------
        >>> portfolio = compute_portfolio(datetime.now(), information_set)
        """

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
        """Create an equal weight portfolio.

        Parameters
        ----------
        information_set : dict
            Dictionary containing 'companies' (list of company names).

        Returns
        -------
        dict
            A portfolio with equal weights assigned to each company.
            Returns an empty dictionary if no companies are provided.
        """
        if not information_set or 'companies' not in information_set:
            logging.warning("Information set is missing companies. Defaulting to empty portfolio.")
            return {}

        num_companies = len(information_set['companies'])
        return {k: 1/num_companies for k in information_set['companies']}

    def compute_information(self, t: datetime):
        """Compute information set for portfolio calculation.

        Parameters
        ----------
        t : datetime
            The timestamp for fetching relevant data.

        Returns
        -------
        dict
            A dictionary containing 'covariance_matrix' and 'expected_return'.
        """
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
        """Compute the minimum variance portfolio.

        This method optimizes the asset weights to minimize portfolio variance, subject to the constraint
        that the sum of the weights equals 1, and no short selling is allowed.

        Parameters
        ----------
        t : datetime
            The timestamp for the portfolio computation.
        information_set : dict
            Dictionary containing 'covariance_matrix' (2D array-like) and 'companies' (list of company names).

        Returns
        -------
        dict
            A portfolio dictionary with company names as keys and their weights as values.
            Returns equal weight portfolio if there is an error during optimization or if information is invalid.
        """
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
        """Compute the information set for portfolio optimization.

        This method retrieves historical data, computes the expected returns and the covariance matrix
        of the asset returns, and prepares the information set for portfolio calculations.

        Parameters
        ----------
        t : datetime
            The timestamp for fetching relevant data.

        Returns
        -------
        dict
            A dictionary containing 'covariance_matrix' and 'expected_return' of the assets 
            as well as their respective names.
        """
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
        
    
        
    
