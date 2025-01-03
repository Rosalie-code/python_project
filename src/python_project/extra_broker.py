import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass
from datetime import datetime

import os 
import pickle
from pybacktestchain.data_module import UNIVERSE_SEC, get_stocks_data, DataModule, Information
from pybacktestchain.broker import Position, StopLoss, RebalanceFlag, Broker
from pybacktestchain.utils import generate_random_name
from pybacktestchain.blockchain import Block, Blockchain
from numba import jit 

from user_function import strategy_choice, get_initial_parameter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datetime import timedelta, datetime

#---------------------------------------------------------
# Classes
#---------------------------------------------------------



class CustomBroker(Broker):   
    def execute_portfolio(self, portfolio: dict, prices: dict, date: datetime):

        nb_buy = 0 #modification
        nb_sell = 0  #modification    
        """Executes the trades for the portfolio based on the generated weights."""
        # First, handle all the sell orders to free up cash
        for ticker, weight in portfolio.items():
            price = prices.get(ticker)
            if price is None:
                if self.verbose:
                    logging.warning(f"Price for {ticker} not available on {date}")
                continue
            
            total_value = self.get_portfolio_value(prices)
            target_value = total_value * weight
            current_value = self.positions.get(ticker, Position(ticker, 0, 0)).quantity * price
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)
    

            if quantity_to_trade < 0:
                self.sell(ticker, abs(quantity_to_trade), price, date)
                nb_sell = nb_sell +1 #modification
        
        # Then, handle all the buy orders, checking if there's enough cash
        for ticker, weight in portfolio.items():
            price = prices.get(ticker)
            if price is None:
                if self.verbose:
                    logging.warning(f"Price for {ticker} not available on {date}")
                continue
      
            total_value = self.get_portfolio_value(prices)
            target_value = total_value * weight
            current_value = self.positions.get(ticker, Position(ticker, 0, 0)).quantity * price
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)

            
            if quantity_to_trade > 0:
                available_cash = self.get_cash_balance()
                cost = quantity_to_trade * price
                
                if cost <= available_cash:
                    self.buy(ticker, quantity_to_trade, price, date)
                    nb_buy = nb_buy +1 #modification
                else:
                    if self.verbose:
                        logging.warning(f"Not enough cash to buy {quantity_to_trade} of {ticker} on {date}. Needed: {cost}, Available: {available_cash}")
                        logging.info(f"Buying as many shares of {ticker} as possible with available cash.")
                    quantity_to_trade = int(available_cash / price)
                    self.buy(ticker, quantity_to_trade, price, date)
        total_value_after_execution = self.get_portfolio_value(prices) 
        return total_value_after_execution, nb_sell, nb_buy


@dataclass
class AnalysisTool:
    def __init__(self, portfolio_values, initial_value, final_value, risk_free_rate=0.0):
        self.portfolio_values = np.array(portfolio_values)
        self.initial_value = initial_value
        self.final_value = final_value
        self.risk_free_rate = risk_free_rate

    def total_performance(self):
        return (self.final_value - self.initial_value) / self.initial_value

    def annualized_performance(self):
        num_periods = len(self.portfolio_values) 
        return (self.final_value / self.initial_value) ** (1 / num_periods) - 1

    def calculate_returns(self):
        return np.diff(self.portfolio_values) / self.portfolio_values[:-1]
    
    def mean_returns(self):
        returns = self.calculate_returns()
        return np.mean(returns)
    
    def volatility_returns(self):
        returns = self.calculate_returns()
        return np.std(returns)

    def maximum_drawdown(self):
        cumulative_max = np.maximum.accumulate(self.portfolio_values)
        drawdowns = (self.portfolio_values - cumulative_max) / cumulative_max
        return drawdowns.min()

    def sharpe_ratio(self):
        returns = self.calculate_returns()
        excess_returns = returns - self.risk_free_rate
        return np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0

    def analyze(self):
        """Portfolio Analysis Summary:"""
        return {
            "Portfolio Total Performance": self.total_performance(),
            "Portfolio Annualized Performance": self.annualized_performance(),
            "Mean of the Returns": self.mean_returns(),
            "Volatility of the Returns": self.volatility_returns(),
            "Maximum Drawdown": self.maximum_drawdown(),
            "Sharpe Ratio": self.sharpe_ratio(),  
        }


@dataclass
class Backtest:
    initial_date: datetime
    final_date: datetime
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'CSCO', 'NFLX']
    information_class : type  = Information
    s: timedelta = timedelta(days=360)
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column : str ='Adj Close'
    rebalance_flag_class : type = RebalanceFlag
    rebalance_flag: RebalanceFlag = None
    risk_model : type = StopLoss
    initial_cash: int = 1000000  # Initial cash in the portfolio
    name_blockchain: str = 'backtest'
    verbose: bool = True
    broker = None
  
    def __post_init__(self):
        self.backtest_name = generate_random_name()
        self.initial_cash, self.threshold, self.initial_date,self.final_date  = get_initial_parameter()
        self.broker = CustomBroker(cash=self.initial_cash, verbose=self.verbose)
        self.broker.initialize_blockchain(self.name_blockchain)
                # Get initial cash and other parameters from user
        self.risk_model_strategy = strategy_choice()  # Store the chosen strategy class for later use
        self.information_class = self.risk_model_strategy 
        self.rebalance_flag = self.rebalance_flag_class()


    def run_backtest(self):

        strategy_name = strategy_choice()[1]

        evolution_nb_sell = []
        evolution_nb_buy = []
        evolution_portfolio_value = []
        evolution_time = []
    

        logging.info(f"Running backtest from {self.initial_date} to {self.final_date}.")
        logging.info(f"Retrieving price data for universe")
        self.risk_model = self.risk_model(self.threshold)
        # self.initial_date to yyyy-mm-dd format
        init_ = self.initial_date.strftime('%Y-%m-%d')
        # self.final_date to yyyy-mm-dd format
        final_ = self.final_date.strftime('%Y-%m-%d')
        df = get_stocks_data(self.universe, init_, final_)

        # Initialize the DataModule
        data_module = DataModule(df)

        # Create the Information object
        info = self.information_class(s = self.s, 
                                    data_module = data_module,
                                    time_column=self.time_column,
                                    company_column=self.company_column,
                                    adj_close_column=self.adj_close_column)
        
        # Run the backtest
        for t in pd.date_range(start=self.initial_date, end=self.final_date, freq='D'):
            if self.risk_model is not None:
                portfolio = info.compute_portfolio(t, info.compute_information(t))
                prices = info.get_prices(t)
                self.risk_model.trigger_stop_loss(t, portfolio, prices, self.broker)
           
            if self.rebalance_flag().time_to_rebalance(t):
                logging.info("-----------------------------------")
                logging.info(f"Rebalancing portfolio at {t}")
                information_set = info.compute_information(t)
                portfolio = info.compute_portfolio(t, information_set)
                prices = info.get_prices(t)

                value_portfolio_after_execution, nb_sell, nb_buy = self.broker.execute_portfolio(portfolio, prices, t)
                evolution_portfolio_value.append(value_portfolio_after_execution)
                evolution_nb_buy.append(nb_buy)
                evolution_nb_sell.append(nb_sell)
                evolution_time.append(t)

        initial_value = self.initial_cash
        final_value = self.broker.get_portfolio_value(info.get_prices(self.final_date))
        analysis_tool = AnalysisTool(evolution_portfolio_value, initial_value, final_value)
        analysis_results = analysis_tool.analyze()
        logging.info(f"Analysis results: {analysis_results}")

        with open(f"backtests_analysis/statistics/analysis_results_{self.backtest_name}({strategy_name}).txt", 'w') as file:
            for key, value in analysis_results.items():
                file.write(f"{key}: {value}\n")

        logging.info(f"Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(info.get_prices(self.final_date))}")
        df = self.broker.get_transaction_log()

        # create backtests folder if it does not exist
        if not os.path.exists('backtests'):
            os.makedirs('backtests')

        # save to csv, use the backtest name 
        df.to_csv(f"backtests_analysis/backtests/baktest_{self.backtest_name}({strategy_name}).csv")

        # store the backtest in the blockchain
        self.broker.blockchain.add_block(self.backtest_name, df.to_string())
    
        # Plotting the portfolio value evolution
        plt.figure(figsize=(10, 6))  
        plt.plot(evolution_time, evolution_portfolio_value, label="Portfolio Value", color='blue', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.title(f"Portfolio Value Evolution from {self.initial_date} to {self.final_date} with {strategy_name}")
        plt.grid(True)
        plt.xticks(rotation=45)  
        plt.legend()
        plt.tight_layout()  
  
        plt.savefig(f"backtests_analysis/graphs/Portfolio_Value_Evolution_with_baktest_{self.backtest_name}({strategy_name})", dpi=900)
        plt.show()

        # Optionally, plot the number of buys and sells over time
        plt.figure(figsize=(10, 6))  
        plt.plot(evolution_time, evolution_nb_buy, label="Number of Buys", color='green', marker='o')
        plt.plot(evolution_time, evolution_nb_sell, label="Number of Sells", color='red', marker='x')
        plt.xlabel('Time')
        plt.ylabel('Number of Trades')
        plt.title(f"Buy and Sell Evolution from {self.initial_date} to {self.final_date} with {strategy_name}")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"backtests_analysis/graphs/Number_Buy_Sell_with_baktest_{self.backtest_name}({strategy_name})", dpi=900)
        plt.show()