import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass
from datetime import datetime

import os 
from pybacktestchain.data_module import UNIVERSE_SEC, get_stocks_data, DataModule, Information
from pybacktestchain.broker import Position, StopLoss, RebalanceFlag, Broker
from pybacktestchain.utils import generate_random_name
from numba import jit 


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datetime import timedelta, datetime

#---------------------------------------------------------
# Classes
#---------------------------------------------------------

class CustomBroker(Broker): 
# We modify the execute_portfolio function from pybacktestchain.broker to improve it: we add 2 counters to count the number of time we sell and buy. 
# The function returns the number of time we sell and buy and the total value of the portfolio.
    
    def execute_portfolio(self, portfolio: dict, prices: dict, date: datetime):
        """Executes the trades for the portfolio based on the generated weights."""
        
        nb_buy = 0  # Initialize buy counter
        nb_sell = 0  # Initialize sell counter

        # 1) Handle sell orders first to free up cash:
        for ticker, weight in portfolio.items():
            price = prices.get(ticker)
            if price is None:
                if self.verbose:
                    logging.warning(f"Price for {ticker} not available on {date}")
                continue
            
            total_value = self.get_portfolio_value(prices)
            target_value = total_value * weight
            current_value = self.positions.get(ticker, Position(ticker, 0, 0)).quantity * price

            # Calculate the difference and determine quantity to trade
            diff_value = target_value - current_value
            quantity_to_trade = int(diff_value / price)

            # Execute selling
            if quantity_to_trade < 0:
                self.sell(ticker, abs(quantity_to_trade), price, date)
                nb_sell += 1  # Increment the sell counter

        # 2) Handle buy orders:
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
        
            # Execute buying
            if quantity_to_trade > 0:
                available_cash = self.get_cash_balance()
                cost = quantity_to_trade * price
                
                if cost <= available_cash:
                    self.buy(ticker, quantity_to_trade, price, date)
                    nb_buy += 1  # Increment the buy counter
                else:
                    # Handle insufficient cash case
                    if self.verbose:
                        logging.warning(f"Not enough cash to buy {quantity_to_trade} of {ticker} on {date}. "
                                        f"Needed: {cost}, Available: {available_cash}")
                    
                    # Buy as many shares as possible with available cash
                    quantity_to_trade = int(available_cash / price)
                    if quantity_to_trade > 0:  # Ensure we can actually buy
                        self.buy(ticker, quantity_to_trade, price, date)
                        nb_buy += 1  # Increment the buy counter

        # 3) Calculate total portfolio value after all trades:
        total_value_after_execution = self.get_portfolio_value(prices)

        return total_value_after_execution, nb_sell, nb_buy


#Creation of a new class that computes different statistics to analyse the portfolio. The class includes the below functions:
#   - Computation of the total performance and annualized performance of the strategy
#   - Calculation or returns
#   - Computation the mean and the volatilty of the returns
#   - Computation the maximum drawdown
#   - Computation the sharpe ratio

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

#Modification of the Backtest class:
#   - A graph plotting the evolution of the value of the portfolio is saved in the folder \python_project\backtest_analysis\graphs
#   - A graph plotting the number of buy and sell over time is save in the folder \python_project\backtest_analysis\graphs
#   - A summary of the strategy analysis is saved in the folder \python_project\backtest_analysis\statistics
#   - A backtest is saved in the folder \python_project\backtest_analysis\backtests

@dataclass
class Backtest:
    initial_date: datetime
    final_date: datetime
    strategy_name : str
    initial_cash: int = 1000000  # Default initial cash  
    threshold: float = 0.1  
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'CSCO', 'NFLX']
    information_class : type  = Information
    s: timedelta = timedelta(days=360)
    time_column: str = 'Date'
    company_column: str = 'ticker'
    adj_close_column : str ='Adj Close'
    rebalance_flag_class : type = RebalanceFlag
    rebalance_flag: RebalanceFlag = None
    risk_model : type = StopLoss
    name_blockchain: str = 'backtest'
    verbose: bool = True
    broker = None
  
    def __post_init__(self):
        self.backtest_name = generate_random_name()
        self.broker = CustomBroker(cash=self.initial_cash, verbose=self.verbose)
        self.broker.initialize_blockchain(self.name_blockchain)
        self.rebalance_flag = self.rebalance_flag_class()
        pass

    def run_backtest(self):

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
        logging.info(f"Retrieved data:\n{df}") #modification

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
           
            if self.rebalance_flag.time_to_rebalance(t):
                logging.info("-----------------------------------")
                logging.info(f"Rebalancing portfolio at {t}")
                information_set = info.compute_information(t)
                print(information_set)
                portfolio = info.compute_portfolio(t, information_set)
                print(portfolio)
                prices = info.get_prices(t)

                value_portfolio_after_execution, nb_sell, nb_buy = self.broker.execute_portfolio(portfolio, prices, t)
                evolution_portfolio_value.append(value_portfolio_after_execution)
                logging.info(f"Date: {t}, Portfolio: {portfolio}, Prices: {prices}")
                evolution_nb_buy.append(nb_buy)
                evolution_nb_sell.append(nb_sell)
                evolution_time.append(t)

        print(f"CHECKING EVOLUTION OF PORTFOLIO TO IF RUNNING: {evolution_portfolio_value}")

        initial_value = self.initial_cash
        final_value = self.broker.get_portfolio_value(info.get_prices(self.final_date))
        analysis_tool = AnalysisTool(evolution_portfolio_value, initial_value, final_value)
        analysis_results = analysis_tool.analyze()
        logging.info(f"Analysis results: {analysis_results}")

        with open(f"backtests_analysis/statistics/analysis_results_{self.backtest_name}({self.strategy_name}).txt", 'w') as file:
            for key, value in analysis_results.items():
                file.write(f"{key}: {value}\n")

        logging.info(f"Backtest completed. Final portfolio value: {self.broker.get_portfolio_value(info.get_prices(self.final_date))}")
        df = self.broker.get_transaction_log()

        # create backtests folder if it does not exist
        if not os.path.exists('backtests'):
            os.makedirs('backtests')

        # save to csv, use the backtest name 
        df.to_csv(f"backtests_analysis/backtests/baktest_{self.backtest_name}({self.strategy_name}).csv")

        # store the backtest in the blockchain
        self.broker.blockchain.add_block(self.backtest_name, df.to_string())
    
        # Plotting the portfolio value evolution
        plt.figure(figsize=(10, 6))  
        plt.plot(evolution_time, evolution_portfolio_value, label="Portfolio Value", color='blue', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.title(f"Portfolio Value Evolution from {self.initial_date} to {self.final_date} with {self.strategy_name}")
        plt.grid(True)
        plt.xticks(rotation=45)  
        plt.legend()
        plt.tight_layout()  
  
        plt.savefig(f"backtests_analysis/graphs/Portfolio_Value_Evolution_with_baktest_{self.backtest_name}({self.strategy_name})", dpi=900)
        plt.show()

        # Optionally, plot the number of buys and sells over time
        plt.figure(figsize=(10, 6))  
        plt.plot(evolution_time, evolution_nb_buy, label="Number of Buys", color='green', marker='o')
        plt.plot(evolution_time, evolution_nb_sell, label="Number of Sells", color='red', marker='x')
        plt.xlabel('Time')
        plt.ylabel('Number of Trades')
        plt.title(f"Buy and Sell Evolution from {self.initial_date} to {self.final_date} with {self.strategy_name}")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"backtests_analysis/graphs/Number_Buy_Sell_with_baktest_{self.backtest_name}({self.strategy_name})", dpi=900)
        plt.show()
