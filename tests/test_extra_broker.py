import pytest
import pandas as pd
from datetime import datetime
from python_project.extra_broker import AnalysisTool, CustomBroker

import numpy as np

#test for custom class Broker
def test_execute_portfolio():
    initial_cash = 1000
    broker = CustomBroker(cash=initial_cash, verbose=False)
    portfolio = {"AAPL": 0.5, "GOOGL": 0.5,}
    prices = {"AAPL": 100,"GOOGL": 200,}
 
    broker.buy(ticker="AAPL", quantity=5, price=100, date=datetime(2024, 1, 1)) 
    assert broker.get_cash_balance() == 500, "Cash balance should reflect the purchase"

    initial_value = broker.get_portfolio_value(prices)
    total_value_after_execution, nb_sell, nb_buy = broker.execute_portfolio(portfolio, prices, date=datetime(2024, 1, 2))
    
    # Assertions
    assert total_value_after_execution > initial_value, "Total value after execution should be greater than initial value."
    assert nb_buy == 1
    assert nb_sell == 0
    
    assert broker.get_cash_balance() < initial_cash, "Cash balance should decrease after buying shares."

    assert broker.positions["AAPL"].quantity == 5, "AAPL quantity should be 5."
    assert broker.positions["GOOGL"].quantity == 0, "GOOGL quantity should be 0 since we didn't buy any."

    print("All assertions passed for execute_portfolio!")



#test for class AnalysisTool
def test_total_performance():
    portfolio_values = [100, 105, 102, 108, 107, 110]
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    
    analysis_tool = AnalysisTool(portfolio_values, initial_value, final_value)
    assert np.isclose(analysis_tool.total_performance(), (final_value - initial_value) / initial_value), "Total Performance does not match."

def test_annualized_performance():
    portfolio_values = [100, 105, 102, 108, 107, 110]
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]

    analysis_tool = AnalysisTool(portfolio_values, initial_value, final_value)
    num_periods = len(portfolio_values)
    expected_annualized_performance = (final_value / initial_value) ** (1 / num_periods) - 1
    assert np.isclose(analysis_tool.annualized_performance(), expected_annualized_performance), "Annualized Performance does not match."

def test_mean_returns():
    portfolio_values = [100, 105, 102, 108, 107, 110]
    
    analysis_tool = AnalysisTool(portfolio_values, portfolio_values[0], portfolio_values[-1])
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    expected_mean_returns = np.mean(returns)
    assert np.isclose(analysis_tool.mean_returns(), expected_mean_returns), "Mean of the Returns does not match."

def test_volatility_returns():
    portfolio_values = [100, 105, 102, 108, 107, 110]
    
    analysis_tool = AnalysisTool(portfolio_values, portfolio_values[0], portfolio_values[-1])
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    expected_volatility = np.std(returns)
    assert np.isclose(analysis_tool.volatility_returns(), expected_volatility), "Volatility of the Returns does not match."

def test_maximum_drawdown():
    portfolio_values = [100, 105, 102, 108, 107, 110]
    
    analysis_tool = AnalysisTool(portfolio_values, portfolio_values[0], portfolio_values[-1])
    expected_drawdown = np.min((portfolio_values - np.maximum.accumulate(portfolio_values)) / np.maximum.accumulate(portfolio_values))
    assert np.isclose(analysis_tool.maximum_drawdown(), expected_drawdown), "Maximum Drawdown does not match."

def test_sharpe_ratio():
    portfolio_values = [100, 105, 102, 108, 107, 110]
    
    analysis_tool = AnalysisTool(portfolio_values, portfolio_values[0], portfolio_values[-1], risk_free_rate=0.01)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    mean_returns = np.mean(returns)
    volatility = np.std(returns)
    expected_sharpe_ratio = (mean_returns - 0.01) / volatility if volatility > 0 else 0
    assert np.isclose(analysis_tool.sharpe_ratio(), expected_sharpe_ratio), "Sharpe Ratio does not match."


# Run all test functions
def run_tests():
    test_total_performance()
    test_annualized_performance()
    test_mean_returns()
    test_volatility_returns()
    test_maximum_drawdown()
    test_sharpe_ratio()
    test_execute_portfolio()
    print("All tests passed!")

run_tests()