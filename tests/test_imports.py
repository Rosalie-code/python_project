# content of test_python_project.py
import pytest
from unittest.mock import patch

def test_data_import():
    with patch('builtins.input', side_effect=['1']):  # Simulate input '1'
        from src.python_project.user_function import strategy_choice
        strategy, strategy_name = strategy_choice()
        assert strategy is not None
    
def test_broker_import():
    from src.python_project.extra_broker import Backtest, AnalysisTool, CustomBroker
    assert Backtest is not None
    assert AnalysisTool is not None
    assert CustomBroker is not None
