# content of test_python_project.py
def test_data_import():
    from python_project.user_function import strategy_choice
    strategy,  = strategy_choice()
    assert strategy is not None
    
def test_broker_import():
    from python_project.extra_broker import Backtest, AnalysisTool, CustomBroker
    assert Backtest is not None
    assert AnalysisTool is not None
    assert CustomBroker is not None