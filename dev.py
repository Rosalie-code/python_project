from src.python_project_RD.extra_broker import Backtest
from pybacktestchain.broker import StopLoss
from pybacktestchain.blockchain import load_blockchain
from src.python_project_RD.user_function import get_initial_parameter, strategy_choice, ask_user_for_comment

def algo_backtest():
    # Set verbosity for logging
    verbose = False  # Set to True to enable logging, or False to suppress it

    initial_cash, stop_loss_threshold, start_date, end_date = get_initial_parameter()
    strategy, strategy_name = strategy_choice()
    ask_user_for_comment()


    backtest = Backtest(initial_date=start_date,
            final_date=end_date,
            initial_cash=initial_cash,
            threshold = stop_loss_threshold,
            information_class=strategy,
            strategy_name=strategy_name,
            risk_model=StopLoss,
            name_blockchain='backtest',
            verbose=verbose
            )
    backtest.run_backtest()


    block_chain = load_blockchain('backtest')
    print(str(block_chain))
    # check if the blockchain is valid
    print("Blockchain valid:", block_chain.is_valid())

if __name__ == "__main__":
    algo_backtest()
