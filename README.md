# Project: python_project_RD

## Purposes
Improve the package pybacktestchain by:
1) **Adding new asset allocation strategies (extra_modules.py)**:
    - Added `RiskParity` class as a subclass to the `Information` class from pybacktestchain,
    - Added `MinimumVariancePortfolio`  class as a subclass to the `Information` class from pybacktestchain.
2) **Extending the analysis for the user (extra_broker.py)**:
    - Created the `CustomBroker` subclass from the `Broker` class of pybacktestchain in order to modify the `execute_portfolio` function,
    - Developed the `AnalysisTool` class, which computes different statistics for portfolio analysis,
    - Modified `Backtest` class so that after running the `run_backtest()` function,  CSV, PNG, and TXT files are saved in the  `\python_project_RD\backtest_analysis` directory, providing users with additional analysis on the backtest.
3) **Being participative and user-friendly (user_function.py)**:
    - Created `strategy_choice()` function to allow users to choose a strategy for the backtest among several options,
    - Created `ask_user_for_comment()` function enabling users to leave a comment on the project's discussions page on GitHub,
    - Created `get_initial_parameter()` function for users to select parameters for running the backtest, such as the initial cash, the stop loss threshold, the start date and the end date of the backtest.
    
Overall, this package enables users to create tailored and bespoke backtests on Equity and provides tools for comparing different backtests.


## Installation

```bash
$ pip install python_project_RD
```


## Usage

```python

from src.python_project_RD.extra_broker import Backtest
from pybacktestchain.broker import StopLoss
from src.python_project_RD.user_function import get_initial_parameter, strategy_choice, ask_user_for_comment

# Enable or disable logging verbosity
verbose = False  # Set to True to enable logging, or False to suppress it


initial_cash, stop_loss_threshold, start_date, end_date = get_initial_parameter()# Gather initial parameters from the user
strategy, strategy_name = strategy_choice()# Allow the user to choose a trading strategy and store the selected strategy name
ask_user_for_comment() # Prompt the user for additional comments regarding the backtest setup

# Create a customized backtest instance with user-defined parameters
backtest = Backtest(
    initial_date=start_date,          
    final_date=end_date,              
    initial_cash=initial_cash,        
    threshold=stop_loss_threshold,     
    information_class=strategy,       
    strategy_name=strategy_name,      
    risk_model=StopLoss,               
    name_blockchain='backtest',        
    verbose=verbose                 
)

# Execute the backtest with the specified parameters
backtest.run_backtest()

```


## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`python_project_RD` was created by Rosalie Dechance. This same package is based on the `pybacktestchain` package created by by Juan F. Imbet as part of a project for the course Python Programming for Finance at Paris Dauphine University - PSL. 


It is licensed under the terms of the MIT license.

## Credits

`python_project_RD` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).