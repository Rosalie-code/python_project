import webbrowser
from datetime import datetime


def strategy_choice():
    print("Which method would you like to choose?")
    print("1 - First Two Moment")
    print("2 - Risk Parity")
    print("3 - Minimum Variance Portfolio")
    
    strategy = None
    while strategy is None:
        choice = input("Please enter the number of your choice (1, 2, or 3): ")
        
        if choice == '1':
            from pybacktestchain.data_module import FirstTwoMoments
            strategy = FirstTwoMoments
            strategy_name =  "First Two Moment asset allocation strategy"
        elif choice == '2':
            from extra_modules import RiskParity 
            strategy =  RiskParity
            strategy_name =  "Risk Parity asset allocation strategy"
        elif choice == '3':
            from extra_modules import MinimumVariancePortfolio 
            strategy = MinimumVariancePortfolio
            strategy_name =  "Minimum Variance Portfolio asset allocation strategy" 
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            strategy_choice()  # Restart the function for a valid input
    print(f"You chosed {strategy_name}")
    return strategy, strategy_name



def ask_user_for_comment():
    # Ask the user if they want to leave a comment
    choice = input("The aim of this package is to continually improve in order to adapt as closely as possible to users' needs. Would you like to comment on Github which strategy you would like to be developed? ? (yes/no): ").strip().lower()

    if choice == 'yes':
        # Redirect to GitHub Discussions page with URL
        print("Great! Please visit the GitHub discussions page to leave your comment.")
        discussion_url = "https://github.com/Rosalie-code/python_project/discussions"
        webbrowser.open(discussion_url)  # Open the Discussions page in the default web browser
    elif choice == 'no':
        print("Okay, no comment will be left.")
    else:
        print("Invalid input. Please respond with 'yes' or 'no'.")


def get_initial_parameter():
    while True:
        try:
            initial_cash = int(input("Please enter your initial investment amount: "))
            stop_loss_threshold = float(input("Please enter your Stop Loss threshold in decimal (ie. for 10% threshold, enter 0,1):"))
            if stop_loss_threshold > 1:
                print("Invalid choice. Please enter decimal number")
                continue

            start_date_str = input("Please enter the start date (YYY_MM_DD):")
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

            end_date_str = input("Please enter the start date (YYY_MM_DD):")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

            if end_date <= start_date:
                print("The end date must be after the start date")
                continue
            
            return initial_cash, stop_loss_threshold, start_date, end_date
    
        except ValueError:
            print("Invalid input. Investment and threshold has to be numeric values and dates has to be in the format YYY-MM-DD")
        