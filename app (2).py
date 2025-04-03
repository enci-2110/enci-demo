
import streamlit as st
import pandas as pd
import pickle
from pyngrok import ngrok

# Load the pipeline from the pickle file
with open('/content/xgb_gridsearch.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

# Set up the Streamlit app title
st.title("Loan Approval Prediction App")

# Create input fields for user to enter data
id = st.number_input("ID", min_value=1, value=1)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
annual_income = st.number_input("Annual Income", min_value=0, value=50000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
loan_duration = st.number_input("Loan Duration (months)", min_value=1, value=12)
experience = st.number_input("Experience", min_value=0, value=5)
number_of_dependents = st.number_input("Number of Dependents", min_value=0, value=2)
monthly_debt_payments = st.number_input("Monthly Debt Payments", min_value=0, value=1000)
credit_card_utlization_rate = st.number_input("Credit Card Utilization Rate", min_value=0.0, max_value=1.0, value=0.5)
number_of_open_credit_lines = st.number_input("Number of Open Credit Lines", min_value=0, value=5)
number_of_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=3)
debt_to_income_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.3)
bankruptcy_history = st.number_input("Bankruptcy History", min_value=0, value=1)
previous_loan_default = st.number_input("Previous Loan Default", min_value=0, value=1)
payment_history = st.number_input("Payment History", min_value=0, value=3)
length_of_credit_history = st.number_input("Length of Credit History (years)", min_value=0, value=5)
saving_account_balance = st.number_input("Saving Account Balance", min_value=0, value=10000)
checking_account_balance = st.number_input("Checking Account Balance", min_value=0, value=5000)
total_assets = st.number_input("Total Assets", min_value=0, value=100000)
total_liabilities = st.number_input("Total Liabilities", min_value=0, value=50000)
monthly_income = st.number_input("Monthly Income", min_value=0, value=1000)
utility_bills_payment_history = st.number_input("Utility Bills Payment History", min_value=0, value=3)
job_tenure = st.number_input("Job Tenure (years)", min_value=0, value=5)
net_worth = st.number_input("Net Worth", min_value=0, value=500000)
base_interest_rate = st.number_input("Base Interest Rate", min_value=0.0, max_value=1.0, value=0.05)
interest_rate = st.number_input("Interest Rate", min_value=0.0, max_value=1.0, value=0.05)
monthly_loan_payment = st.number_input("Monthly Loan Payment", min_value=0, value=1000)
total_debt_to_income_ratio = st.number_input("Total Debt to Income Ratio", min_value=0.0, max_value=1.0, value=0.3)

# Create a button to trigger prediction
if st.button("Predict"):
    # Create a DataFrame from user input
    new_data = pd.DataFrame({
        'ID': [id],
        'Age': [age],
        'AnnualIncome': [annual_income],
        'CreditScore': [credit_score],
        'LoanAmount': [loan_amount],
        'LoanDuration': [loan_duration],
        'Experience': [experience],
        'NumberOfDependents' : [number_of_dependents],
        'MonthlyDebtPayments': [monthly_debt_payments],
        'CreditCardUtilizationRate': [credit_card_utlization_rate],
        'NumberOfOpenCreditLines': [number_of_open_credit_lines],
        'NumberOfCreditInquiries': [number_of_credit_inquiries],
        'DebtToIncomeRatio': [debt_to_income_ratio],
        'BankruptcyHistory': [bankruptcy_history],
        'PreviousLoanDefault': [previous_loan_default],
        'PaymentHistory': [payment_history],
        'LengthOfCreditHistory': [length_of_credit_history],
        'SavingAccountBalance': [saving_account_balance],
        'CheckingAccountBalance': [checking_account_balance],
        'TotalAssets': [total_assets],
        'TotalLiabilities': [total_liabilities],
        'MonthlyIncome': [monthly_income],
        'UtilityBillsPaymentHistory': [utility_bills_payment_history],
        'JobTenure': [job_tenure],
        'NetWorth': [net_worth],
        'BaseInterestRate': [base_interest_rate],
        'InterestRate': [interest_rate],
        'MonthlyLoanPayment': [monthly_loan_payment],
        'TotalDebtToIncomeRatio': [total_debt_to_income_ratio]

    })


    # Make prediction using the loaded pipeline
    prediction = loaded_pipeline.predict(new_data)

    # Display the prediction
    if prediction[0] == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Rejected.")
