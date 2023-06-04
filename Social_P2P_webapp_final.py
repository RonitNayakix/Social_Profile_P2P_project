# -*- coding: utf-8 -*-
"""
Created on Sun May 21 09:25:24 2023

@author: ronit
"""
import pickle
import streamlit as st
import pandas as pd
import numpy as np

def Application():
    st.title('Social Profile LoanStatus Prediction App by Classification💵')

    model1 = pickle.load(open('model_ensemble.pkl', 'rb'))
    model_EMI = pickle.load(open('model_lr_EMI.pkl', 'rb'))
    model_PROI = pickle.load(open('model_lr_PROI.pkl', 'rb'))
    model_ELA = pickle.load(open('model_lr_ELA.pkl', 'rb'))

    def main():
        with st.form('prediction_form'):
            st.header("Predict the input for the following features:")
            st.header("")
            st.header("![Alt Text](https://i.imgur.com/71KxTWy.jpg)")

            BorrowerAPR = st.number_input('BorrowerAPR:', min_value=0)
            BorrowerRate = st.number_input('BorrowerRate:', min_value=0)
            LenderYield = st.number_input('LenderYield:', min_value=0)
            LoanCurrentDaysDelinquent = st.number_input('LoanCurrentDaysDelinquent:', min_value=0)
            LoanFirstDefaultedCycleNumber = st.number_input('LoanFirstDefaultedCycleNumber:', min_value=0)
            LP_CustomerPayments = st.number_input('LP_CustomerPayments:', min_value=0)
            LP_CustomerPrincipalPayments = st.number_input('LP_CustomerPrincipalPayments:', min_value=0)
            LP_GrossPrincipalLoss = st.number_input('LP_GrossPrincipalLoss:', min_value=0)
            LP_NetPrincipalLoss = st.number_input('LP_NetPrincipalLoss:', min_value=0)
            LP_NonPrincipalRecoverypayments = st.number_input('LP_NonPrincipalRecoverypayments:', min_value=0)
            LoanOriginalAmount = st.number_input('LoanOriginalAmount:', min_value=0)
            MonthlyLoanPayment = st.number_input('MonthlyLoanPayment:', min_value=0)
            LP_InterestandFees = st.number_input('LP_InterestandFees:', min_value=0)
            LP_ServiceFees = st.number_input('LP_ServiceFees:', min_value=0)
            LoanTenure = st.number_input('LoanTenure:', min_value=0)
            InterestAmount = st.number_input('InterestAmount:', min_value=0)
            TotalAmount = st.number_input('TotalAmount:', min_value=0)
            CreditScoreRangeLower = st.number_input('CreditScoreRangeLower:', min_value=0)
            CreditScoreRangeUpper = st.number_input('CreditScoreRangeUpper:', min_value=0)
            TotalInquiries = st.number_input('TotalInquiries:', min_value=0)
            AvailableBankcardCredit = st.number_input('AvailableBankcardCredit:', min_value=0)
            ROI = st.number_input('ROI:', min_value=0)
            CreditGrade = st.number_input('CreditGrade:', min_value=0)
            CreditGrade_description = st.number_input('CreditGrade_description:', min_value=0)

            button1 = st.form_submit_button('Predict')

        if button1:
            data1 = np.array([BorrowerAPR, BorrowerRate, LenderYield, LoanCurrentDaysDelinquent, LoanFirstDefaultedCycleNumber,
                             LP_CustomerPayments, LP_CustomerPrincipalPayments, LP_GrossPrincipalLoss, LP_NetPrincipalLoss,
                             LP_NonPrincipalRecoverypayments]).reshape(1, -1)

            prediction1 = model1.predict(data1)
            if prediction1 == [0]:
                st.write('Loan is not approved')
            else:
                st.write('Loan is approved')

            st.write(f"The predicted LoanStatus is: {prediction1}")
            
            data2= np.array([BorrowerRate,LoanOriginalAmount,MonthlyLoanPayment,LP_CustomerPayments, 
                            LP_CustomerPrincipalPayments,LP_InterestandFees,LP_ServiceFees,LoanTenure, 
                            InterestAmount,TotalAmount]).reshape(1, -1)

            predictionEMI = model_EMI.predict(data2)
            st.write(f"EMI: {predictionEMI}")

            data3= np.array([BorrowerAPR,BorrowerRate,LenderYield,CreditScoreRangeLower,CreditScoreRangeUpper,
                    TotalInquiries, AvailableBankcardCredit, ROI, 
                    CreditGrade,CreditGrade_description]).reshape(1, -1)
                
            predictionPROI = model_PROI.predict(data3)
            st.write(f"PROI: {predictionPROI}")
            
            data4= np.array([LoanOriginalAmount,MonthlyLoanPayment,LP_CustomerPayments, 
                            LP_CustomerPrincipalPayments, LP_InterestandFees, LP_ServiceFees, 
                            LoanTenure, InterestAmount, TotalAmount, ROI]).reshape(1, -1)

            predictionELA = model_ELA.predict(data4)
            st.write(f"ELA: {predictionELA}")

    main()

if __name__ == '__main__':
    Application()
