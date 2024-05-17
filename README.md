# Automovie Loan Default Prediction

## Overview

This repository contains the analysis and model for predicting vehicle loan defaults for a non-banking financial institution (NBFI). The goal is to determine whether a client will default on their vehicle loan payment based on their data.

## Dataset Description

A non-banking financial institution (NBFI) or non-bank financial company (NBFC) is a financial institution that does not have a full banking license or is not supervised by a national or international banking regulatory agency. NBFC facilitates bank-related financial services, such as investment, risk pooling, contractual savings, and market brokering.

An NBFI is struggling to mark profits due to an increase in defaults in the vehicle loan category. The company aims to determine the client’s loan repayment abilities and understand the relative importance of each parameter contributing to a borrower’s ability to repay the loan.

## Goal

The goal of this problem is to predict whether a client will default on the vehicle loan payment or not. For each ID in the Test_Dataset, we predict the “Default” level.

## Datasets

The problem contains two datasets:

Train_Dataset: Used for building the model.

Test_Dataset: Used for testing the model and generating predictions.

The submission output from the Test_Dataset is to be provided as a CSV file.

## Evaluation Metric

The performance of the model is evaluated using the F1_Score, which is the harmonic mean of Recall and Precision. More details on F1_Score can be found here.

## Submission File Format

The submission file should be a CSV file with exactly 80,900 entries plus a header row. The file should have exactly two columns:

## ID (sorted in any order)

Default (contains 0 & 1, where 1 represents a default)
Repository Contents
Automovie Loan Default Dataset.ipynb: Jupyter notebook containing the data analysis, preprocessing, model building, and evaluation.

submission.csv: CSV file with the resultant predictions for the Test_Dataset.

## Dataset origin
https://www.kaggle.com/datasets/saurabhbagchi/dish-network-hackathon/data
