---
layout: post
title: Predicting Customer Signup Rates Using Classification techniques
image: "/posts/ofm-new-customer-campaign.jpg"
tags: [Python, Machine Learning, Decision Tree Classification, Random Forest Classification, Logistic Regression, K-Nearest-Neighbours (KNN) for Classification]
---

In this post I'm going to be using the machine learning algorithms Logistic Regression, Decision Tree Classification and Random Forest Classification and K-Nearest-Neighbours (KNN) for Classification to find an accurate model to predict customer signup rates following a marketing campaign. This campaign has run before but the business want to target mailers at customers likely to sign up rather than all customers to minimise cost. Therefore, we need to identify those customers that are likely to sign up based on the customer data available.

---

First I will create a dataset that is at customer level and customer loyalty score as well as key features I believe will help predict the customer loyalty score.

---

First I will import the required packages. I have chosen pickle as it can save and load objects specific to Python.

```ruby
import pandas as pd
import pickle
```
---

Next I will import the data, which is on a number of sheets in a single excel file.

```ruby
loyalty_scores = pd.read_excel("data/grocery_database.xlsx", sheet_name = "loyalty_scores")
customer_details = pd.read_excel("data/grocery_database.xlsx", sheet_name = "customer_details")
transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions")
```
