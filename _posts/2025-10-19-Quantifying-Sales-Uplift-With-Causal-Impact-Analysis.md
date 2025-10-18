---
layout: post
title: Quantifying Sales Uplift With Causal Impact Analysis
image: "/posts/causal_impact_analysis.png"
tags: [Python, Machine Learning, Causal Impact Analysis]
---

In this post I'm going to be quantifying sales uplift following a marketing campaign with causal impact analysis.

---
First I will import the required packages.

```ruby
from causalimpact import CausalImpact
import pandas as pd
```
---
Next I will import the data. As we are trying to quantify the sales uplift following marketing campaign I will need to transactions and campaign data.

```ruby
transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions")
campaign_data = pd.read_excel("data/grocery_database.xlsx", sheet_name = "campaign_data")
```
The transacions data is at item level. Therefore I will aggregate the data at customer and date level using the sales cost. This would provide me with the total sales each day for each customer.

```ruby
customer_daily_sales = transactions.groupby(["customer_id", "transaction_date"])["sales_cost"].sum().reset_index()
```
---
I will now merge this dataset with the campaign data.

```ruby
customer_daily_sales = pd.merge(customer_daily_sales, campaign_data, how = "inner", on = "customer_id")
```
---
For the analysis I need the total sales each day by whether someone had signed up for promotion or not. To do this I will pivot the data and aggrgate sales by sign up group

```ruby
causal_impact_df = customer_daily_sales.pivot_table(index = "transaction_date",
                                                   columns = "signup_flag",
                                                   values = "sales_cost",
                                                   aggfunc = "mean")
```
---
For the analysis I need the total sales each day by whether someone had signed up for promotion or not. To do this I will pivot the data and aggrgate sales by sign up group

```ruby
causal_impact_df = customer_daily_sales.pivot_table(index = "transaction_date",
                                                   columns = "signup_flag",
                                                   values = "sales_cost",
                                                   aggfunc = "mean")
```
---
To avoid warning messages I will set the frequency to daily as our data is sales by day as shown below.
```ruby
causal_impact_df.index.freq = "D"
```
---
For the analysis the impacted group needs to be in the first column so the line of code below will order them accordingly. I also renamed the columns so each group is easily identifiable.
```ruby
causal_impact_df = causal_impact_df[[1,0]]
causal_impact_df.columns = ["member", "non_member"]
```
---
Next I will apply the causal impact. As part of this I have specified the pre-period, which is before promotional event and the post period following the promotional event.
```ruby
pre_period = ["2020-04-01", "2020-06-30"]
post_period = ["2020-07-01", "2020-09-30"]

ci = CausalImpact(causal_impact_df, pre_period, post_period)
```
---
The impact was plotted as follows.
```ruby
ci.plot()
```
The plot showed a positive uplift in sales following the promotional event as the difference between actual and counterfactual was largely positive. This can be further checked with figures by extracting the summary as shown below.

```ruby
print(ci.summary())
print(ci.summary(output = "report"))
>>> Actual                    171.33
>>> Absolute effect (s.d.)    49.92 (4.23)
>>> Prediction (s.d.)         121.42
>>> Relative effect (s.d.)    41.11%
```
This summary above shows campaign had a 41% increase in sales following the promotion.
