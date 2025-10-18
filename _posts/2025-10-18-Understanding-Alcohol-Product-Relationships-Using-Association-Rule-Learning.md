---
layout: post
title: Understanding Alcohol Product Relationships Using Association Rule Learning
image: "/posts/apriori.png"
tags: [Python, Machine Learning, Associated Rule Learning, Apriori algorithm]
---

In this post I'm going to be using Assosciated Rule Learning with the Apriori algorithm to determine the strength of relatioships between different alcohol products.

---
First I will import the required packages.

```ruby
from apyori import apriori
import pandas as pd
```
---
Next I will import the sample data, which contains the different alcohol products purchased each transaction. After importing I will drop the ID column from the dataset as it is not needed.

```ruby
alcohol_transactions = pd.read_csv("data/sample_data_apriori.csv")
alcohol_transactions.drop("transaction_id", axis=1, inplace=True)
```
---
Before applying the Apriori algorithm the dataset needs to be converted into a master list with a series of lists that represent each transaction. As part of creating this master list I will also drop empty fields.

```ruby
transactions_list = []

for index, row in alcohol_transactions.iterrows():
    transaction = list(row.dropna())
    transactions_list.append(transaction)
```
---
Next I will apply the Apriori algorithm. Using the apriori function we will pass in a number of parameters. For he purposes of this example I am focusing on relationships between two items and setting a length of 2.

```ruby
apriori_rules = apriori(transactions_list,
                        min_support = 0.003,
                        min_confidence = 0.2,
                        min_lift = 3,
                        min_length = 2,
                        max_length = 2)
```

---
Due to the nature of the output I will convert results into a pandas dataframe so we can easily interpret the results.

```ruby
product1 = [list(rule[2][0][0])[0] for rule in apriori_rules]
product2 = [list(rule[2][0][1])[0] for rule in apriori_rules]
support = [rule[1] for rule in apriori_rules]
confidence = [rule[2][0][2] for rule in apriori_rules]
lift = [rule[2][0][3] for rule in apriori_rules]

apriori_rules_df = pd.DataFrame({"product1" : product1,
                                 "product2" : product2,
                                 "support" : support,
                                 "confidence" : confidence,
                                 "lift" : lift})
```
---
As the results are in no particular order I will sort them by the descending lift score. This is because the lift score  indicates the strength of two items being bought together more often than expected.
```ruby
apriori_rules_df.sort_values(by = "lift", ascending = False, inplace = True)
```
---
We can see from the output there are some interesting results. Wine and Beer gift sets tend to be purchased together. So a store may consider putting them next to each other rather than in their respective categories. There also appears to be a relationship between small size of wines as well as boxes of red and white wines. This information is valuable in understanding which items can be put for promotion as well as placement in store.
However, it is important to note that items that have a high lift score but a low support score should be approached with caution. This might indicate a relationship by chance rather than there actually being one. Such parameters can be adjusted when applying the Apriori algorithm.
