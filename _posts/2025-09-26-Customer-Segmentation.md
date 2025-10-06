---
layout: post
title: The "You Are What You Eat" Customer Segmentation
image: "/posts/Customer-Segmentation-cover.jpg"
tags: [Python, Machine Learning, Unsupervised learning, K-Means Clustering]
---

In this post I'm going to be using the machine learning algorithm K-Means Clustering to segment customers based on product areas they are shopping in. In particular there will be a focus on grocery products. 

---

First I will import the required packages for the K-Means Clustering model.

```ruby
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
```
---

Next I will import the datasets with the relevant data. As there is a focus on grocery types I will need to highlight the product areas and in order to understand the spending habits I will also need the transactions data.
```ruby
transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name = "transactions")
product_areas = pd.read_excel("data/grocery_database.xlsx", sheet_name = "product_areas")
```
---
I will now merge the datasets to get the product type mapped to the transaction. This will be an inner join on product_area_id.
```ruby
transactions = pd.merge(transactions, product_areas, how = "inner", on = "product_area_id")
```
---
As the focus is on grocery types I will drop the 'non-food' transactions.
```ruby
transactions.drop(transactions[transactions["product_area_name"] == "Non-Food"].index, inplace = True)
```
---
Next I will need to aggregate the data so it is at customer level by product area type. This will allow us to see the levels of expenditure in the different grocery areas by each customer.
```ruby
transaction_summary = transactions.groupby(["customer_id", "product_area_name"])["sales_cost"].sum().reset_index()
```
---
We now have multiple rows for each customer with each row representing the different type of area they did their grocery shopping and the corresponding transactions. I will now use the pivot table function to transpose the data so that there is a single row per customer and the different product areas are represented by columns.
```ruby
transaction_summary_pivot = transactions.pivot_table(index = "customer_id",
                                                    columns = "product_area_name",
                                                    values = "sales_cost",      
                                                    aggfunc = "sum",            
                                                    fill_value = 0,             
                                                    margins = True,
                                                    margins_name = "Total").rename_axis(None,axis = 1)
```
---
Next I will turn the transaction values into percentages so it is interpret the transactions values.
```ruby
transaction_summary_pivot = transaction_summary_pivot.div(transaction_summary_pivot["Total"], axis = 0)
```
---
Now that these are represented by percentages the total value of sales can be dropped.
```ruby
data_for_clustering = transaction_summary_pivot.drop(["Total"], axis = 1)
```
---
Next I will go through a couple of data cleaning and preparation steps. Firstly I will check for any missing values.
```ruby
data_for_clustering.isna().sum()
>>> Dairy         0
>>> Fruit         0
>>> Meat          0
>>> Vegetables    0
```
---
As there are no missing values no further action is required. Next I will look at scaling. As this algorithm uses distance between centroids and data points it is vital the data is scaled. Although the data is already between 0 and 1 I will still normalise as one product area might be more dominant and I want to be sure all categories are spread proportionately.
```ruby
scale_norm = MinMaxScaler()
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering), columns = data_for_clustering.columns)
```
---
Before I begin the modelling I need to know the 'k' (number of clusters) value parameter I need to set. I will use the WCSS approach to find the ideal 'k' values.
```ruby
k_values = list(range(1,10))
wcss_list = []

for k in k_values:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(data_for_clustering_scaled)
    wcss_list.append(kmeans.inertia_)
    
plt.plot(k_values, wcss_list)
plt.title("Witin Clustet Sum of Squares - by k")
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.tight_layout()
plt.show()
```
---
Looking at the plot we can see that after 3 clusters the line begins to tail off. So I will use this value as a parameter to instantiate and fit the model.
```ruby
kmeans = KMeans(n_clusters = 3, random_state = 42, n_init = 10)
kmeans.fit(data_for_clustering_scaled)
```
---
I will now add the cluster labels to our data.
```ruby
data_for_clustering["cluster"] = kmeans.labels_
```
---
Checking the spread of data we can see that there is bias towards one cluster.
```ruby
data_for_clustering["cluster"].value_counts()
```
---
Finally I will profile to better understand the groupings. 
```ruby
cluster_summary = data_for_clustering.groupby("cluster")[["Dairy", "Fruit", "Meat", "Vegetables"]].mean().reset_index()
>>>	cluster	Dairy	Fruit	Meat	Vegetables
>>>   0	    0.363	0.394	0.029	0.212
>>>	  1	    0.220	0.264	0.376	0.138
>>>   2	    0.002	0.637	0.003	0.356
```
Looking at the clusters it looks like cluster 0 are probably vegetarian as there is a very low meat but a high dairy, fruit and vegetables percentage. Cluster 1 are probably omnivores with a good spread across all categories. Cluster 3 could be vegan or probably prefer to only buy fruit and vegetables at this store.

This information could allow the business to target individual offers at customers and as well as begin to understand there shopping habits.
