---
layout: post
title: Predicting Customer Loyalty Using Regression techniques
image: "/posts/supermarket_loyalty_image.jpg"
tags: [Python, Machine Learning, Decision Tree, Random Forest, Regression]
---

In this post I'm going to be using machine learning algorithms Linear Regression, Decision Trees and Random Forest Trees to find an accurate model to predict customer loyalty scores based on the information we know about the customer.

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

The three different data sets will need to be combined to produce a single customer level dataset. The customer_details will be merged with loyalty_scores so that we can assign loyalty scores that we know of to the corresponding customers. The transaction data is will be grouped by customer and corresponding transaction data will be aggregated so that there is a single row of transactional data per customer. This will then be merged to the customer_details and loyalty_scores.

```ruby
data_for_regression = pd.merge(customer_details, loyalty_scores, how = "left", on = "customer_id")
sales_summary = transactions.groupby("customer_id").agg({"sales_cost" : "sum",
                                                         "num_items" : "sum",
                                                         "transaction_id" : "count",
                                                         "product_area_id" : "nunique"}).reset_index()
sales_summary.columns = ["customer_id", "total_sales", "total_items", "transaction_count", "product_area_count"]
sales_summary["average_basket_value"] = sales_summary["total_sales"] / sales_summary["transaction_count"]
data_for_regression = pd.merge(data_for_regression, sales_summary, how = "inner", on = "customer_id")
```

The single dataset data_for_regression will now be split into those with loyalty scores and those without. The models will be built on the former and if accurate used on the latter.

```ruby
regression_modelling = data_for_regression.loc[data_for_regression["customer_loyalty_score"].notna()]
regression_scoring = data_for_regression.loc[data_for_regression["customer_loyalty_score"].isna()]
regression_scoring.drop(["customer_loyalty_score"], axis = 1, inplace = True)
```

Next I will save the objects using pickle so that when building models they can be imported and used directly.

```ruby
pickle.dump(regression_modelling, open("data/abc_regression_modelling.p", "wb"))    
pickle.dump(regression_scoring, open("data/abc_regression_scoring.p", "wb"))
```
The first model I will build and assess is a Linear Regression model using the following packages.

```ruby
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV
```
Next I imported the customer dataset created earlier. As customer_id is not an input variable and will have no impact in predicting the customer loyalty score this will be dropped.

```ruby
data_for_model = pickle.load(open("data/abc_regression_modelling.p", "rb"))
data_for_model.drop("customer_id", axis=1, inplace=True)
```
I will also shuffle the data to ensure there is no ordering, which may or may not have an impact on the model.

```ruby
data_for_model = shuffle(data_for_model)
```
Next I will be going through a series of data preparation steps, beginning with missing values. These can be identified as follows.

```ruby
data_for_model.isna().sum()
>>> distance_from_store       2
>>> gender                    3
>>> credit_score              2
```
As there are a low number of missing values I will not use imputation and drop the rows with missing data.

```ruby
data_for_model.dropna(how="any", inplace=True)
```
Linear regression model can be affected significantly with outliers. We can get a high level overview using the following code.

```ruby
outlier_investigation = data_for_model.describe()
```
The table shows some extreme values in distance_from_store, total_sales and total_items. So using the boxplot approach these will be removed. The code below informs us of how many were removed.

```ruby
for column in outlier_columns:
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2  # Changed to 2 to stop eliminating too many
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace = True)
>>> 2 outliers detected in column distance_from_store
>>> 23 outliers detected in column total_sales
>>> 0 outliers detected in column total_items
```
The final preparation step is to split the data into the input variables and output variable.

```ruby
X = data_for_model.drop(["customer_loyalty_score"], axis=1)
y = data_for_model["customer_loyalty_score"]
```
The X and y datasets can now be split into training and test sets. In order for our model to learn we need as much training data as possible. Therefore, I have used a 80-20 split.

```ruby
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
```
The data has a categorical data input variable called gender. I will use One Hot Encoding to deal with this variable so it uses binary option. You will note that I have dropped the first encoded column as two will be produced to avoid the dummy trap variable. I have also ensured the encoding rules learnt on the training data are applied to the test data rather than learned on each new set of data. This ensures the rules will be the same rather than different each time. The code below will replace the gender column with gender_m having a binary value of 0 or 1.

```ruby
categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse_output=False, drop = "first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])  
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(categorical_vars, axis=1, inplace=True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)
X_test.drop(categorical_vars, axis=1, inplace=True)
```
Next I will apply Feature Selection by using Recursive Feature Elimination with Cross Validation to find the optimal number of inputs for the training the model. The code below will find the optimal number and drop the rest. In this particular situation the process selected all 8 input variables.

```ruby
regressor = LinearRegression()
feature_selector = RFECV(regressor)

fit = feature_selector.fit(X_train,y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]


plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()
>>> Optimal number of features: 8
```
Now we are ready to train the model, which can be done using the following.

```ruby
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```
We are now ready to assess the model by using it on the test set to predict customer loyalty scores. 

```ruby
# Predict on Test Set
y_pred = regressor.predict(X_test)
```
These predictions can then be assessed against the actual customer loyalty scores by calculating r2

```ruby
r_squared = r2_score(y_test, y_pred)
print(r_squared)
>>> 0.7805702910327409
```
Whilst the model score looks promising it is useful to apply cross validation technique. This is to ensure the model is generalising well and not overfitting. 

```ruby
cv_object = KFold(n_splits = 4, shuffle = True)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv_object, scoring = "r2")
cv_scores.mean()
>>> 0.8532327536589753
```
This higher value probably shows how initial test/train split was not entriely representative of the data.

At this stage it is also worth calculating the adjusted r2 score. This gives a fairer representation as for every input variable added to the model will increase the r2 score. The adjusted score takes this into account. As you can see it is slightly lower than r2, which is to be expected but still a good score.

```ruby
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 -(1-r_squared) * (num_data_points -1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
>>> 0.7535635576213859
```
Finally I will calculate the coefficients for each input variable, which is essentially its weighting.

```ruby
coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names,coefficients], axis=1)
summary_stats.columns = ["input_variable", "coefficient"]
>>> 	distance_from_store	-0.20123171509921695
```
This shows that the distance from store has a significant impact. It shows for every extra mile the customer is further away the loyalty score will reduce by 0.2 (20%). This makes sense as the further they are away from this store would imply they are closer to another one and do most shopping there.

I have also calculated the intercept value for the model, which would complete the information for the plane of best fit along with the coefficients above.

```ruby
regressor.intercept_
>>> 0.5160974174646146
```


I will be using Random Forests for Regression to predict loyalty scores. The following packages will be imported to build the model.

```ruby
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
```

Next I imported the customer dataset created earlier.

```ruby
data_for_model = pickle.load(open("data/abc_regression_modelling.p", "rb"))
data_for_model.drop("customer_id", axis=1, inplace=True)
```
The data was shuffled in case it is ordered and has an impact on the model  

```ruby
data_for_model = shuffle(data_for_model)
```
The dataset was then split into input and output variable(s). As we are predicitng the customer loyalty score this is our output variable and is represented by y. All other fields will form part of the input variables and are represented by X.

```ruby
X = data_for_model.drop(["customer_loyalty_score"], axis=1)
y = data_for_model["customer_loyalty_score"]
```

The datasets are then split into training and test sets where the test size represents the percentage of data for test set. In this case there is a 80-20 split between training and test.

```ruby
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
```
Random Forest typically requires numerical input data. One hot encoding will be used to handle one of the input variables (gender) which is categorical. This method will allocate a numerical value depending on whether the customer is male (1) or female (0) and drop the gender column. Note that there is not a separate column where female is represented by 1 and male by 0. This has been dropped to eliminate issue of multi-collinearity.

```ruby
categorical_vars = ["gender"]

one_hot_encoder = OneHotEncoder(sparse_output=False, drop = "first") 

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)   #axis=1 means we are looking at columns
X_train.drop(categorical_vars, axis=1, inplace=True) #Drops Categorical variable columns

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)   #axis=1 means we are looking at columns
X_test.drop(categorical_vars, axis=1, inplace=True) #Drops Categorical variable columns
```
Next I will train the model

```ruby
regressor = RandomForestRegressor(random_state = 42)
regressor.fit(X_train, y_train)
```
We can now predict on the test set

```ruby
y_pred = regressor.predict(X_test)
```
Using the output test and predictions I will determine the r2 value
```ruby
r_squared = r2_score(y_test, y_pred)
print(r_squared)
>>> 0.9598617481534647
```
To check the accuracy of the r2 value I ran a cross validation technique and this showed the scores were not too disimilar.

```ruby
cv_object = KFold(n_splits = 4, shuffle = True)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv_object, scoring = "r2")
cv_scores.mean()
>>> 0.9248592219288347
```
As we are using multiple input variables I calculated the adjusted r2 score. This compensates for the addition of input variables and only increases if the variable improves the model above what would be obtained by probability.

```ruby
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 -(1-r_squared) * (num_data_points -1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
>>> As our adjusted r2 score of 0.9552745193710035 shows that our model if fit for purpose.
```
Next I checked the impact of each input variable on our model

```ruby
feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names,feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable", "feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)

plt.barh(feature_importance_summary["input_variable"],feature_importance_summary["feature_importance"])
plt.title("Feature importance of Random Forest")
plt.xlabel("Feature Importance")
plt.show()
>>> The image shows that distance from store is the most important feature for predicting loyalty scores. <img width="939" height="563" alt="image" src="https://github.com/user-attachments/assets/3ffdbfc4-3df5-4e01-b8c8-3d2f82550358" />
```
Next I used the permuation importance technique, which measures how much a model's performance metric decreases when the values of a specific feature are randomly shuffled. This disruption breaks the relationship between the feature and the target variable, allowing us to assess the feature's contribution to the model's predictive power.

```ruby
result = permutation_importance(regressor, X_test, y_test, n_repeats = 10, random_state = 42)
print(result)

permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names,permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable", "permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.show()
>>> The plot shows a slight difference where total_sales is considered to have a bit more of an impact but verifies that distance from store is the most significant metric in predicting loyalty score. In fact this technique has show it is more impactful. <img width="936" height="563" alt="image" src="https://github.com/user-attachments/assets/f4e38b36-eeb3-40aa-a9d6-17f5e6621888" />
```
The model and objects were then saved to use later for predicting loyalty scores.

```ruby
pickle.dump(regressor, open("data/random_forest_regression_model.p", "wb"))
pickle.dump(one_hot_encoder, open("data/random_forest_regression_ohe.p", "wb"))
```
Now with the model trained we are ready to predict customer loyalty scores. The dataset was first imported as well as our model and model objects.

```ruby
to_be_scored = pickle.load(open("data/abc_regression_scoring.p", "rb"))
regressor = pickle.load(open("data/random_forest_regression_model.p", "rb"))
one_hot_encoder = pickle.load(open("data/random_forest_regression_ohe.p", "rb"))
```
Any unused columns and missing data was dropped.

```ruby
to_be_scored.drop(["customer_id"], axis = 1, inplace = True)
to_be_scored.dropna(how = "any", inplace = True)
```
One hot encoding was applied to the dataset

```ruby
categorical_vars = ["gender"]
encoder_vars_array = one_hot_encoder.transform(to_be_scored[categorical_vars])
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)
encoder_vars_df = pd.DataFrame(encoder_vars_array, columns = encoder_feature_names)
to_be_scored = pd.concat([to_be_scored.reset_index(drop=True), encoder_vars_df.reset_index(drop=True)], axis=1)
to_be_scored.drop(categorical_vars, axis=1, inplace=True)
```
Finally the model was used on our dataset to make predictions.

```ruby
loyalty_predictions = regressor.predict(to_be_scored)
```
These scores could now be tagged to the respective customer id's and back into the database for either marketing purposes or just to gain an understanding of the customers.
