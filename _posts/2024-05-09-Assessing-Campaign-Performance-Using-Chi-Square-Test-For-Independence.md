---
layout: post
title: Assessing Campaign Performance Using Chi-Square Test For Independence
image: "/posts/primes_image.jpeg"
tags: [Python, Chi-Square, p-value, acceptance criteria, critical value, null hypothesis, alternate hypothesis ]
---

In this post I'm going to use AB Testing to assess whether the signup rate increases for promotional content using a nicer looking mailer over a basic and cheaper version.

---

First I will import the required packages. I will be using chi_2_contingency to compute chi-square statistic and p-value and chi2 to calculate the critical value

```ruby
import pandas as pd
from scipy.stats import chi2_contingency, chi2
```
---

Next I will import the dataset with the campaign data information and filter the results such that the only observations are that of those with either of the two different mailer types

```ruby
campaign_data = pd.read_excel("grocery_database.xlsx", sheet_name="campaign_data")
campaign_data = campaign_data.loc[campaign_data["mailer_type"] != "Control"]
```

For a chi-square test of independence I need to create 2x2 matrix with observed frequencies of the type of mailer and the those that signed up or not for the promotion.
To do this I will use the crosstab function in Python and use 'values' to return an array.

```ruby
observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values
>>>[252 123]
   [209 127]

```

Using the observed_values I can now calculate the signup rate for each mailer type. Intially it seems as though there is a relationship between the mailer type and signup rate.

```ruby
mailer1_signup_rate = 123 / (252 + 123) #32.8%
mailer2_signup_rate = 127 / (209 + 127) #37.8%
print(mailer1_signup_rate, mailer2_signup_rate)
>>> 0.328 0.37797619047619047
```

The next step is to state my hypotheses and acceptance criteria. 

```ruby
null_hypothesis = "There is no relationship between mailer type and sign up rate. They are independent"
alternate_hypothesis = "There is a relationship between mailer type and sign up rate. They are not independent"
acceptance_criteria = 0.05
```
I can now use the observed frequencies to calculate the Chi-Square statistic, p-value, degrees of freedom (dof) and expected values by passing through observed_values through chi2_contingency.

```ruby
chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction = False)   #use correction=False as dof=1
print(chi2_statistic, p_value)
>>> 1.9414468614812481 0.16351152223398197
```

I can now calculate the critical value as I can pass through the acceptance criteria and dof calculated above through chi2.

```ruby
critical_value = chi2.ppf(1 - acceptance_criteria, dof)
print(critical_value)
>>> 3.841458820694124
```

Now that we have all our values we can compare to determine whether the signup rate was indeed higher using the more expensive mailer type. There are two approaches we can take.

The first is to compare the chi-square statistic against the critical-value. If the chi-square statistic is greater than the critical value we can reject the null hypothesis. The reason for this being that the probability of the null hypothesis occuring is 5%. From the results we can see that it is lower and we can retail the thr null hypothesis and there is no relationship between the mailer type and sign up rate.

```ruby
if chi2_statistic >= critical_value:
    print(f"As our chi-square statistic of {chi2_statistic} is higher than out critical value of {critical_value} - we reject the null hypothesis, and conclude that: {alternate_hypothesis}")
else:
    print(f"As our chi-square statistic of {chi2_statistic} is lower than out critical value of {critical_value} - we retain the null hypothesis, and conclude that: {null_hypothesis}")
>>> As our chi-square statistic of 1.9414468614812481 is lower than out critical value of 3.841458820694124 - we retain the null hypothesis, and conclude that: There is no relationship between mailer type and sign up rate. They are independent
```

We can also use the p-value and compare against the acceptance criteria to confirm our findings. If the p-value is less than the acceptance criteria then we can reject the null hypothesis as the probability of event happening is very unlikely. The p-value is higher therefore we will accept our null hypothesis.

```ruby
if p_value <= acceptance_criteria:
    print(f"As our p-value of {p_value} is lower than out acceptance criteria of {acceptance_criteria} - we reject the null hypothesis, and conclude that: {alternate_hypothesis}")
else:
    print(f"As our p-value of {p_value} is higher than out acceptance criteria of {acceptance_criteria} - we retain the null hypothesis, and conclude that: {null_hypothesis}")
>>> As our p-value of 0.16351152223398197 is higher than out acceptance criteria of 0.05 - we retain the null hypothesis, and conclude that: There is no relationship between mailer type and sign up rate. They are independent
```

As we can see there is no relationship between the mailer type and signup rate. Whilst intially we can see there is a difference as the cheaper version led to 32.8% signup rate and for the nicer version it was 37.8%. However, this difference is not statistically significant.

This test is important as it allows organisations to compare different strategies and see if one approach is having a positive impact before spending money on an approach that likely is not going to improve outcomes.
