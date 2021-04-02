**EDA**

For this particular assignment, I initially started by visualizing and understanding the dataset. Firstly, I checked for any missing values and noticed that 2 records contained missing values. I replaced them with 0 because there weren’t very many of them, and because the primary goal is to build a predictive model rather than perform data analysis (if that was the case I would have probably replaced with the mean of the column instead of the 0).
The second thing I noticed was that there was no variable provided to flag if a customer actually is considered churned. For the purpose of this assignment, I assumed that customers pay month-to-month and that if a customer does not have any future activity records after month X, that means they have “churned” on month X.
Lastly, I looked to see if there was any degree of multicollinearity between the variables. I tried both a correlation matrix and calculating variance inflation factor (VIF). Neither of those results suggested any glaring issues with the original variable set, so I did not feel the need to do feature engineering or to drop any of these variables from model consideration.

**Data Wrangling**

I needed to create a variable to measure tenure, which is the difference between the max and min of the billing segment id for each user. I aggregated data up to the user level as the model input instead of having it at the user-month level because it was just simpler.
I also ended up standardizing the variables as well so that the different ranges in magnitude did not skew the model in an unintended way.

**Model Selection**

Given the scope of this assignment, I decided to use a simple-to-interpret model to try and predict churn. I chose Logistic Regression for this reason, even though it does have quite a few assumptions (variables are independent, little/no multicollinearity, large sample size, etc.). I did consider trying tree-based models, but because I did not have the dependent variable explicitly provided and the number of assumptions I had to make, I didn’t want the model to be too prone to overfitting.
Model Interpretation
Looking at the coefficients, we see that:
•	Longer tenure = decreased likelihood to churn
•	Usage of product A or product B  = slightly decreased likelihood to churn
•	Ticket creation = slightly increased likelihood to churn

**Model Validation**

I initially trained the model on a simple 80-20 split (80% train, 20% test) and the accuracy score was reasonably high (~78%). Trying a confusion matrix to assess the number of false positives and false negatives seem to suggest a similar conclusion.
I decided to try cross validation (stratified k-fold where k=5) but still got similar results. Logistic Regression did output the probability as well for each prediction in the test set, which can be used to stratify users based on the “degree” of risk (i.e. low-medium-high). This way we can prioritize and action on the users who have the highest probability of churning.

**Additional data/modifications to improve**

I would have really liked to have more insight into how the business is set up, in terms of how customers are billed (month to month?), how to determine whether a customer has churned, and what product A and product B are. 
For additional data points, I would also like to know the length to resolution for tickets created by users, as well as how quickly bugs are resolved.

**Recommendations and Interpretations**

Since longer tenure in users mean less likely to churn, we should prioritize our focus on new users and how we can make their experience better. This could mean better tutorials or support in the onboarding process, or a regular email series to “welcome” them to the product.
It seems like users who have used either product A and B are less likely to churn. How do we incentivize users to use these products? Are they free products or do they cost money to the user? If they are premium products that cost money, perhaps offer them to the user as a free trial.
Lastly, there needs to be an increased focus on addressing customer tickets, especially for users who have a high probability of churning. These tickets should be prioritized by the customer service team.
