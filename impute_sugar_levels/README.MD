**Goal**

The goal of this project is to predict the sugar level column from a dataset of USDA foods that contain complete nutritional information from over 7000 foods. We will be using the rest of the column as independent variables.

**EDA**

Here we attempt to understand our data by analyzing the distributions of the numerical features, the number of distinct values in the categorical features, and looking for signs of multicollinearity.

**Data Wrangling**

Given the number of features, we want to do some form of feature selection by assessing which features are most strongly correlated with the dependent variable. We are also dropping categorical features because there are too many unique categories to create dummy variables for each one.
We will also want to clean up any missing records and remove glaring outliers.

**Model Selection**

The first model attempted was the most simple one, being linear regression. This was a rather bad fit. Next I tried applying a Polynomial transformation to the features in case there is an underlying non-linear pattern. The issue with higher degrees is that there is a lot of variance between the accuracy of predicting on training set and on the test set.
The polynomial regression model actually performed worse than linear regression on test data. The last attempt was a non-parametric supervised learning model, which was Decision Tree Regression. This one fared the best out of the 3 models.

**Model Evaluation**

I decided to skip k-folds or other more indepth evaluation methods for this project and just stuck with the 80-20 train/test set. The metric used to evaluate the models were RMSE and R-squared. 
RMSE is the square root of the variance of the residuals, which measures absolute fit. Lower values of RMSE indicate better fit, however this metric is harder to interpret.
R-squared a measure of goodness of fit, and is more intuitive in that it ranges from 0 to 1. Generally speaking, the closer it is to 1, the better the fit. However for my polynomial regression model, I was getting a really bizzare R-squared value, which means it might not be the right metric for it...

**Next Steps**

Overall, the Tree Regression model was able to provide the best RMSE/R-squared results on the test set, and the features ranked by importance makes sense intuitively. I would however like to test the tree based model on the entire feature set and not just the subset of 15 that were selected earlier on in the process.
We can let the model decide on the best features and just control for overfitting by putting controlling the tree size and depth.
