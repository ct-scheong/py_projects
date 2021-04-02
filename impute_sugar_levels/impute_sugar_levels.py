#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Title: Food Data Analysis
# Author: Shery Cheong
# Description: Imputate missing sugar levels, rank fruit sweetness
# Date: 03/31/21
# Version: 1.0

######################################################################################################################################################
##### Housekeeping #####
######################################################################################################################################################

# Import all needed packages
import csv
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# Suppress scientific notation
pd.options.display.float_format = '{:.2f}'.format

# Don't truncate display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print ("Imported libraries - success")


# In[2]:


######################################################################################################################################################
##### Data Input #####
######################################################################################################################################################

# Read in excel input (header is 2nd row)
df = pd.read_excel('2017-2018 FNDDS At A Glance - FNDDS Nutrient Values.xlsx', header=1,sheet_name='FNDDS Nutrient Values',engine='openpyxl')
print ('Dataframe head')
display (df.head())
print('\n')


# In[3]:


######################################################################################################################################################
##### EDA Part 1 #####
######################################################################################################################################################

# Check column types
print ('Dataframe column types')
df.info()
print('\n')

# Check data size
print ('Dataframe size')
print(df.shape)
print('\n')

# Check uniqueness of each column
print ('Column Uniqueness')
print(df.nunique(axis=0))
print('\n')

# Check for null/missing values
print ('Dataframe missing values across all columns')
print (df.isnull().sum().sum())
print('\n')


# In[4]:


######################################################################################################################################################
##### Feature Selection #####
######################################################################################################################################################

# There are too many features to do more in-depth EDA, so we would like to trim down the number

# Define x and y variables
y = df['Sugars, total\n(g)']
X = df.drop(['Sugars, total\n(g)'],axis=1)

# There are too many unique categorical variables to convert them all to dummy variables, so we should eliminate them from X
# We will also eliminate Food code, which is just the unique identifier for each food
X = X.drop(['Main food description','WWEIA Category description','Food code'], 1)

# Select top features that have strongest relationship with Y variable using SelectKBest. Let's say k=15 (this is arbitrary but can probably be tuned)
# We plan to use this in a regression model, so let's set score function to be f_regression
best_features = SelectKBest(score_func=f_regression,k=15)
fit = best_features.fit(X,y)

# Extract out the scores and columns
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)

# Concatenate both dataframes
feature_scores = pd.concat([df_columns, df_scores],axis=1)
feature_scores.columns = ['Feature_Name','Score']  # name output columns
print(feature_scores.nlargest(15,'Score'))  # print best features

# Define new X set
X_reduced = X[(feature_scores.nlargest(15,'Score'))['Feature_Name']]

# Check distribution of data for each independent var in X reduced
print ('Column Data Distribution')
print(X_reduced.describe().apply(lambda s: s.apply(lambda x: format(x, 'f'))))
print('\n')


# In[5]:


######################################################################################################################################################
##### EDA Part 2 #####
######################################################################################################################################################

# Calculate correlation matrix
corr = X_reduced.corr()
fig, ax = plt.subplots(figsize=(10,10))
sn.heatmap(corr, annot=True,ax=ax)

# Calculating variance inflation factor to check multicollinearity
vif = pd.DataFrame()
vif["variables"] = X_reduced.columns
vif["VIF"] = [variance_inflation_factor(X_reduced.values, i) for i in range(X_reduced.shape[1])]
print ("VIF")

display (vif)


# ##### Analysis #####
# Both suggest there are extremely high correlations between Water and Carb, Water and Energy, Folic Acid and Folate, DFE. In that case, we will drop Water and Folate from the feature set.

# In[6]:


######################################################################################################################################################
##### Remove highly correlated variables #####
######################################################################################################################################################

X_reduced = X_reduced.drop(['Folate, DFE (mcg_DFE)','Water\n(g)'], 1)


# In[7]:


######################################################################################################################################################
##### EDA Part 3 #####
######################################################################################################################################################
# Standardize independent variables
# This is so we can plot all the columns on the same boxplot and also not have variable weights skewed by magnitude in our model later on

def standardize(df):
    scaler = StandardScaler()
    df2 = pd.DataFrame(scaler.fit_transform(df))
    df2.columns = df.columns.values
    df2.index = df.index.values
    return df2

# Standardize all
X_reduced_std = standardize(X_reduced)

# Create a box plot for each column in standardized reduced X set to detect outlier possibilities
fig, ax = plt.subplots(figsize=(10,10))
ax.boxplot(X_reduced_std)
ax.set_xticklabels(list(X_reduced.columns),rotation = 45, ha="right")
plt.show()


# ##### Analysis #####
# 
# It looks like there are extreme outlier records in some columns, such as Theobromine, Folic acid, 12:0, and Cholesterol. Let's remove records where they are 3 or more standard deviations from the mean (more extreme than 99.7% of observed dataset)
# 

# In[8]:


######################################################################################################################################################
##### Remove outlier records #####
######################################################################################################################################################

# Combine new X and Y
df_new = pd.concat([X_reduced_std, y],axis=1)

z = np.abs(stats.zscore(X_reduced_std))

print (z)

df_new = df_new[(z < 3).all(axis=1)]

print (df_new.shape)


# In[9]:


# Split final data set to dependent vs independent variables    
y_final = df_new['Sugars, total\n(g)']
X_final = df_new.drop(['Sugars, total\n(g)'], axis = 1)

######################################################################################################################################################
##### First pass: Simple 80/20 split of Data Set #####
######################################################################################################################################################

# Split data into train vs test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.2, shuffle=True, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

display(X_train.head())
display(X_test.head())


# In[10]:


######################################################################################################################################################
##### Linear Regression #####
######################################################################################################################################################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

reg = LinearRegression().fit(X_train, y_train)

# Test model on the test set
y_test_pred = reg.predict(X_test)

######################################################################################################################################################
##### Linear Regression - Evaluation #####
######################################################################################################################################################

# evaluate via RMSE
#note that for the mean_squared_error() function, the parameter "squared=False" will return the RMSE value
rms = mean_squared_error(y_test, y_test_pred, squared=False)
print('RMSE - test:', rms)

# evaluate via R-squared
r_sq = r2_score(y_test, y_test_pred)
print('R Squared Score - test:', r_sq)
print ('')

# Print actual vs predicted values for Sugar
print (pd.DataFrame(list(zip(y_test,y_test_pred)),columns=['y_test_actual','y_test_pred']).head())


# ##### Analysis #####
# The R squared is rather low when ran on the test set, so perhaps a linear regression model isn't the best fit.
# Let's try a polynomial regression next, which might capture some non-linear patterns in the underlying data

# In[11]:


######################################################################################################################################################
##### Polynomial Regression #####
######################################################################################################################################################

from sklearn.preprocessing import PolynomialFeatures

# Fitting Polynomial Regression to the dataset
poly_reg = PolynomialFeatures(degree=3)
X_train_poly = poly_reg.fit_transform(X_train)
pol_reg = LinearRegression()
pol_reg.fit(X_train_poly, y_train)


# Trying polynomial regression model on the test set

# Transform test data to convert into polynomial form
X_test_poly = poly_reg.fit_transform(X_test)

# Predicting on test set
y_test_pred_poly = pol_reg.predict(poly_reg.fit_transform(X_test))

######################################################################################################################################################
##### Polynomial Regression - Evaluation #####
######################################################################################################################################################

# evaluate via RMSE
rms2 = mean_squared_error(y_test, y_test_pred_poly, squared=False)
print('RMSE - test:', rms2)

r_sq2 = r2_score(y_test, y_test_pred_poly)
print('R Squared Score - test:', r_sq2)
print ('')

# Print actual vs predicted values for Sugar
print (pd.DataFrame(list(zip(y_test,y_test_pred_poly)),columns=['y_test_actual','y_test_pred']).head())


# ##### Analysis #####
# 
# Based on the RMSE, it looks like Polynomial Regression with degree of 3 actually does worse for testing data set for linear regression!
# Also the R-squared value is abnormal. We will try a non parametric method next.

# In[12]:


######################################################################################################################################################
##### Decision Tree Regression #####
######################################################################################################################################################

from sklearn.tree import DecisionTreeRegressor 
from sklearn import tree

# create a regressor object
rt = DecisionTreeRegressor()
  
# fit the regressor with training data
tree_reg = rt.fit(X_train, y_train)

# show feature importance
print ('Feature Importance, Ranked')
i = 0
for importance, name in sorted(zip(rt.feature_importances_, X_train.columns),reverse=True):
    i += 1
    print (i, name, importance)
print ('')


# run on test set
y_test_pred_tree = tree_reg.predict(X_test)

######################################################################################################################################################
##### Decision Tree Regression - Evaluation #####
######################################################################################################################################################

# evaluate via RMSE
rms3 = mean_squared_error(y_test, y_test_pred_tree, squared=False)
print('RMSE - test:', rms3)

# evaluate via R-squared
r_sq3 = r2_score(y_test, y_test_pred_tree)
print('R Squared Score - test:', r_sq3)
print ('')

# Print actual vs predicted values for Sugar
print (pd.DataFrame(list(zip(y_test,y_test_pred_tree)),columns=['y_test_actual','y_test_pred']).head())


# ##### Analysis #####
# 
# R-Squared is reasonably high for this model, so I am ok with this. 
# The feature ranks make sense intuitively as well, such as the carb content ranked as most important in determining sugar level.

# 
# ##### Extending to Brazilian Market  #####
# 
# If I were to apply this model to the Brazilian Market, I would assume that there is a similar breakout in food types (processed vs homemade), access to the same raw ingredients (i.e. fruits, vegetables, meat cuts), and flavor profiles (similar preference in level of sweetness and saltiness). It was also mentioned in the assignment memo that often times we do not have sugar content data available for the Brazilian market, so we would also want to know what features are available and what overlap is there with the US feature set. 
# 
# To prove or disprove these assumptions, I would compare the distribution of available features like carbohydrate content, energy (kcal), between the US and Brazilian data sets to see if they are relatively similar. I would compare the mean, median, standard deviation and the percentiles of these features for both sets.

# In[13]:


######################################################################################################################################################
##### Rank Fruit Sweetness #####
######################################################################################################################################################

# Rank the basic fruit groups (cherries, bananas, apple, oranges etc) by their sweetness meaning total sugar / 100g
# Since we imported the dataset earlier, we can re-use the dataframe for part 2 of the assignment

# Define basic fruit groups
fruit_groups = ['Apple','Bananas','Grapes','Peaches and nectarines','Strawberries','Blueberries and other berries',
                'Citrus fruits', 'Melons', 'Dried fruits', 'Other fruits and fruit salads', 'Pears',
                'Pineapple', 'Mango and papaya']

# Assuming that this excludes dried and canned fruits and we only want "raw" fruits

# Filter original data set for basic fruits category and "raw" fruits
df_fruits = df[(df['WWEIA Category description'].isin(fruit_groups)) & (df['Main food description'].str.contains(", raw"))]

# Sort by sugar content, decreasing
df_fruits_sorted = df_fruits.sort_values('Sugars, total\n(g)', ascending=False).reset_index()

# Output relevant columns
print ('Basic Fruits (Raw) - Ranked from most to least sweet')
df_fruits_sorted[['Main food description','WWEIA Category description','Sugars, total\n(g)']]


# In[ ]:




