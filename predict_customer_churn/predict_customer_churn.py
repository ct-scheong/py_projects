#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Title: User Activity Analysis
# Author: Shery Cheong
# Description: Understand the monthly volume of customer churn, increase customer retention.
# Date: 03/22/21
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
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

# Suppress scientific notation
pd.options.display.float_format = '{:.2f}'.format

print ("Imported libraries - success")


# In[2]:


######################################################################################################################################################
##### Data Cleaning #####
######################################################################################################################################################

# Read in csv input
df = pd.read_csv('user_activity_log.csv')
print ('Dataframe head')
display (df.head())
print('\n')

# Check column types
print ('Dataframe column types')
print (df.info())
print('\n')

# Check for null/missing values
print ('Dataframe missing values')
print (df.isnull().sum(axis = 0))
print('\n')

# Replace missing values with zero. 
# Since we are not doing trend analysis but instead modelling at the user level, it should be fine to assume zero if missing.
df = df.fillna(0)


# In[3]:


######################################################################################################################################################
##### EDA Part 1 #####
######################################################################################################################################################

# Aggregate mean metrics by uuid (user)
df_user = df.drop(['billing_segment_id'], axis=1)
df_user.groupby('uuid').agg(['mean'])

# Plot user level data
plt = df_user.plot(subplots=True, layout=(2, 3), figsize=(10, 10), sharex=False)


# In[4]:


######################################################################################################################################################
##### EDA Part 2 #####
######################################################################################################################################################

# Set independent variables
X = df[['monthly_spend_sum','ticket_creation_cnt', 'referral_cnt','product_a_cnt','product_b_cnt','website_bug_cnt']] 

# Calculate correlation matrix
corr = X.corr()
sn.heatmap(corr, annot=True)

# Calculating variance inflation factor to check multicollinearity
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print ("VIF")
display (vif)


# In[5]:


######################################################################################################################################################
##### Data Wrangling Part 1 #####
######################################################################################################################################################

######################################################################################################################################################
# Create quantitative variables
######################################################################################################################################################

# Create columns for min and max month
extra_vars = df.groupby('uuid').agg({'billing_segment_id': ['min', 'max']}) 
# Flatten column names
extra_vars.columns = ['_'.join(col).strip() for col in extra_vars.columns.values]

# Create column for user "tenure"
extra_vars['tenure'] = extra_vars['billing_segment_id_max'] - extra_vars['billing_segment_id_min']

# Get an idea of average user "tenure"
print ("Average Tenure")
print(extra_vars['tenure'].describe())
print('\n')

# Get distribution of months (suppress scientific notation)
print ("Billing Segment Id Distribution")
print (df['billing_segment_id'].describe().apply(lambda x: format(x, 'f')))
print('\n')

# Assign a churn flag
# If customer was NOT active in the last 2 months, we can assume they have "churned"
# Assume that the current month is 103, and anyone who does not have a record  prior to 102 is churned
extra_vars['churn'] = (np.where(extra_vars.billing_segment_id_max <= 102, 1, 0)) 

# Get an idea of churn vs active user ratio
print ("Churn vs Active Distribution")
print(extra_vars["churn"].value_counts())
print ("Percent of churned users")
print (sum(extra_vars["churn"])/len(extra_vars["churn"]))
print('\n')

# Join these user level variables back to the original data set
df_extended = pd.merge(left=df, right=extra_vars, how='left', left_on='uuid', right_on='uuid')

print (df_extended.head())


# In[6]:


######################################################################################################################################################
##### Data Wrangling Part 3 #####
######################################################################################################################################################

# Aggregate data to the user level for model input
# While we can possibly work with time series data, it will be simpler if there was one input record per user

df_user_final = df_extended.groupby(['uuid', 'tenure', 'churn']).agg({'monthly_spend_sum': ['mean','sum'], 'ticket_creation_cnt': ['sum'], 
                                 'referral_cnt' : ['sum'], 'product_a_cnt' : ['sum'], 'product_b_cnt' : ['sum'],
                                 'website_bug_cnt' : ['sum'], 
                                 'billing_segment_id_max' : ['max']}) 

# Flatten column names
df_user_final.columns = ['_'.join(col).strip() for col in df_user_final.columns.values]
df_user_final.reset_index(inplace=True)

# Rename some columns so they are less confusing
# We only used the "max" function in the aggregation to get whether or not the value is 1 for each user
df_user_final.rename({"bug_on_churn_max": "bug_on_churn", "ticket_on_churn_max": "ticket_on_churn", "billing_segment_id_max_max" : "billing_segment_id_max"}, axis=1, inplace=True)

display (df_user_final.head())


# In[7]:


######################################################################################################################################################
##### Standardize Data Set #####
######################################################################################################################################################

# Split final data set to dependent vs independent variables    
y = df_user_final['churn']
X = df_user_final.drop(['churn'], axis = 1)

# Standardize independent variables to reduce multicollinearity and so all variables contribute equally to the model
def standardize(df):
    scaler = StandardScaler()
    df2 = pd.DataFrame(scaler.fit_transform(df))
    df2.columns = df.columns.values
    df2.index = df.index.values
    return df2

# Standardize all except uuid
X_std = standardize(X.loc[:, X.columns != 'uuid'])

# Append uuid column back to standardized dataset
X_std['uuid'] = X['uuid']

######################################################################################################################################################
##### First pass: Simple 80/20 split of Data Set #####
######################################################################################################################################################

# Split data into train vs test (80% train, 20% test)
# Stratify based on response so that both sets contain similar ratio of churn vs active customers
X_train, X_test, y_train, y_test = train_test_split(X_std, y,stratify=y, test_size = 0.2, shuffle=True, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

display(X_train.head())
display(X_test.head())


# In[8]:


######################################################################################################################################################
##### Logistic Regression: Initial pass #####
######################################################################################################################################################

# Store X without uuid or the max billing month (exclude both as a feature in model)
X_train_no_uuid = X_train.drop(['uuid','billing_segment_id_max'], axis= 1)
X_test_no_uuid = X_test.drop(['uuid', 'billing_segment_id_max'], axis= 1)

# Instantiate model and train (exclude uuid column in model)
log_reg = LogisticRegression()
log_reg.fit(X_train_no_uuid, y_train)

print ('Logistic Regression: Model Coefficients')
display (pd.DataFrame(zip(X_train_no_uuid.columns, np.transpose(log_reg.coef_)), columns=['features', 'coef']))
print ('\n')

# Run model on entire test set
y_predict = log_reg.predict(X_test_no_uuid)

# Evaluate accuracy using score (what percentage of predictions were correct?)
print ('Accuracy Score')
print(metrics.accuracy_score(y_test, y_predict))
print ('\n')

# Evaluate via confusion matrix
cm = metrics.confusion_matrix(y_test, y_predict)
print ('Confusion Matrix')
print (cm)
print ('\n')

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Get probability score of churn for X_test
y_pred_probs = log_reg.predict_proba(X_test_no_uuid)

y_pred_probs = y_pred_probs[:, 1] # we only care about the second column, which is probably of churn = 1

# Store the relevant columns in a final dataframe
final_results = pd.concat([X_test['uuid'], y_test], axis = 1,  names=['uuid', 'churn_actual'])

# Add the predictive columns
final_results['churn_predict'] = y_predict
final_results['churn_probability'] = y_pred_probs

# Merge in tenure years for the test user set
final_results = pd.merge(left=final_results, right=df_user_final[['uuid','tenure']], how='left', left_on='uuid', right_on='uuid')

print ('Final Prediction Results for X_test')
display (final_results.head())

print ('Average number of months in program, for users with 60%+ likelihood to churn')
print(final_results[final_results['churn_probability'] > .6]['tenure'].agg('mean'))


# In[9]:


######################################################################################################################################################
##### Logistic Regression: Cross Validation #####
######################################################################################################################################################

# Check if we are potentially overfitting our model.

# For cross validation we won't need uuid, billing segment
X_std_no_uuid = X_std.drop(['uuid','billing_segment_id_max'], axis= 1)

# Implementing stratified k fold cross validation (5 fold).
# Use stratified k-fold due to imbalance in churn vs active users
# Since we had sorted the data earlier, should apply shuffle so it is sorted before data is split
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0) 
acc_score = [] # this is an empty list to store the accuracy scores for each fold

fold_num = 0

print ('K-fold cross validation, k =',k)

for i , j in skf.split(X_std_no_uuid, y):
    # Track which "fold" we are at so it can be used to print metrics as we are validating data
    fold_num += 1  
    
    # Loop through each "fold", split data and train
    X_train , X_test = X_std_no_uuid.iloc[i,:],X_std_no_uuid.iloc[j,:]
    y_train , y_test = y[i] , y[j]
    
    # Check the ratio of data in each fold to ensure we have a reasonable churn to active user ratio
    print('Fold',str(fold_num), 'Class Ratio:', sum(y_test)/len(y_test))
     
    log_reg.fit(X_train,y_train)
    y_predict = log_reg.predict(X_test)
    
    # Store the accuracy score for each "fold" as an element in the list
    acc = metrics.accuracy_score(y_test, y_predict)
    acc_score.append(acc)

    print('Fold',str(fold_num), 'Accuracy:', acc)    
print ('\n')

# Get the average accuracy score across all the folds
avg_acc_score = sum(acc_score)/k
print ('Average kfold accuracy score')
print (avg_acc_score)


# In[10]:


######################################################################################################################################################
##### Churn Monetary Impact #####
######################################################################################################################################################

# Calculate average tenure of churned users
df_churned = df_user_final[df_user_final['churn'] == 1]
avg_tenure = df_churned['tenure'].agg('mean')
print ("Average Tenure:",str(avg_tenure))

# Get number of total users by month
total_monthly_users = df_extended.groupby('billing_segment_id').agg({'uuid' : ['nunique']})
total_monthly_users.columns = ['_'.join(col).strip() for col in total_monthly_users.columns.values]
total_monthly_users.reset_index(inplace=True)
total_monthly_users.columns = ['billing_segment_id', 'num_uniq_users']

# Aggregate the sum of the average user monthly spend by the month the user churned (i.e. their last month of activity)
monthly_churn = (df_churned.groupby('billing_segment_id_max').agg({'monthly_spend_sum_mean': ['mean', 'count'], 'tenure' : ['mean']}))

monthly_churn.columns = ['_'.join(col).strip() for col in monthly_churn.columns.values]
monthly_churn.reset_index(inplace=True)
monthly_churn.columns = ['billing_segment_id', 'monthly_avg_user_spend', 'num_users_churned', 'avg_tenure']

# Exclude the current month since these users are still considered active
monthly_churn = monthly_churn[monthly_churn['billing_segment_id']< 103]

# Calculate avg monthly spend of churned users
avg_monthly_spend = df_churned['monthly_spend_sum_mean'].agg('mean')
print ("Average Monthly Spend:",str(avg_monthly_spend))

# Average user tenure is 15.69 months, and average monthly spend is 617.55
average_cltv = avg_tenure * avg_monthly_spend
print ("Average CLTV:",str(average_cltv))

# Merge in the num_uniq_users col to calculate monthly churn rate
monthly_churn = pd.merge(left=monthly_churn, right=total_monthly_users, how='left', left_on='billing_segment_id', right_on='billing_segment_id')
monthly_churn['churn_rate'] = monthly_churn['num_users_churned']/monthly_churn['num_uniq_users']
monthly_churn['monthly_spend_loss_to_churn'] = monthly_churn['num_users_churned']*monthly_churn['monthly_avg_user_spend']

# Calculate average month over month change in active users
monthly_churn['program_growth_rate'] = monthly_churn['num_uniq_users'].pct_change()

avg_monthly_growth = monthly_churn['program_growth_rate'].agg('mean')
print ("Average Program Growth Rate:",str(avg_monthly_growth))

display(monthly_churn)


# In[ ]:





# In[ ]:




