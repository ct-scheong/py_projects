# Title: NYC Restaurant Grades
# Author: Shery Cheong
# Description:  Identify trends, insights, correlations and recommendations on inspection ratings for NYC restaurants.
# Date: 06/20/17
# Version: 1.0

######################################################################################################################################################
##### Housekeeping #####
######################################################################################################################################################

# Import all needed packages
import csv
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# Import each of the tab separated Data Files as Data Frame
borough_names = pd.DataFrame(pd.read_csv('assignment_files/borough_names.tsv', sep='\t'))
cuisine_names = pd.DataFrame(pd.read_csv('assignment_files/cuisine_names.tsv', sep='\t'))
restaurant_attributes = pd.DataFrame(pd.read_csv('assignment_files/restaurant_attributes.tsv', sep='\t'))
restaurant_names = pd.DataFrame(pd.read_csv('assignment_files/restaurant_names.tsv', sep='\t'))
restaurant_violations = pd.DataFrame(pd.read_csv('assignment_files/restaurant_violations.tsv', sep='\t'))
violation_names = pd.DataFrame(pd.read_csv('assignment_files/violation_names.tsv', sep='\t'))

######################################################################################################################################################
##### Data Wrangling #####
######################################################################################################################################################

# Given the large number of restaurants, we would like to bucket them based on attributes like cuisine type and borough. This can help us identify trends.

# Let's start by making the appropriate data joins.
rv = restaurant_violations.merge(restaurant_attributes, left_on='restaurant_id',right_on='restaurant_id', how='left')
rv = rv.merge(cuisine_names, left_on='cuisine_id',right_on='id', how='left')

# Drop duplicate columns (i.e. cuisine id dupe) to keep things clean.
rv= rv.drop('id',1)

# Reading restaurant id strings isn't fun. We'd want the restaurant names too.
rv = rv.merge(restaurant_names, left_on='restaurant_id',right_on='id', how='left')
rv= rv.drop('id',1)

# Let's get boroughs and violation names while we're at it.
rv = rv.merge(borough_names, left_on='borough_id',right_on='id', how='left')
rv= rv.drop('id',1)
rv = rv.merge(violation_names, left_on='violation_id',right_on='id', how='left')
rv= rv.drop('id',1)

# Nominal scales can be hard to work with. Let's transform the critical_flag to a boolean.
rv['critical_flag'] = np.where(rv['critical_flag'] == 'Critical', 1, 0)

# We may also want to do some time series analysis. Create a new var to store timestamps as month
rv['month_year'] = pd.to_datetime(rv['grade_date'], format='%m/%d/%Y')
rv['month_year'] = pd.DatetimeIndex(pd.Series(rv['month_year']).apply(lambda x: x.replace(day=1)).dt.date)

# Add in a boolean variable to distinguish warm months from cool months
rv['month_type'] = np.where(rv['month_year'].dt.month.isin([4, 5, 6, 7, 8, 9]), 'warm', 'cool')

# Check in on our mega data set to make sure everything looks good...
# print rv

######################################################################################################################################################
##### Data Aggregation #####
######################################################################################################################################################

##############################################################################
### Level One: Restaurant Visit
##############################################################################

# First I'd like to get the data to one row per restaurant visit, since each visit can have multiple violations. Let's do some aggregation here.
rv_visit_groupby = rv.groupby(['restaurant_id', 'restaurant_name', 'borough_name','cuisine_description','grade_date', 'month_year', 'month_type'])

# We want to get the total number of flags, the score, and the grade for each visit
rv_agg_visit = rv_visit_groupby.agg({'critical_flag' : 'sum', 'score' : 'mean', 'grade' : (lambda x:x.value_counts().index[0])}) # the last bit here aggregates grade by 'mode'
rv_agg_visit.reset_index(inplace=True) # here we are flattening a multi-level index that results from aggregation

# QA Check
# print rv_agg_visit

##############################################################################
### Level Two: Restaurant Average
##############################################################################

# Next, roll it up again to the restaurant level, so we can get an average across all their visits.
rv_res_groupby = rv_agg_visit.groupby(['restaurant_id', 'restaurant_name', 'borough_name','cuisine_description',])
rv_agg_restaurant = rv_res_groupby.agg({'critical_flag' : 'mean', 'score' : 'mean', 'grade' : (lambda x:x.value_counts().index[0]), 'grade_date' : 'count'})
rv_agg_restaurant.reset_index(inplace=True)

# Rename column for clarity
rv_agg_restaurant.rename(columns={'grade_date':'num_visits'}, inplace=True)

# QA Check
# print rv_agg_restaurant.sort(["restaurant_name"])

##############################################################################
### Level Three: By Month/Year
##############################################################################

# Roll up data by month/year
rv_my_groupby = rv_agg_visit.groupby(['month_year'])
rv_agg_my = rv_my_groupby.agg({'critical_flag' : 'mean', 'score' : 'mean', 'grade' : (lambda x:x.value_counts().index[0]), 'grade_date' : 'count'})
rv_agg_my.reset_index(inplace=True)

# Rename column for clarity
rv_agg_my.rename(columns={'grade_date':'num_visits'}, inplace=True)

# Roll up data by warm/cool months
rv_mt_groupby = rv_agg_visit.groupby(['month_type', 'cuisine_description'])
rv_agg_mt = rv_mt_groupby.agg({'critical_flag' : 'mean', 'score' : 'mean', 'grade' : (lambda x:x.value_counts().index[0]), 'grade_date' : 'count'})
rv_agg_mt.reset_index(inplace=True)

# Rename column for clarity
rv_agg_mt.rename(columns={'grade_date':'num_visits'}, inplace=True)

# We want to get one row per cuisine type for the warm/cool month aggregation.
rv_agg_mt_pivot = pd.pivot_table(rv_agg_mt, values='score', index=['cuisine_description'],  columns=['month_type'], aggfunc=np.sum)
rv_agg_mt_pivot.reset_index(inplace=True)

# Create another column to get the percent change between warm and cool months
rv_agg_mt_pivot['score_perc_change'] = (rv_agg_mt_pivot['cool'] - rv_agg_mt_pivot['warm'])/rv_agg_mt_pivot['cool']

# QA Check
# print rv_agg_my
# print rv_agg_mt
# print rv_agg_mt_pivot

##############################################################################
### Level Four: By Cuisine and/or Borough
##############################################################################

# Roll up data by cuisine and borough
rv_cb_groupby = rv_agg_restaurant.groupby(['borough_name','cuisine_description'])
rv_agg_cb = rv_cb_groupby.agg({'critical_flag' : 'mean', 'score' : 'mean', 'grade' : (lambda x:x.value_counts().index[0]), 'restaurant_name' : 'count', 'num_visits' : 'mean'})
rv_agg_cb.reset_index(inplace=True)

# Rename column for clarity
rv_agg_cb.rename(columns={'restaurant_name':'num_restaurants'}, inplace=True)

# It turns out that aggregating by cuisine AND borough gives us too small of a sample size per group to confidently draw conclusions...

# Roll up data by cuisine only
rv_c_groupby = rv_agg_restaurant.groupby(['cuisine_description'])
rv_agg_c = rv_c_groupby.agg({'critical_flag' : 'mean', 'score' : 'mean', 'grade' : (lambda x:x.value_counts().index[0]), 'restaurant_name' : 'count', 'num_visits' : 'mean'})
rv_agg_c.reset_index(inplace=True)

# Rename column for clarity
rv_agg_c.rename(columns={'restaurant_name':'num_restaurants'}, inplace=True)

# Drop row where cuisine is not listed or N/A
rv_agg_c = rv_agg_c[rv_agg_c.cuisine_description != 'Not Listed/Not Applicable']

# Roll up data by borough only
rv_b_groupby = rv_agg_restaurant.groupby(['borough_name'])
rv_agg_b = rv_b_groupby.agg({'critical_flag' : 'mean', 'score' : 'mean', 'grade' : (lambda x:x.value_counts().index[0]), 'restaurant_name' : 'count', 'num_visits' : 'mean'})
rv_agg_b.reset_index(inplace=True)

# Rename column for clarity
rv_agg_b.rename(columns={'restaurant_name':'num_restaurants'}, inplace=True)

# QA Check
# print rv_agg_cb
# print rv_agg_c
# print rv_agg_b

######################################################################################################################################################
##### Exploratory Data Analysis (EDA) #####
######################################################################################################################################################

# Here is where we get to the fun stuff (charts! visualizations!)

##############################################################################
### Worse/Best Cuisine Lists
##############################################################################

# Top 5 worst safety rated cuisines
print tabulate(rv_agg_c.sort_values('score', ascending=False).head(5), headers='keys', tablefmt='psql')

# Top 5 best safety rated cuisines
print tabulate(rv_agg_c.sort_values('score').head(5), headers='keys', tablefmt='psql')

# Average number of inspections and score, by borough
print tabulate(rv_agg_b.sort_values('borough_name', ascending=False), headers='keys', tablefmt='psql')

##############################################################################
### Average Inspection Score vs. Number of Visits
##############################################################################

# Scatterplot
colors = (0,0,0)
area = np.pi*3

plt.scatter(rv_agg_restaurant['num_visits'], rv_agg_restaurant['score'], s=area, c=colors, alpha=0.5)
plt.title('Average Inspection Score vs. Number of Visits')
plt.xlabel('Number of Inspection Visits')
plt.ylabel('Average Inspection Score')
plt.show()

##############################################################################
### Average Inspection Score by Month/Year of Visit
##############################################################################

# Line plot
plt.plot(rv_agg_my['month_year'], rv_agg_my['score'], alpha=0.5)
plt.title('Average Inspection Score by Month')
plt.xlabel('Month/Year')
plt.ylabel('Average Inspection Score')
plt.show()

##############################################################################
### Average Inspection Score by Cuisine, by Cool vs. Warm Months
##############################################################################

# As a followup to the above, what if certain cuisines are more sensitive to temperature changes as the weather warms up?

# Top 5 cuisines where scores are most sensitive to seasonal changes from cool to warm weather
top5warm = pd.DataFrame(rv_agg_mt_pivot.sort_values('score_perc_change').head(5))
top5warm = top5warm.reset_index(drop=True)
print tabulate(top5warm, headers='keys', tablefmt='psql')

# Create a bar for warm/cool weather scores
width = 0.25
x = list(range(len(top5warm['cool'])))
fig, ax = plt.subplots(figsize=(10,5))

warm = plt.bar(x, top5warm['warm'], width, color='r',  alpha=0.5, label=top5warm['cuisine_description'][0])
cool = plt.bar([i + width for i in x], top5warm['cool'], width,  alpha=0.5, color='b' , label=top5warm['cuisine_description'][1])
ax.set_ylabel('Average Inspection Score')
ax.set_title('Fluctuations in Average Inspection Scores from Cool to Warm Months, by Cuisine')
ax.set_xticks([p + width/2 for p in x])
ax.set_xticklabels(top5warm['cuisine_description'])
ax.legend((warm[0], cool[0]), ('Warm', 'Cool'))
plt.show()
