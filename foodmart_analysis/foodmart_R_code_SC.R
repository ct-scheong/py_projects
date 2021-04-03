#####################################################################################################
# Title: FoodMart Data Analyst Project
# Author: Ka Ieng Shery Cheong
# Date Created: 12/1/2015
#####################################################################################################

#####################################################################################################
# Setup and Housekeeping
#####################################################################################################

# Set working directory, change as needed
setwd("~/Desktop/gma_homework-master/raw_data")

# Read in CSV files into separate dataframes
class <- read.csv("product_class.csv", na.strings=c("", "NA"))
product <- read.csv("product.csv", na.strings=c("", "NA"))
promotion <- read.csv("promotion.csv", na.strings=c("", "NA"),sep=",",quote = "'")
transactions <- read.csv("transactions.csv", na.strings=c("", "NA"),stringsAsFactors = FALSE) #we want the char variables to remain a char type

# Merge Data Sets between the 4 dataframes using joins
require(dplyr)
transactions2 <- transactions %>% left_join(product) %>% left_join(class) %>% left_join(promotion)

# Some numerical variables have an apostrophe, so let's get rid of them
transactions2[] <- lapply(transactions2, gsub, pattern="'", replacement="")

# Convert numeric values stored as characters back to numeric
transactions2[c("store_sales","store_cost","unit_sales")] <- sapply(transactions2[c("store_sales","store_cost","unit_sales")],as.numeric)

# Create a new column called 'profits', calculated as (sales-cost)*units
transactions2$profits <- (transactions2$store_sales-transactions2$store_cost)*transactions2$unit_sales

#####################################################################################################
# Create Aggregation summaries -- brand lvl and product category lvl
#####################################################################################################

# Product Category -- get top 5 ranked by total profits

# First, get the top 5 categories ranked by total profits
prod.top5.sales <- transactions2 %>%
  group_by(product_category) %>%
  summarise(total_profits = sum(profits)) %>%  #aggregated by total profits
  arrange(desc(total_profits)) %>%  #sort by descending total profits
  top_n(n=5,wt=total_profits) #return top 5 ranked

# Use that list to filter the full dataset by the top 5, aggregated by total profits
prod.top5.sales.summary <- transactions2 %>%
  group_by(product_category,month_of_year,the_year) %>%
  summarise(total_profits = sum(profits)) %>% #aggregated by total profits
  filter(product_category %in% prod.top5.sales$product_category) #filter for the top 5 only

#####################################################################################################
# Product Category -- get top 5 ranked by greatest cumulative month-to-month change
# (these are the most seasonal items with biggest fluctuations in month to month sales)
prod.top5.change <- transactions2 %>%
  group_by(product_category,month_of_year,the_year) %>%
  summarise(total_profits = sum(profits))

# Create the percent change column (can't do it with dplyr chaining since it doesn't like the lag() function,
# so we will do it the old school way)

prod.top5.change$last_month = lag(prod.top5.change$total_profits) #create a variable to store the sales from previous row
prod.top5.change$last_month[prod.top5.change$month==1] <-0 #however, for all January values this should be a 0 since we don't have data for the previous month
prod.top5.change <- mutate(prod.top5.change,percent_change = (total_profits-last_month)/last_month) #create the new % change variable
prod.top5.change$percent_change[prod.top5.change$percent_change==Inf] <-0 #January months have a percent change error value, so we will change them to 0's

# Going back to dplyr chaining again
prod.top5.change <- prod.top5.change %>%
  group_by(product_category)  %>% #we want to aggregate the total percent change over all months
  summarise(total_percent_change = sum(abs(percent_change))) %>%
  arrange(desc(total_percent_change)) %>%  #sort by descending total percent change
  top_n(n=5,wt=total_percent_change) #return top 5 ranked

# Use that list to filter the full dataset "transactions2" by the top 5, aggregated by most change
prod.top5.change.summary <- transactions2 %>%
  group_by(product_category,month_of_year,the_year) %>%
  summarise(total_profits = sum(profits)) %>% #aggregated by total profits
  filter(product_category %in% prod.top5.change$product_category) #filter for the top 5 only

#####################################################################################################
# Now, look for the most profitable brands
# Product Category -- get top 5 ranked by total profits

# First, get the top 5 brands ranked by total profits
brand.top5.sales <- transactions2 %>%
  group_by(brand_name) %>%
  summarise(total_profits = sum(profits)) %>%  #aggregated by total profits
  arrange(desc(total_profits)) %>%  #sort by descending total profits
  top_n(n=5,wt=total_profits) #return top 5 ranked

# Use that list to filter the full dataset by the top 5, aggregated by total profits
brand.top5.sales.summary <- transactions2 %>%
  group_by(brand_name,month_of_year,the_year) %>%
  summarise(total_profits = sum(profits)) #aggregated by total profits

# Change the brands not in top 5 to 'Other'
brand.top5.sales.summary$brand_name[!brand.top5.sales.summary$brand_name %in% brand.top5.sales$brand_name] <- '*Other*'

# Re-aggregate the data after labeling the 'Other' brands
brand.top5.sales.summary <- brand.top5.sales.summary %>%
  group_by(brand_name,month_of_year,the_year) %>%
  summarise(total_profits = sum(total_profits)) #aggregated by total profits

#####################################################################################################
# For the top 5 brands, look at the percentage of products sold that offer recyclable packaging

# Use the previously created top 5 list to filter the full dataset, aggregated by total profits
brand.top5.recyl.summary <- transactions2 %>%
  group_by(brand_name) %>%
  mutate(total_recy_sold = as.numeric(recyclable_package)*unit_sales) %>% #multiply the recyclable boolean by units sold to get the total recyclable units sold
  summarise(total_profits = sum(profits), total_items =sum(unit_sales),total_recy_sold=sum(total_recy_sold)) %>% #aggregated by total profits, total units sold
  mutate(percentage_recy = total_recy_sold/total_items) %>% #create a new var as percentage of total units sold
  arrange(desc(total_profits)) %>%  #sort by descending total profits
  top_n(n=5,wt=total_profits) #return top 5 ranked

#####################################################################################################
# For the top 5 product categories, look at the percentage of products sold that are low-fat

# Use the previous list to filter the full dataset by the top 5, aggregated by total profits
prod.top5.lowfat.summary <- transactions2 %>%
  group_by(product_category) %>%
  mutate(total_lf_sold = as.numeric(low_fat)*unit_sales) %>% #multiple the lowfat boolean by units sold to get the total recyclable sold
  summarise(total_profits = sum(profits),total_lf_sold = sum(total_lf_sold), total_items =sum(unit_sales)) %>% #aggregated by total profits
  mutate(percentage_lf = total_lf_sold/total_items) %>% #create a new var as percentage of total units sold
  arrange(desc(total_profits)) %>%  #sort by descending total profits
  top_n(n=5,wt=total_profits) #return top 5 ranked

#####################################################################################################
# EDA Data Visualizations
#####################################################################################################

library(ggplot2)
# Time series plot of Top 5 profitable product categories
ggplot(prod.top5.sales.summary, aes(x=as.numeric(month_of_year),y=total_profits,col=factor(product_category))) + 
  geom_line() + geom_point() +scale_x_continuous(breaks=c(1:12), labels=c(1:12),limits=c(1,12)) +
  ggtitle("Top 5 most profitable product categories") +  labs(x="Month",y="Total Profits") + 
  guides(color=guide_legend(title="Product Category"))

# Time series plot of Top 5 product categories with most fluctuations
ggplot(prod.top5.change.summary, aes(x=as.numeric(month_of_year),y=total_profits,col=factor(product_category))) + 
  geom_line() + geom_point() +scale_x_continuous(breaks=c(1:12), labels=c(1:12),limits=c(1,12)) +
  ggtitle("Top 5 product categories with most fluctuations in total profits") +  labs(x="Month",y="Total Profits") + 
  guides(color=guide_legend(title="Product Category"))

# Stacked barplot showing total profits, with breakouts,for the top 5 most profitable brands
ggplot(brand.top5.sales.summary, aes(x=as.numeric(month_of_year),y=total_profits,fill=brand_name)) + 
  geom_bar(stat = "identity") + ggtitle("Monthly Profits - By Brand") +  labs(x="Month",y="Total Profits") + 
  guides(color=guide_legend(title="Brand Name")) +scale_x_continuous(breaks=c(1:12), labels=c(1:12),limits=c(0,13))

# Percentage of items sold that are recyclable, for the top 5 most profitable brands
ggplot(brand.top5.recyl.summary, aes(x=brand_name,y=percentage_recy)) + geom_bar(stat = "identity") +
  ggtitle("Percentage of items sold that have recyclable packaging, top 5 brands") +  labs(x="Brand",y="Percent") + 
  guides(color=guide_legend(title="Brand Name")) + scale_fill_hue()

# Percentage of items sold that are low-fat, for the top 5 most profitable brands
ggplot(prod.top5.lowfat.summary, aes(x=product_category,y=percentage_lf)) + geom_bar(stat = "identity") +
  ggtitle("Percentage of items sold that are low fat") +  labs(x="Brand",y="Percent") + 
  guides(color=guide_legend(title="Product Category")) + scale_fill_hue()

#####################################################################################################
# Promotional Analysis -- Where to target promotions? Which ones are most effective?
#####################################################################################################

# Create a summary table showing the average sales per transaction that occurred with a promotion
promo.summary <- transactions2 %>%
  filter(promotion_id != 0) %>% #filter for the promotion transactions only
  group_by(promotion_id,product_name,product_department,promotion_name,media_type) %>%
  summarise(avg_sales = mean(unit_sales)) #aggregated by avg number sold per product name (can compare profits on a product-level)

# Create a summary table showing the average sales per transaction that occurred without a promotion, on a product level
no.promo.summary <- transactions2 %>%
  filter(promotion_id == 0) %>% #filter for the no promotion transactions
  group_by(promotion_id,product_name,product_department,promotion_name,media_type) %>%
  summarise(avg_sales = mean(unit_sales))

# Merge the 2 tables together
promo.summary.joined <- data.frame(rbind(promo.summary,no.promo.summary))
promo.summary.joined$promo <- ifelse(promo.summary.joined$promotion_id ==0,0,1)

# Do one-way ANOVA to see whether a promotion in general had a significant effect on the average sales per product
anov1 <- anova(lm(avg_sales ~ promo ,promo.summary.joined))

# Do two-way ANOVA to see whether there is had a significant effect on the average sales per product for different media types and promotion names
anov2 <- anova(lm(avg_sales ~ media_type + promotion_name, promo.summary.joined))

# Now do pairwise comparison for Media Type since the F-test is significant, and export resulting p-values as a matrix
pairwise.t.test(promo.summary.joined$avg_sales, promo.summary.joined$media_type, p.adj = "bonf",sd=FALSE)[["p.value"]]

# Results show that although some media types lead to signifantly different sales units than others, there is only one which
# is significant against "No Media". This result isn't too useful. Let's look at the Promotion Name ...

# Now do pairwise comparison for Promotion Name, and since we have more than 1 significant factor, we export resulting p-values as a matrix
pairwise.matrix <- data.frame(pairwise.t.test(promo.summary.joined$avg_sales, promo.summary.joined$promotion_name, p.adj = "bonf",sd=FALSE)[["p.value"]])
pairwise.matrix[is.na(pairwise.matrix)] <- 1 #replace NAs with 1, rule them out to be insignificant and will eventually be filtered out

# Focus only on the comparisons with "No Promotion" and filter for those which are statistically significant at alpha level = 0.05
nopromo.comp <- subset(pairwise.matrix, No.Promotion < 0.05, select = No.Promotion)

# Now that we narrowed our focus to 4 Promotion Names, let's filter the promo.summary.joined dataframe for just those 4 
promo.significant <- promo.summary.joined %>%
  filter(promotion_name %in% row.names(nopromo.comp))

# Let's join in a column with the average units sold for each those products, to use as a baseline for comparison
promo.significant.join <- promo.significant[c("product_name","product_department","promotion_name","media_type","avg_sales")] %>%
  left_join(no.promo.summary[c("product_name","avg_sales")],by="product_name")

# Fix up the names
names(promo.significant.join) <- c("product_name","product_department","promotion_name","media_type","avg_sales_promo","avg_sales_nopromo")

# Get the percentage change due to promo
promo.significant.join$perc_change <- (promo.significant.join$avg_sales_promo -promo.significant.join$avg_sales_nopromo)/promo.significant.join$avg_sales_nopromo

# Use dyplr chaining to collapse the product name, it is too granular.
# Summaize by average percent change by product department since it is a little more high level
promo.significant.sort <- promo.significant.join %>%
  group_by(product_department,promotion_name ,media_type) %>% #collapse product name
  summarize(avg_perc_change = mean(perc_change)) #re-calculate the average after collapsing

# Sort by descending average percent change
promo.significant.sort <- promo.significant.sort[order(-promo.significant.sort$avg_perc_change),]

# Print top 10 most effective promos
promo.significant.sort[1:10,]

# Print top 10 least effective promos
promo.significant.sort[order(promo.significant.sort$avg_perc_change),][1:10,]

#####################################################################################################
# Calculating impact

# Aggreate transaction data by total profits and arrange in descending order
sales_by_product <- transactions2 %>%
  group_by(product_category) %>%
  summarise(total_profits = sum(profits)) %>%
  arrange(desc(total_profits))

# Add a variable for percentage of total profits
sales_by_product$percent_profit <- sales_by_product$total_profits/sum(sales_by_product$total_profits)
