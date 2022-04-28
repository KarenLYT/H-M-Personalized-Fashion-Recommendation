import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

## First of all, exploratory Data Analysis of provided datasets: 

articles = pd.read_csv("h-and-m-personalized-fashion-recommendations/articles.csv")
customers = pd.read_csv("h-and-m-personalized-fashion-recommendations/customers.csv")
transactions = pd.read_csv("h-and-m-personalized-fashion-recommendations/transactions_train.csv")


## Articles Data Analysis: 

print(articles.head())


f, ax = plt.subplots(figsize=(15, 7))
ax = sns.histplot(data = articles, y='index_name', color='orange')
ax.set_xlabel('count by index name')
ax.set_ylabel('index name')
plt.show()
## Observation from the histogram:
## Ladieswear is the biggest part among all merchandize. Sportwear take the least part.



f, ax = plt.subplots(figsize=(15,7))
ax = sns.histplot(data=articles, y='garment_group_name', color='orange', hue='index_group_name', multiple="stack")
ax.set_xlabel('count by garment group')
ax.set_ylabel('garment group')
plt.show()
## Findings:
## Jersey fancy for women and childrren is the most frequent-purchased garment 
## And accessories are the next big portion - since many accessories have lower pricepoints comparing to other departments



artidx = articles.groupby(['index_group_name', 'index_name']).count()['article_id']
print(artidx)
## Using the group by menthod to find which department has subgroups


pd.options.display.max_rows = 999
protype = articles.groupby(['product_group_name', 'product_type_name']).count()['article_id']
print(protype)
## Findings:
## Accessories provide the most various selections for customers. 
## And trousers have the most amount(11169)


for col in articles.columns:
    if not 'no' in col and not 'code' in col and not 'id' in col:
        un_n = articles[col].nunique()
        print(f'n of unique {col}: {un_n}')



## Customers Data Analysis: 

## A sneakpeak: 
pd.options.display.max_rows = 50
print(customers.head())


## check if there is any duplication in data, and the answer is no:
checkdup = customers.shape[0] - customers['customer_id'].nunique()
print(checkdup)


## check the more often shopping group's age: 
sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(15,7))
ax = sns.histplot(data=customers, x='age', bins=50, color='orange')
ax.set_xlabel('Distribution of the customers age')
plt.show()
## the most often shopping age range is between 21-23 years old


## Check each customer's status in H&M Club:
sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(15,7))
ax = sns.histplot(data=customers, x='club_member_status', color='orange')
ax.set_xlabel('Distribution of club member status')
plt.show()
## Majority of cutsomers keep active in the club, little portion of customers are in the process of activation 
## very very less(almost none) customers have left the club

## clean all no data cells
print(customers['fashion_news_frequency'].unique())

customers.loc[~customers['fashion_news_frequency'].isin(['Regularly', 'Monthly']), 'fashion_news_frequency'] = 'None'
print(customers['fashion_news_frequency'].unique())


## Use the pie chart to find out how customers are like to get fashion news
pie_data = customers[['customer_id', 'fashion_news_frequency']].groupby('fashion_news_frequency').count()

sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(15,7))
colours = sns.color_palette('pastel')
ax.pie(pie_data.customer_id, labels=pie_data.index, colors = colours)
ax.set_facecolor('lightgrey')
ax.set_xlabel('Distribution of fashion news frequency')
plt.show()
## According to the pie chart, more than half of custoners prefer not receving fashion news


## Transactions Data Analysis: 

## Sneakpeak first: 
print(transactions.head())

pd.set_option('display.float_format', '{:.4f}'.format)
print(transactions.describe()['price'])


## Find out price outliers:
sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(15,7))
ax = sns.boxplot(data=transactions, x='price', color='orange')
ax.set_xlabel('Price outliers')
plt.show()


## Find top 10 customers who placed most transactions:  
print(transactions.groupby('customer_id').count().sort_values(by='price', ascending=False)['price'][:10])



## Find price varianes for different groups
articles_for_merge = articles[['article_id', 'prod_name', 'product_type_name', 'product_group_name', 'index_name']]
articles_for_merge = transactions[['customer_id', 'article_id', 'price', 't_dat']].merge(articles_for_merge, on='article_id', how='left')


sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(25,18))
ax = sns.boxplot(data=articles_for_merge, x='price', y='product_group_name')
ax.set_xlabel('Price outliers', fontsize=22)
ax.set_ylabel('Index names', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)

plt.show()
## Findings: 
## Price variances of Lower/Upper/Full body are big - probably due to differenct collections with designer collborations


## Shown on the previous plot that in Accessories there are some high price point merch, find out which ones: 
sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(25,18))
_ = articles_for_merge[articles_for_merge['product_group_name'] == 'Accessories']
ax = sns.boxplot(data=_, x='price', y='product_type_name')
ax.set_xlabel('Price outliers', fontsize=22)
ax.set_ylabel('Index names', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
del _

plt.show()
## Findings: largest outliers are from bags, scarf and others - which is pretty reasonable,
## considering bags are more expensive than small accs such as hair band or cap


## find the highest & lowest average prices by indexes: 
articles_index = articles_for_merge[['index_name', 'price']].groupby('index_name').mean()
sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(15,7))
ax = sns.barplot(x=articles_index.price, y=articles_index.index, color='orange', alpha=0.8)
ax.set_xlabel('Price by index')
ax.set_ylabel('Index')
plt.show()
## Findings: The index with the highest avg price is Ladieswear, 
## and with the lowest avg price is children.


## find the highest & lowest average price by product group: 
articles_index = articles_for_merge[['product_group_name', 'price']].groupby('product_group_name').mean()
sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(10,5))
ax = sns.barplot(x=articles_index.price, y=articles_index.index, color='orange', alpha=0.8)
ax.set_xlabel('Price by product group')
ax.set_ylabel('Product group')
plt.show()
## Findings: Shoes have the highest avg price & stationary has the lowest


## Check the avg price change in time for the top five product groups: 
articles_for_merge['t_dat'] = pd.to_datetime(articles_for_merge['t_dat'])


product_list = ['Shoes', 'Garment Full body', 'Bags', 'Garment Lower body', 'Underwear/nightwear']
colors = ['cadetblue', 'orange', 'mediumspringgreen', 'tomato', 'lightseagreen']
k = 0
f, ax = plt.subplots(3, 2, figsize=(20, 15))
for i in range(3):
    for j in range(2):
        try:
            product = product_list[k]
            articles_for_merge_product = articles_for_merge[articles_for_merge.product_group_name == product_list[k]]
            series_mean = articles_for_merge_product[['t_dat', 'price']].groupby(pd.Grouper(key="t_dat", freq='M')).mean().fillna(0)
            series_std = articles_for_merge_product[['t_dat', 'price']].groupby(pd.Grouper(key="t_dat", freq='M')).std().fillna(0)
            ax[i, j].plot(series_mean, linewidth=4, color=colors[k])
            ax[i, j].fill_between(series_mean.index, (series_mean.values-2*series_std.values).ravel(), 
                             (series_mean.values+2*series_std.values).ravel(), color=colors[k], alpha=.1)
            ax[i, j].set_title(f'Mean {product_list[k]} price in time')
            ax[i, j].set_xlabel('month')
            ax[i, j].set_xlabel(f'{product_list[k]}')
            k += 1
        except IndexError:
            ax[i, j].set_visible(False)
plt.show()


## Images with Descriptions and Prices:

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



max_price_ids = transactions[transactions.t_dat==transactions.t_dat.max()].sort_values('price', ascending=False).iloc[:5][['article_id', 'price']]
min_price_ids = transactions[transactions.t_dat==transactions.t_dat.min()].sort_values('price', ascending=True).iloc[:5][['article_id', 'price']]


## Find the top five most expensive merch with photos 
f, ax = plt.subplots(1, 5, figsize=(20,10))
i = 0
for _, data in max_price_ids.iterrows():
    desc = articles[articles['article_id'] == data['article_id']]['detail_desc'].iloc[0]
    desc_list = desc.split(' ')
    for j, elem in enumerate(desc_list):
        if j > 0 and j % 5 == 0:
            desc_list[j] = desc_list[j] + '\n'
    desc = ' '.join(desc_list)
    img = mpimg.imread(f'h-and-m-personalized-fashion-recommendations/images/0{str(data.article_id)[:2]}/0{int(data.article_id)}.jpg')
    ax[i].imshow(img)
    ax[i].set_title(f'price: {data.price:.2f}')
    ax[i].set_xticks([], [])
    ax[i].set_yticks([], [])
    ax[i].grid(False)
    ax[i].set_xlabel(desc, fontsize=10)
    i += 1
plt.show()


## Find the top five least expensive merch with photos 
f, ax = plt.subplots(1, 5, figsize=(20,10))
i = 0
for _, data in min_price_ids.iterrows():
    desc = articles[articles['article_id'] == data['article_id']]['detail_desc'].iloc[0]
    desc_list = desc.split(' ')
    for j, elem in enumerate(desc_list):
        if j > 0 and j % 4 == 0:
            desc_list[j] = desc_list[j] + '\n'
    desc = ' '.join(desc_list)
    img = mpimg.imread(f'h-and-m-personalized-fashion-recommendations/images/0{str(data.article_id)[:2]}/0{int(data.article_id)}.jpg')
    ax[i].imshow(img)
    ax[i].set_title(f'price: {data.price:.4f}')
    ax[i].set_xlabel(desc, fontsize=10)
    ax[i].set_xticks([], [])
    ax[i].set_yticks([], [])
    ax[i].grid(False)
    i += 1
plt.axis('off')
plt.show()


## For the predction, the rationale is: 
## If there are articles for a certain client, pick the customer's most recent buys
## If there are no purchased articles for a certain client, recommend the most frequently buyed articles



