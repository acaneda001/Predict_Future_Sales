import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('ggplot')

df_train = pd.read_csv('input/sales_train.csv')
df_test = pd.read_csv("input/test.csv")
df_categories = pd.read_csv("input/item_categories.csv")
df_items = pd.read_csv("input/items.csv")
df_shops = pd.read_csv("input/shops.csv")


def graph_insight(data):
    df_num = data.select_dtypes(include=['float64', 'int64'])
    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8)
    plt.show()


def boxplot(data):
    df_num = data.select_dtypes(include=['float64', 'int64'])
    df_num.boxplot()


def eda(data):
    print("----------Top-5- Record----------")
    print(data.head(5))
    print("-----------Information-----------")
    print(data.info())
    print("----------Describe-------------")
    print(data.describe())
    print("----------Columns--------------")
    print(data.columns)
    print("-----------Data Types-----------")
    print(data.dtypes)
    print("----------Missing value-----------")
    print(data.isnull().sum())
    print("----------Null value-----------")
    print(data.isna().sum())
    print("----------Shape of Data----------")
    print(data.shape)
    print("----------Duplicates----------")
    print("Duplicated rows   " + str(len(data[data.duplicated()])))


def stat_data(data):
    print("----------Stat Data ----------")
    print("Min Value:", data.min())
    print("Max Value:", data.max())
    print("Average Value:", data.mean())
    print("Center Point of Data:", data.median())


# eda(df_train)
# boxplot(df_train)
# graph_insight(df_train)
# stat_data(df_train)


df_train.drop_duplicates(['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day'], keep='first',
                         inplace=True)  # remove duplicates
df_train = df_train[(df_train.item_price > 0) & (df_train.item_price < 300000)]  # remove price outliers
df_train.date = pd.to_datetime(df_train.date)

# Check how things plot
# df_train_group = df_train.groupby(["date_block_num", "shop_id", "item_id"]).sum().reset_index()
#
# df_plot = df_train_group[(df_train_group.shop_id == 31) & (df_train_group.item_id == 5821)]
# plt.plot(df_plot["date_block_num"], df_plot["item_cnt_day"])
# plt.show()
#
# new_df = df_train_group.groupby(["shop_id", "item_id"]).count().reset_index()
# new_df.sort_values(by=['date_block_num'], ascending=False, inplace=True)
#
# print(new_df.head(4))

df_train_pivot = pd.pivot_table(df_train, values=['item_cnt_day'], index=['shop_id', 'item_id'],
                                columns=['date_block_num'],
                                aggfunc=sum).fillna(0).reset_index()

df_train_pivot = df_train_pivot.merge()
