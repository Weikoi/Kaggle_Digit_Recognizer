#%%
import pandas as pd
import os


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data_path = "./data/"
data_train = pd.read_csv(data_path + "train.csv")
data_test = pd.read_csv(data_path + "test.csv")

#%%
print(data_train.shape)
print(data_test.shape)