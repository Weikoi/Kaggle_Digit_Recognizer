import pickle as pk
import pandas as pd
import sys
from xgboost import plot_importance
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data_path = "../data/"

df = pd.read_csv(data_path + "train.csv")
df_test = pd.read_csv(data_path + "test.csv")

# %%
columns = list(df.columns)[1:]
print(columns)
# %%

print("================== 正在加载数据集 ==================")

X = df[columns]
y = df["label"]

print("================== 正在构建数据特征 ================")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=41)

# 网络构造
network = models.Sequential()
network.add(layers.Dense(units=784, activation='relu', input_shape=(784,)))
network.add(layers.Dense(units=600, activation='relu', input_shape=(784,)))
network.add(layers.Dense(units=400, activation='relu', input_shape=(600,)))
network.add(layers.Dense(units=200, activation='relu', input_shape=(400,)))
network.add(layers.Dense(units=100, activation='relu', input_shape=(200,)))
network.add(layers.Dense(units=50, activation='relu', input_shape=(100,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='Adam',
                loss='categorical_crossentropy', metrics=['accuracy'])

# 数据集准备
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %%
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_train = X_train.values.reshape((29400, 784 * 1))
X_test = X_test.values.reshape((12600, 784 * 1))

from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical

# lb = LabelBinarizer()
# y_train = lb.fit_transform(y_train)  # transfer label to binary value
# y_test = lb.fit_transform(y_test)  # transfer label to binary value

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# %%
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %%
# train
history = network.fit(X_train, y_train, epochs=100, shuffle=True, batch_size=128, validation_split=0.3)

# test
test_loss, test_acc = network.evaluate(X_test, y_test)
print('test_acc:', test_acc)

# model save
network.save("./dnn_model.h5")

# %%
history_dict = history.history

print(history_dict.keys())
print(history_dict.values())

# %%

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# # # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()

# %%
# plt.clf()  # clear figure

fig, ax = plt.subplots(figsize=(8, 8))
plt.plot(epochs, acc, '--', color='C4', label='Training acc')
plt.plot(epochs, val_acc, '--', color="red", label='Validation acc')
# Add a vertical line, here we set the style in the function call
# ax.axhline(test_acc, ls='--', color='g')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
