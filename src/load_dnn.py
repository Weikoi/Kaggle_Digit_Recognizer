from keras.models import load_model
from keras.models import load_model
import pickle as pk
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

network = load_model('./dnn_model.h5')

data_path = "../data/"

df = pd.read_csv(data_path + "train.csv")
df_test = pd.read_csv(data_path + "test.csv")

print(df_test)
print(df_test.columns)

# data_label = pk.load(
#     file=open(r'D:\PycharmProjects\19_s1\Graduation_Project\data\raw_data_4_dataset\label_data_list.bin', 'rb'))
# df_label = pd.DataFrame(data_label, columns=all_columns)
# X = df_label[columns]

result_columns = ["ID", "Label"]

df_test = df_test.values.reshape((28000, 784 * 1))
result = network.predict(df_test)

print(result)

print(len(result))
id_list = []
label_list = []

for idx1, i in enumerate(result):
    id_list.append(idx1 + 1)

    for idx, j in enumerate(i):
        if j > 0.5:
            label_list.append(idx)


print(len(id_list))
print(len(label_list))
result_dict = {"ID": id_list, "Label": label_list}
result_df = pd.DataFrame(result_dict)

print(result_df)

result_df.to_csv("submission.csv", index=False)
