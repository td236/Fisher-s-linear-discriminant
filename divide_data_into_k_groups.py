import pandas as pd
df = pd.read_csv('data_banknote_authentication.txt')
df = df.sample(frac=1).reset_index(drop=True)

k = 5

for i in range(k):
    df_test = df.iloc[int(i*len(df)/k):int((i+1)*len(df)/k), :]
    df_learning_1 = df.iloc[:int(i*len(df)/k), :]
    df_learning_2 = df.iloc[int((i+1)*len(df)/k):, :]
    df_learning = pd.concat([df_learning_1, df_learning_2], axis=0)
    df_test.to_csv("data_test_set_" + str(i) + ".csv", index=False)
    df_learning.to_csv("data_learning_set_" + str(i) + ".csv", index=False)
