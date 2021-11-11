import pandas as pd
df = pd.read_csv('data_banknote_authentication.txt')
df.sample(frac=1).to_csv("data_shaked.csv", index=False)
