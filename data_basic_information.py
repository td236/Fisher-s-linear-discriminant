import pandas as pd
df = pd.read_csv('data_banknote_authentication.txt')

pd.options.display.float_format = "{:.1f}".format
print(df.describe())
print('Licznosc poszczegolnych klas: \n', df['class'].value_counts())