import pandas as pd
df = pd.read_csv('data_banknote_authentication.txt')

authentic = df[df['class'] == 0]  # class 0
fake = df[df['class'] == 1]  # class 1

print(df.corr("pearson"))
print()
print("Fake")
print(authentic.corr("pearson"))
print()
print("Authentic")
print(fake.corr("pearson"))