import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('data_banknote_authentication.txt')
sns.pairplot(df, hue='class') 

plt.show()