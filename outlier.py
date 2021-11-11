import pandas as pd
df = pd.read_csv('data_banknote_authentication.txt')


def outliers(factor, df):
    print("factor: ", factor)
    for column in df:
        if column == "class":
            break
        columnSeriesObj = df[column]
        Q1 = columnSeriesObj.quantile(0.25)
        Q3 = columnSeriesObj.quantile(0.75)
        IQR = Q3 - Q1
        print('Colunm', column, "- ", end=" ")
        lower = Q1 - factor*IQR  # Standard value is 1.5
        upper = Q3 + factor*IQR
        outliers = ((columnSeriesObj < lower) |
                    (columnSeriesObj > upper)).sum()
        if column == "entropy":
            print(" ", end="")
        print("# outliers: ", outliers)


outliers(1.5, df)
outliers(2.5, df)
print()

authentic = df[df['class'] == 0]  # class 0
fake = df[df['class'] == 1]  # class 1

outliers(1.5, authentic)
outliers(2.0, authentic)
outliers(2.5, authentic)
print()

outliers(1.5, fake)
outliers(2.0, fake)
outliers(2.5, fake)
