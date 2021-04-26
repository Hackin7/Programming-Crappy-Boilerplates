import numpy as np
import pandas as pd

### Opening File ##############################################################
df = pd.DataFrame(
    [[4, 7, 10],
    [5, 8, 11],
    [6, 9, 12]],
    index=[1, 2, 3],
    columns=['a', 'b', 'c'])

#pd.read_csv("train.csv")

df.head()

##### Read Data #######################################################
df.columns
for col in df:
    print(col)

### Rows
for index, row in df.iterrows():
    print(index,row)


### Updating Data
df['c'] = [1,2,3]
print(df.head())
