import pandas as pd

cols_1 = ['X', 'Y']
cols_2 = ['X', 'Z']

Tab_1 = pd.read_csv("example_1.csv", header=1, names=cols_1)
Tab_2 = pd.read_csv("example_2.csv", header=1, names=cols_2)

result = pd.merge(Tab_1, Tab_2, on = ['X'], right_index=False, how='left', sort=False);