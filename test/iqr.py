import pandas as pd

train=pd.read_csv('wandb_export_2023-03-07T16_32_50.767+09_00.csv')
q1=train['firm-oath-54 - client_train_time'].quantile(0.25)
q3=train['firm-oath-54 - client_train_time'].quantile(0.75)
iqr=q3-q1

print(train['firm-oath-54 - client_train_time'].describe())

print(iqr)
print(q3+iqr*1)