import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

D_clients = pd.read_csv('https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/D_clients.csv')
D_close_loan = pd.read_csv('https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/D_close_loan.csv')
D_job = pd.read_csv('https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/D_job.csv')
D_last_credit = pd.read_csv('https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/D_last_credit.csv')
D_loan = pd.read_csv('https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/D_loan.csv')
D_pens = pd.read_csv('https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/D_pens.csv')
D_salary = pd.read_csv('https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/D_salary.csv')
D_target = pd.read_csv('https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/D_target.csv')
D_work = pd.read_csv('https://raw.githubusercontent.com/aiedu-courses/stepik_linear_models/main/datasets/D_work.csv')


df_list = [D_clients, D_close_loan, D_job, D_last_credit, D_loan, D_pens, D_salary, D_target, D_work]
D_clients.rename(columns={'ID': 'ID_CLIENT'}, inplace=True)

df = D_target.merge(D_clients, how='inner', on='ID_CLIENT')
df = df.merge(D_job, how='inner', on='ID_CLIENT')
df = df.merge(D_salary, how='inner', on='ID_CLIENT')
df = df.merge(D_last_credit, how='inner', on='ID_CLIENT')
df = df.merge(D_loan, how='inner', on='ID_CLIENT')
df = df.merge(D_close_loan, how='inner', on='ID_LOAN')

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df['WORK_TIME'] = df['WORK_TIME'].apply(lambda x: np.log(x))

numerical = ['AGE',
             'WORK_TIME',
             'PERSONAL_INCOME',
             'CREDIT',
             'TERM',
             'FST_PAYMENT']

id_list = ['AGREEMENT_RK',
           'ID_CLIENT',
           'TARGET',
           'ID_LOAN']

cat = list(df.columns)
cat = list(set(cat) - set(numerical) - set(id_list))


#Заменим выбросы на пороговое значение выбросов
for feat in numerical:
    numb = df[stats.zscore(df[feat])>3][feat].min()
    if numb > 0:
        df[feat] = df[feat].where(df[feat] < numb, numb)

scaler = MinMaxScaler()
scaler.fit(df[numerical])
df[numerical] = scaler.transform(df[numerical])

scaler_id = MinMaxScaler()
scaler_id.fit(df[['AGREEMENT_RK', 'ID_CLIENT', 'ID_LOAN']])
df[['AGREEMENT_RK', 'ID_CLIENT', 'ID_LOAN']] = scaler_id.transform(df[['AGREEMENT_RK', 'ID_CLIENT', 'ID_LOAN']])


df.to_csv('./df_prepare.csv', index=False)