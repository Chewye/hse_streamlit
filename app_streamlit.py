# импортируем библиотеку streamlit
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

plt.style.use("dark_background")

df = pd.read_csv('df.csv')

st.header('Cource HSE "Linear models. EDA"')
st.subheader('')

st.write('Посмотрим на баланс целевого признака')
fig = plt.figure()
ax = sns.countplot(df.TARGET)
ax.set (xlabel='Наличие отклика',
 ylabel='Count',
 title='Отлклик на маркетинговую компанию')
st.pyplot(fig)

st.write(f"Соотношенеие целевого признака 1 к {int(df[df['TARGET']==0]['TARGET'].count() / df[df['TARGET']==1]['TARGET'].count())}")

numerical = ['AGE',
             'WORK_TIME',
             'PERSONAL_INCOME',
             'CREDIT',
             'TERM',
             'FST_PAYMENT']

cat_feat = ['OWN_AUTO',
            'FL_PRESENCE_FL',
            'CLOSED_FL',
            'DEPENDANTS',
            'FACT_ADDRESS_PROVINCE',
            'REG_ADDRESS_PROVINCE',
            'POSTAL_ADDRESS_PROVINCE',
            'GENDER',
            'JOB_DIR',
            'EDUCATION',
            'SOCSTATUS_PENS_FL',
            'GEN_INDUSTRY',
            'CHILD_TOTAL',
            'FAMILY_INCOME',
            'MARITAL_STATUS',
            'GEN_TITLE',
            'SOCSTATUS_WORK_FL']

add_selectbox = st.selectbox(
    "Choose type of feature",
    ("Numerical", "Categorical")
)
st.text(f'You choose {add_selectbox} type.')

if add_selectbox == 'Numerical':
    feature_choose = st.selectbox(
        'Now choose feature',
        numerical
    )
else:
    feature_choose = st.selectbox(
        'Now choose feature',
        cat_feat
    )

if add_selectbox == 'Categorical':
    st.write(f'Feature {feature_choose} has category {df[feature_choose].unique()}')

    fig = plt.figure()
    ax = sns.countplot(df[feature_choose])
    ax.set (xlabel=feature_choose,
    ylabel='Count',
    title=f'boxplot {feature_choose}')
    st.pyplot(fig)

    st.write(f'count in each category:')
    st.write(pd.DataFrame([[df[df[feature_choose]==cat].shape[0] for cat in df[feature_choose].unique()]], columns=df[feature_choose].unique()))
    

else:
    st.write(f'Feature {feature_choose} has range {df[feature_choose].min()} - {df[feature_choose].max()}')

    fig = plt.figure()
    ax = sns.histplot(df[feature_choose])
    ax.set (xlabel=feature_choose,
    ylabel='Count',
    title=f'histplot {feature_choose}')
    st.pyplot(fig)

    st.write(df[[feature_choose]].describe().T)


fig = plt.figure()
ax = sns.heatmap(df[numerical])
ax.set (xlabel=f'-',
ylabel='-',
title=f'heatmap')
st.pyplot(fig)