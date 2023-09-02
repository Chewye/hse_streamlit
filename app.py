# импортируем библиотеку streamlit
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats


plt.style.use("dark_background")

feat_dict = {
    'AGE': 'возраст клиента',
    'GENDER': 'пол клиента (1 — мужчина, 0 — женщина)',
    'EDUCATION': 'образование',
    'MARITAL_STATUS': 'семейное положение',
    'CHILD_TOTAL':'количество детей клиента',
    'DEPENDANTS': 'количество иждивенцев клиента',
    'SOCSTATUS_WORK_FL': 'социальный статус клиента относительно работы (1 — работает, 0 — не работает)',
    'SOCSTATUS_PENS_FL': 'социальный статус клиента относительно пенсии (1 — пенсионер, 0 — не пенсионер)',
    'REG_ADDRESS_PROVINCE': 'область регистрации клиента',
    'FACT_ADDRESS_PROVINCE': 'область фактического пребывания клиента',
    'POSTAL_ADDRESS_PROVINCE': 'почтовый адрес области',
    'FL_PRESENCE_FL': 'наличие в собственности квартиры (1 — есть, 0 — нет)',
    'OWN_AUTO': 'количество автомобилей в собственности',
    'AGREEMENT_RK': 'уникальный идентификатор объекта в выборке',
    'TARGET': 'целевая переменная: отклик на маркетинговую кампанию (1 — отклик был зарегистрирован, 0 — отклика не было)',
    'GEN_INDUSTRY': 'отрасль работы клиента',
    'GEN_TITLE': 'должность',
    'JOB_DIR': 'направление деятельности внутри компании',
    'WORK_TIME': 'время работы на текущем месте (в месяцах)',
    'FAMILY_INCOME': 'семейный доход (несколько категорий)',
    'PERSONAL_INCOME': 'личный доход клиента (в рублях)',
    'CREDIT': 'сумма последнего кредита клиента (в рублях)',
    'TERM': 'срок кредита',
    'FST_PAYMENT': 'первоначальный взнос (в рублях)',
    'ID_LOAN': 'идентификатор кредита',
    'CLOSED_FL': 'текущий статус кредита (1 — закрыт, 0 — не закрыт)'
}

df = pd.read_csv('df.csv')

st.header('Cource HSE "Linear models. EDA"')
st.subheader('')

st.write('Баланс целевого признака')
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
    "Выберите тип признака (непрерывный, категориальный)",
    ("Numerical", "Categorical")
)
st.text(f'You choose {add_selectbox} type.')

if add_selectbox == 'Numerical':
    feature_choose = st.selectbox(
        'Выберите признак',
        numerical
    )
else:
    feature_choose = st.selectbox(
        'Выберите признак',
        cat_feat
    )

st.write(feat_dict[feature_choose])

if add_selectbox == 'Categorical':
    st.write(f'Feature {feature_choose} has category {df[feature_choose].unique()}')

    fig = plt.figure()
    ax = sns.countplot(df[feature_choose])
    ax.set (xlabel=feature_choose,
    ylabel='Count',
    title=f'countplot')
    st.pyplot(fig)

    st.write(f'Количество в каждой категории:')
    st.write(pd.DataFrame([[df[df[feature_choose]==cat].shape[0] for cat in df[feature_choose].unique()]], columns=df[feature_choose].unique()))
    

else:
    st.write(f'Признак {feature_choose} имеет диапозон {df[feature_choose].min()} - {df[feature_choose].max()}')

    fig = plt.figure()
    ax = sns.histplot(df[feature_choose], kde=True)
    ax.set (xlabel=feature_choose,
    ylabel='Count',
    title=f'histplot')
    st.pyplot(fig)

    st.write(df[[feature_choose]].describe().T)

    fig = plt.figure()
    ax = sns.boxplot(df[feature_choose])
    ax.set (xlabel=feature_choose,
    title=f'boxplot')
    st.pyplot(fig)

    st.write(f'Предварительные выбросы:')
    if df[stats.zscore(df[feature_choose])< -3].shape[0] > 0:
        st.write(f'до {df[stats.zscore(df[feature_choose])< -3][feature_choose].max()}')
    if df[stats.zscore(df[feature_choose]) > 3].shape[0] > 0:
        st.write(f'от {df[stats.zscore(df[feature_choose]) > 3][feature_choose].min()}')


options = st.multiselect(
    'Выберите признаки для генерации тепловой карты',
    numerical + cat_feat + ['TARGET'],
    ['TARGET', 'OWN_AUTO', 'PERSONAL_INCOME', 'GENDER', 'FAMILY_INCOME', 'CHILD_TOTAL'])



fig = plt.figure()
ax = sns.heatmap(df[options].corr(), annot=True)
ax.set (title=f'heatmap')
st.pyplot(fig)
