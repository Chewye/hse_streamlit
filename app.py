# импортируем библиотеку streamlit
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def metrics(y, predict):
    accur = round(accuracy_score(y, predict), 3)
    prec = round(precision_score(y, predict), 3)
    rec = round(recall_score(y, predict), 3)
    f1 = round(f1_score(y, predict), 3)

    return pd.DataFrame({'Accuracy': accur, 'Precision': prec, 'Recall': rec, 'f1_score': f1}, index=['score'])


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
    'WORK_TIME': 'время работы на текущем месте (в месяцах)- взят логарифм ',
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

st.write(f'Количество пропусков по признакам:')
st.write(df.isna().sum().to_frame().T)
st.write(f'Количество дубликатов: {df.duplicated().sum()}')


st.write('Баланс целевого признака')
fig = plt.figure()
ax = sns.countplot(df.TARGET)
ax.set (xlabel='Наличие отклика',
 ylabel='Count',
 title='Отлклик на маркетинговую компанию')
st.pyplot(fig)

st.write(f"Соотношенеие целевого признака 1 к {int(df[df['TARGET']==0]['TARGET'].count() / df[df['TARGET']==1]['TARGET'].count())}")
st.subheader('Распределение признаков и их оценки')

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

add_selectbox = st.sidebar.selectbox(
    "Выберите тип признака (непрерывный, категориальный)",
    ("Numerical", "Categorical")
)


if add_selectbox == 'Numerical':
    feature_choose = st.sidebar.selectbox(
        'Выберите признак для его анализа',
        numerical
    )
else:
    feature_choose = st.sidebar.selectbox(
        'Выберите признак его анализа',
        cat_feat
    )


st.text(f'Вы выбрали признак {feature_choose} из {add_selectbox} типа.')
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
    if df[stats.zscore(df[feature_choose]) > 3].shape[0] > 0:
        st.write(f'от {df[stats.zscore(df[feature_choose]) > 3][feature_choose].min()}')

st.subheader('График зависимости целевой переменной от признака')


if add_selectbox == 'Categorical':
    fig = plt.figure()
    ax = sns.countplot(data=df, x='TARGET', hue=feature_choose)
    ax.set (xlabel='TARGET',
    ylabel='Count',
    title=f'Зависимость целевой переменной от {feature_choose}')
    st.pyplot(fig)    
    
    st.write(f'Таблица зависимости (количество) целевой переменной (колонки) от признака {feature_choose}')
    st.write(pd.DataFrame(\
        data=[[df[(df['TARGET']==j)&(df[feature_choose]==i)].shape[0] for j in df['TARGET'].unique()] for i in df[feature_choose].unique()], \
             index=df[feature_choose].unique() ,columns=df['TARGET'].unique()), index=df[feature_choose].unique())


    st.write(f'Таблица зависимости (проценты) целевой переменной (колонки) от признака {feature_choose}')
    st.write(pd.DataFrame(\
        data=[[f"{round(df[(df['TARGET']==j)&(df[feature_choose]==i)].shape[0] / df.shape[0]*100, 2)}%" for j in df['TARGET'].unique()] for i in df[feature_choose].unique()], \
             index=df[feature_choose].unique() ,columns=df['TARGET'].unique()), index=df[feature_choose].unique())

else:
    fig = plt.figure()
    ax = sns.histplot(data=df, x=feature_choose, hue='TARGET' , kde=True)
    ax.set (xlabel=feature_choose,
    ylabel='Count',
    title=f'histplot')
    st.pyplot(fig)  
    st.write(f"Корреляция целевого признака и {feature_choose}: {round(df[['TARGET', feature_choose]].corr().iloc[0, 1], 3)}") 




st.subheader('Тепловая карта')
options = st.multiselect(
    'Выберите признаки для генерации тепловой карты',
    numerical + cat_feat + ['TARGET'],
    ['TARGET', 'OWN_AUTO', 'PERSONAL_INCOME', 'GENDER', 'FAMILY_INCOME', 'CHILD_TOTAL'])



fig = plt.figure()
ax = sns.heatmap(df[options].corr(), annot=True)
ax.set (title=f'heatmap')
st.pyplot(fig)


df_model = pd.read_csv('df_model.csv')  #загруска датасета для модели
X_train, X_test, y_train, y_test = train_test_split(df_model.drop(['TARGET', 'AGREEMENT_RK', 'ID_CLIENT', 'ID_LOAN'], axis=1), df['TARGET'], test_size=0.2, random_state=42)

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

st.subheader('Модель линейной классификации')

threshold = st.slider('Выберите порог предсказания', 0.0, 1.0, 0.01)

predict_prob = model.predict_proba(X_test)


st.write(metrics(y_test, np.where(predict_prob[:, 1] < threshold, 0, 1)))



id_client = st.number_input(
        f"Выберите клиента банка по ID и наша модель сделает прогноз отклика. От {df_model['ID_CLIENT'].min()} до {df_model['ID_CLIENT'].max()}",
        106804370
    )
if id_client >= df_model['ID_CLIENT'].min() and id_client <= df_model['ID_CLIENT'].max():
    st.write(f"Вероятность отклика  клиента с ID {id_client} = \
              {round(model.predict_proba(df_model[df_model['ID_CLIENT']==id_client].drop(['TARGET', 'AGREEMENT_RK', 'ID_CLIENT', 'ID_LOAN'], axis=1))[0, 1], 3)}")
else: 
    st.write(f"Клиента с ID {id_client} не существует")