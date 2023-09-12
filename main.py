import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import streamlit as st

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        else:
            return 25
    else:
        return Age

age = st.sidebar.number_input('Person age', min_value = 0, max_value = 80, value = 18, step = 1)
gender = st.sidebar.selectbox('Select gender of the person', ('Male', 'Female'))
Sibsp = st.sidebar.number_input('Number of siblings', min_value = 0, max_value = 8, value = 0, step = 1)
Parch = st.sidebar.number_input('Number of parents/children aboard', min_value = 0, max_value = 6, value = 1, step = 1)
Embarked = st.sidebar.selectbox('Select port of embarkation', ('Cherbourg', 'Queenstown', 'Southampton'))
Pclass = st.sidebar.selectbox('Select passenger class', (1,2,3))
Fare = st.sidebar.number_input('Input passenger fare', min_value = 0, max_value = 513,value = 0, step = 1)

if gender == 'Male':
    n_gender = 1
else: n_gender = 0

if Embarked == 'Cherbourg':
    Q, S = 0, 0
elif Embarked == 'Queenstown':
    Q, S = 1, 0

data = dict(
    Pclass = Pclass,
    Age = age,
    SibSp = Sibsp,
    Parch = Parch,
    Fare = Fare,
    male = n_gender,
    Q = Q,
    S = S
)

data_vis = dict(
    Pclass = Pclass,
    Age = age,
    Siblings = Sibsp,
    Parents_children = Parch,
    Fare = Fare,
    Gender = gender,
    Port = Embarked)

st.title('This app predicts whether person could have survived on Titanic or not')

dv = pd.DataFrame(data_vis, index = ['Information about person'])
df = pd.DataFrame(data, index = ['Information about person'])
st.dataframe(dv)
train_base = pd.read_csv('C:/Users/ADMIN/Downloads/train.csv')
train = train_base.copy()

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
train.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis = 1, inplace=True)
train.dropna(inplace = True)
sex = pd.get_dummies(train['Sex'], drop_first=True) #drop the sex, embarked, name and Ticket
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train.drop(['Sex', 'Embarked'], axis = 1, inplace = True)
train = pd.concat([train, sex, embark], axis=1) #check the head of dataframe
x = train.drop('Survived', axis=1).copy()
y = train['Survived']
scaler = RobustScaler()
# x.iloc[:,:] = scaler.fit_transform(x)
# df.iloc[:,:] = scaler.fit_transform(df)
logmodel = LogisticRegression(max_iter = 100000)
logmodel.fit(x, y)
prediction = logmodel.predict(df)
predictions_probabilities = logmodel.predict_proba(df)
max_prob = np.max(predictions_probabilities) * 100
predict = lambda a:'survived' if a==1 else 'did not survive'

st.markdown(f"The person presumably **{predict(prediction)}** with probability of **{round(max_prob, 2)}** %")
