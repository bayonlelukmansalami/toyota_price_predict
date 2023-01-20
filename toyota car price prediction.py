import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn import linear_model
from sklearn.utils import resample
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV



# import data file csv
data = pd.read_csv('toyota.csv')
df = data.copy()
df = pd.get_dummies(df, columns=["model", "transmission","fuelType"], drop_first=False)

# set page title
st.set_page_config('Toyota Price Prediction App')

st.title('Predict Toyota Car Prices (in Pounds)')
social_acc = ['About', 'Kaggle', 'Medium', 'LinkedIn']
social_acc_nav = st.sidebar.selectbox('About', social_acc)
if social_acc_nav == 'About':
    st.sidebar.markdown("<h2 style='text-align: center;'> Sarvesh Kishor Talele</h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown('''
    • Data Analytics (Python/SQL/Tableau) \n 
    • Industrial Robotics (KUKA Robots) \n 
    • Interned as a Data Engineer''')
    st.sidebar.markdown("[ Visit Google Scholar Account](https://scholar.google.com/citations?user=-4Vyig8AAAAJ&hl=en)")

elif social_acc_nav == 'Kaggle':
    st.sidebar.image('kaggle.jpg')
    st.sidebar.markdown("[Kaggle](https://www.kaggle.com/bayonlesalami)")

elif social_acc_nav == 'Medium':
    st.sidebar.image('medium.jpg')
    st.sidebar.markdown("[Click to read my blogs](https://medium.com/@bayonlelukmansalami/)")

elif social_acc_nav == 'LinkedIn':
    st.sidebar.image('linkedin.jpg')
    st.sidebar.markdown("[Visit LinkedIn account](https://www.linkedin.com/in/salamibayonlelukman/)")

menu_list = ["Predict Price"]
menu = st.radio("Menu", menu_list)

if menu == 'Predict Price':

    model_dic = {'model_ Auris': 0, 'model_ Avensis': 1, 'model_ Aygo': 2, 'model_ C-HR': 3, 'model_ Camry': 4, 'model_ Corolla': 5, 'model_ GT86': 6,
                 'model_ Hilux': 7, 'model_ IQ': 8, 'model_ Land Cruiser': 9, 'model_ PROACE VERSO': 10, 'model_ Prius': 11, 'model_ RAV4': 12,
                 'model_ Supra': 13, 'model_ Urban Cruiser': 14, 'model_ Verso': 15, 'model_ Verso-S': 16, 'model_ Yaris': 17}
    
    transmission_dic = {'transmission_Automatic': 0, 'transmission_Manual': 1, 'transmission_Other': 2, 'transmission_Semi-Auto': 3}
    fuel_dic = {'fuelType_Diesel': 0, 'fuelType_Hybrid': 1, 'fuelType_Other': 2, 'fuelType_Petrol': 3}

    model_list = [
        'model_ Auris',
       'model_ Avensis', 'model_ Aygo', 'model_ C-HR', 'model_ Camry',
       'model_ Corolla', 'model_ GT86', 'model_ Hilux', 'model_ IQ',
       'model_ Land Cruiser', 'model_ PROACE VERSO', 'model_ Prius',
       'model_ RAV4', 'model_ Supra', 'model_ Urban Cruiser', 'model_ Verso',
       'model_ Verso-S', 'model_ Yaris']
    transmission_list = ['transmission_Automatic',
       'transmission_Manual', 'transmission_Other', 'transmission_Semi-Auto']
    fuel_list = ['fuelType_Diesel', 'fuelType_Hybrid', 'fuelType_Other',
       'fuelType_Petrol']

    year = st.slider("Enter the year", 1970, 2020)

    engine_size = st.number_input('Enter Engine Size  (range = 0 - 7)')

    model_choice = st.selectbox(label='Select your favourite Car Model', options=model_list)
    models = model_dic[model_choice]

    transmission_choice = st.selectbox(label=' Select the Transmission type', options=transmission_list)
    transmissions = transmission_dic[transmission_choice]

    fuel_choice = st.selectbox(label='Select the Fuel type', options=fuel_list)
    fuels = fuel_dic[fuel_choice]

    X = df.drop('price', axis=1)
    y = df.price

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    xgb = XGBRegressor(random_state=4)
    rf = RandomForestRegressor(random_state=4)

    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    pred_rf = rf.predict(X_test)

    xgb_score = r2_score(y_test,pred_xgb)
    rf_score = r2_score(y_test,pred_rf)
    column_data = X.columns.values


    def predict_price_xgb(model, _year, engineSize, transmission, fuel):
        try:
            model_index = model_list.index(model)[0][0]
            transmission_index = transmission_list.index(transmission)[0][0]
            fuel_index = fuel_list.index(fuel)[0][0]
        except ValueError:
            model_index = -1
            fuel_index = -1
            transmission_index = -1

        x = np.zeros(len(column_data))
        x[0] = _year
        x[1] = engineSize
        if transmission_index >= 0:
            x[transmission_index] = 1
        elif fuel_index >= 0:
            x[fuel_index] = 5
        elif model_index >= 0:
            x[model_index] = 9

        return xgb.predict([x])[0]

    def predict_price_rf(model, _year, engineSize, transmission, fuel):
        try:
            model_index = model_list.index(model)[0][0]
            transmission_index = transmission_list.index(transmission)[0][0]
            fuel_index = fuel_list.index(fuel)[0][0]
        except ValueError:
            model_index = -1
            fuel_index = -1
            transmission_index = -1

        x = np.zeros(len(column_data))
        x[0] = _year
        x[1] = engineSize
        if transmission_index >= 0:
            x[transmission_index] = 1
        elif fuel_index >= 0:
            x[fuel_index] = 5
        elif model_index >= 0:
            x[model_index] = 9

        return rf.predict([x])[0]


    alg = ['XGBoost Regression', 'RandomForest']
    select_alg = st.selectbox('Choose Algorithm for Efficient Predict', alg)
    if st.button('Predict'):
        if select_alg == 'XGBoost Regression':
            st.write('Accuracy Score', xgb_score)
            st.subheader(predict_price_xgb(models, year, engine_size, transmissions, fuels))
            st.markdown("<h5 style='text-align: left;'> Pounds </h5>", unsafe_allow_html=True)

        elif select_alg == 'RandomForest':
            st.write('Accuracy Score', rf_score)
            predicted_price = st.subheader(predict_price_rf(models, year, engine_size, transmissions, fuels))
            st.markdown("<h5 style='text-align: left;'> Pounds </h5>", unsafe_allow_html=True)
            
        

    quotes = ['Focus your attention on what is most important', 'Expect perfection (but accept excellence)',
              'Make your own rules',
              'Give more than you take', 'Leverage imbalance']
    quote_choice = random.choice(quotes)
    st.markdown("<h4 style='text-align: left;'> Quote of the Day </h4>", unsafe_allow_html=True)
    st.write(quote_choice)

