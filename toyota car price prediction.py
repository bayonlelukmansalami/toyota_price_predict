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
import xgboost
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import joblib
import pickle
loaded_model = joblib.load('models-toyota-price.pkl')


st.title('Toyota Used Car Price Prediction Web App')
st.write('This is a web app to predict the toyota used car price in pounds based on\
        several features that you can see in the sidebar. Please adjust the\
        value of each feature. After that, click on the Predict button at the bottom to\
        see the prediction of the regressor.')


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

year = st.sidebar.slider(label = 'year', min_value = 1997,
                        max_value = 2022 ,
                        value = 2020,
                        step = 1)

engine_size = st.number_input('Enter Engine Size  (range = 0 - 7)')

model_choice = st.selectbox(label='Select your favourite Car Model', options=model_list)
models = model_dic[model_choice]

transmission_choice = st.selectbox(label=' Select the Transmission type', options=transmission_list)
transmissions = transmission_dic[transmission_choice]

fuel_choice = st.selectbox(label='Select the Fuel type', options=fuel_list)
fuels = fuel_dic[fuel_choice]



features = {
  'toyota_model':model_list,
  'transmission':transmission_list,
  'fuel_type':fuel_list,
  'year':year,
  'engine_size':engine_size}
            
features_df  = pd.DataFrame([features])

st.table(features_df)


if st.button('Predict'):
    prediction = loaded_model.predict(features_df)
    st.write('Toyota Used Price Prediction is GBP {:.0f}'.format(np.round(result[0])))
        
        
run()
