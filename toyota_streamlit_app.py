#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_data
social_acc = ['About', 'Kaggle', 'Medium', 'LinkedIn']
social_acc_nav = st.sidebar.selectbox('About', social_acc)
if social_acc_nav == 'About':
    st.sidebar.markdown("<h2 style='text-align: center;'> Salami Lukman Bayonle</h2> ", unsafe_allow_html=True)
    st.sidebar.markdown('''---''')
    st.sidebar.markdown('''
    • Data Analytics/Scientist (Python/R/SQL/Tableau) \n 
    • Maintenance Specialist (Nigerian National Petroleum Company Limited) \n 
    • IBM/GOOGLE/DATACAMP Certified Data Analyst and Data Scientist''')
    st.sidebar.markdown("[ Visit Github](https://github.com/bayonlelukmansalami)")

elif social_acc_nav == 'Kaggle':
    st.sidebar.image('kaggle.jpg')
    st.sidebar.markdown("[Kaggle](https://www.kaggle.com/bayonlesalami)")

elif social_acc_nav == 'Medium':
    st.sidebar.image('medium.jpg')
    st.sidebar.markdown("[Click to read my blogs](https://medium.com/@bayonlelukmansalami/)")

elif social_acc_nav == 'LinkedIn':
    st.sidebar.image('linkedin.jpg')
    st.sidebar.markdown("[Visit LinkedIn account](https://www.linkedin.com/in/salamibayonlelukman/)")

loaded_model = joblib.load('models-toyota-price.pkl')


st.title('Toyota Used Car Price Prediction Web App')
st.write('This is a web app to predict the toyota used car price in pounds based on        several features that you can see in the sidebar. Please adjust the        value of each feature. After that, click on the Predict button at the bottom to        see the prediction of the regressor.')


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

model_Auris = st.sidebar.slider(label ='model_ Auris', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
    
model_Avensis = st.sidebar.slider(label ='model_ Avensis', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_Aygo = st.sidebar.slider(label ='model_ Aygo', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_C_HR = st.sidebar.slider(label ='model_ C-HR', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_Camry = st.sidebar.slider(label ='model_ Camry', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_Corolla = st.sidebar.slider(label ='model_ Corolla', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_GT86 = st.sidebar.slider(label ='model_ GT86', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_Hilux = st.sidebar.slider(label ='model_ Hilux', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_IQ = st.sidebar.slider(label ='model_ IQ', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_Land_Cruiser = st.sidebar.slider(label ='model_ Land Cruiser', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_PROACE_VERSO = st.sidebar.slider(label ='model_ PROACE VERSO', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_Prius = st.sidebar.slider(label ='model_ Prius', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_RAV4 = st.sidebar.slider(label ='model_ RAV4', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_Supra = st.sidebar.slider(label ='model_ Supra', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_Urban_Cruiser = st.sidebar.slider(label ='model_ Urban Cruiser', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_Verso = st.sidebar.slider(label ='model_ Verso', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_Verso_S = st.sidebar.slider(label ='model_ Verso-S', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)
model_Yaris = st.sidebar.slider(label ='model_ Yaris', min_value = 0,max_value = 1 ,value = 0,
                                           step = 1)



transmission_Automatic = st.sidebar.slider(label ='transmission_Automatic', min_value = 0,
                        max_value = 1 ,
                        value = 0,
                        step = 1)
transmission_Manual = st.sidebar.slider(label ='transmission_Manual', min_value = 0,
                        max_value = 1 ,
                        value = 0,
                        step = 1)
transmission_Other = st.sidebar.slider(label ='transmission_Other', min_value = 0,
                        max_value = 1 ,
                        value = 0,
                        step = 1)

transmission_Semi_Auto = st.sidebar.slider(label ='transmission_Semi-Auto', min_value = 0,
                        max_value = 1 ,
                        value = 0,
                        step = 1)

year = st.sidebar.slider(label = 'year', min_value = 1997,
                        max_value = 2022 ,
                        value = 2020,
                        step = 1)
fuelType_Diesel = st.sidebar.slider(label ='fuelType_Diesel', min_value = 0,
                        max_value = 1 ,
                        value = 0,
                        step = 1)
fuelType_Hybrid = st.sidebar.slider(label ='fuelType_Hybrid', min_value = 0,
                        max_value = 1 ,
                        value = 0,
                        step = 1)

fuelType_Other = st.sidebar.slider(label ='fuelType_Other', min_value = 0,
                        max_value = 1 ,
                        value = 0,
                        step = 1)
fuelType_Petrol = st.sidebar.slider(label ='fuelType_Petrol', min_value = 0,
                        max_value = 1 ,
                        value = 0,
                        step = 1)

engineSize = st.number_input('Enter Engine Size  (range = 0 - 7)')

mileage = st.number_input('mileage  (range = 0 - 100000)')

mpg = st.number_input('mpg  (range = 0 - 100000)')

tax = st.number_input('tax  (range = 0 - 1000)')




features = [year, mileage, tax, mpg, engineSize, model_Auris, model_Avensis, model_Aygo, model_C_HR, model_Camry,
            model_Corolla, model_GT86, model_Hilux, model_IQ, model_Land_Cruiser, model_PROACE_VERSO, model_Prius,
            model_RAV4, model_Supra, model_Urban_Cruiser, model_Verso, model_Verso_S, model_Yaris,
            transmission_Automatic, transmission_Manual, transmission_Other, transmission_Semi_Auto, fuelType_Diesel,
            fuelType_Hybrid, fuelType_Other, fuelType_Petrol]
            
features_np  = np.array([features])

st.table(features_np)


if st.button('Predict'):
    prediction = loaded_model.predict(features_np)
    st.write('Toyota Used Price Prediction is GBP {:.0f}'.format(np.round(prediction[0])))






