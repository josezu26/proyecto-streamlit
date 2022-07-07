from email import header
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import pandas as pd
import numpy as np


header = st.container()
datasets = st.container()
features =st.container()
model_Training = st.container()

with header:
    st.title("Data science Canciones")
    st.text("Este es un proyecto de datos de canciones")

with datasets:
    st.markdown("## **📌 Dataset **")
    datos_data= pd.read_csv("data/canciones.csv")
    st.write(datos_data.head(10))

    st.subheader("Grafico")
    pulocation = pd.DataFrame(datos_data['artists'].value_counts()).head(50)
    st.bar_chart(pulocation)

with features:    
    st.markdown("## **📌 Caracteristicas **")
    

with model_Training:
    st.markdown("## **📌 Modelo **")
    sel_col, disp_col = st.columns(2)
    max_depth=sel_col.slider('Profundidad del modelo?', min_value=10, max_value=100, value=50)
    n_estimators = sel_col.selectbox('Cuantos arboles?',options=[10,50,100,'no limit'],index=0)
    input_feature = sel_col.text_input('Ingrese la caracteristica a predecir',"tempo")

    if n_estimators == 'no limit':
        ran = RandomForestRegressor(max_depth=max_depth)
    else:
        ran = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    
    X= datos_data[[input_feature]]
    y= datos_data[['tempo']]
    ran.fit(X,y)
    prediccion = ran.predict(y)

    disp_col.subheader('Error absoluto medio del modelo es: ')
    disp_col.write(mean_absolute_error(y,prediccion))

    disp_col.subheader('Error cuadratico del modelo es: ')
    disp_col.write(mean_squared_error(y,prediccion))

    disp_col.subheader('R cuadratico del modelo es: ')
    disp_col.write(r2_score(y,prediccion))

