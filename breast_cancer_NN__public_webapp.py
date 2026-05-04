# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:20:39 2026

@author: chandan
"""

import pickle
import numpy as np
import streamlit as st

# loading the trained dada & loaded StandardScaler functions file from local device

from tensorflow.keras.models import load_model

loaded_NN_model = load_model("breast_cancer_model.h5", compile=False)
loaded_scaler = pickle.load(open('scaler.sav', 'rb'))

def breast_cancer(input_data):
    
  # 2 change input data to np array
    input_data_as_np_array = np.asarray(input_data)

  # 3 reshaping the np array
    input_data_reshaped = input_data_as_np_array.reshape(1, -1)

  # 4  standardization
    std_data = loaded_scaler.transform(input_data_reshaped)

    # prediction
    prediction = loaded_NN_model.predict(std_data)
    print(prediction[0])
    print('---------------------------------\n')
    # labels
    prediction_label = [np.argmax(prediction)]
    if (prediction_label[0] == 0 ):
      return (prediction_label[0], 'The Breast Cancer Tumor is Malignant' )
    else:
      return (prediction_label[0], 'The Breast Cancer Tumor is Benign' )

def main():
    
    # giving a title for streamlit app
    st.title('BREAST CANCER PREDICTION')
    
    # getting the input from the user using columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity, mean concave points
    # mean symmetry, mean fractal dimension, radius error, texture error, perimeter error, area error
    # smoothness error, compactness error, concavity error, concave points error, symmetry error, fractal dimension error
    # worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst compactness
    # worst concavity, worst concave points, worst symmetry, worst fractal dimension
    
    with col1:
        mean_radius = st.number_input(' mean radius')
    with col2:
        mean_texture = st.number_input(' mean texture')
    with col3:
        mean_perimeter = st.number_input(' mean perimeter')
    with col4:
        mean_area = st.number_input(' mean area')
    with col5:
        mean_smoothness = st.number_input('mean smoothness')
    with col1:
        mean_compactness = st.number_input(' mean compactness')
    with col2:
        mean_concavity = st.number_input(' mean concavity')
    with col3:
        mean_concave_points = st.number_input(' mean concave points')
    with col4:
        mean_symmetry = st.number_input(' mean symmetry')
    with col5:
        mean_fractal_dimension = st.number_input(' mean fractal dimension')
    with col1:
        radius_error = st.number_input(' radius error')
    with col2:
        texture_error = st.number_input(' texture error')
    with col3:
        perimeter_error = st.number_input(' perimeter error')
    with col4:
        area_error = st.number_input(' area error')
    with col5:
        smoothness_error = st.number_input(' smoothness error')
    with col1:
        compactness_error = st.number_input(' compactness error')
    with col2:
        concavity_error = st.number_input('concavity error')
    with col3:
        concave_points_error = st.number_input('concave points error')
    with col4:
        symmetry_error = st.number_input(' symmetry error')
    with col5:
        fractal_dimension_error = st.number_input('fractal dimension error')
    with col1:
        worst_radius = st.number_input(' worst radius')
    with col2:
        worst_texture = st.number_input(' worst texture')
    with col3:
        worst_perimeter = st.number_input(' worst perimeter')
    with col4:
        worst_area = st.number_input(' worst area')
    with col5:
        worst_smoothness = st.number_input(' worst smoothness')
    with col1:
        worst_compactness = st.number_input(' worst compactness')
    with col2:
        worst_concavity = st.number_input(' worst concavity')
    with col3:
        worst_concave_points = st.number_input(' worst concave points')
    with col4:
        worst_symmetry = st.number_input(' worst symmetry')
    with col5:
        worst_fractal_dimension = st.number_input(' worst fractal dimension')

    # code for prediction
    prediction = ''
    
    # button for prediction result
    if st.button('RESULT FOR BREAST CANCER'):
        prediction = breast_cancer([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension])
    
    st.success(prediction)
    
if __name__ == '__main__':
    main()
# 25250  Rajhansh










