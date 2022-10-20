import streamlit as st
import tensorflow as tf
import numpy as np
from helper_functions import speech_predict


# @st.cache(allow_output_mutation=True)
@st.experimental_singleton
def get_model():
    model = tf.keras.models.load_model('Model')
    model.make_predict_function() 
    return model
    
model = get_model()

st.header("Predict the speaker's party - Labour or Conservative?")
st.write('You can find some speeches at:')
st.write('\
https://www.ukpol.co.uk, \
https://labour.org.uk/category/latest/, \
https://www.conservatives.com/news')
speech = st.text_area("Enter the speech here then press 'Make prediction' below: ")

if st.button('Make prediction'):

    pred = speech_predict(speech, model, overlap = 90)

    threshold = 0.5

    if pred < threshold:
        prediction = 'Labour'
        prob = 1- pred
    else:
        prediction = 'Conservative'
        prob = pred
    st.write(f'*We predict this speech was given by a*  **{prediction}**  *speaker*.')
    st.write(f'Probability of accuracy:  {prob*100:.0f} %')