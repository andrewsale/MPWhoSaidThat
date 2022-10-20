import streamlit as st
import tensorflow as tf
import numpy as np

def speech_predict(speech, model, overlap = 50):
    split_speech = speech.split()
    length = len(split_speech)
    if length < 100:
        segments = [speech]
    else:
        segments = []
        i=0
        while i <= length - 100:
            segments.append(' '.join(split_speech[i:i+100]))
            i+=(100-overlap)
    preds = model.predict(segments, verbose=0)
    pred = np.sum(preds)/len(preds)
    return pred

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