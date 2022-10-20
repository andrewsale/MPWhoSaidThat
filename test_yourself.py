import pandas as pd
import streamlit as st
import os

@st.cache(allow_output_mutation=True)
def initialize():
    df_samples = pd.read_csv('./Sampled_speeches/test.csv').sample(frac=1)
    record=[]
    return df_samples, record

df_samples, record = initialize()

top_container = st.container()
speech_container = st.container()
result_container = st.container()

i = len(record)

test_speech=df_samples.iloc[i,2]
test_party=df_samples.iloc[i,1]
test_speaker=df_samples.iloc[i,0]
test_label=df_samples.iloc[i,4]

top_container.header("Predict the speaker's party - Labour or Conservative - Test yourself!")

speech_container.subheader('Who said this?')
speech_container.write('"...'+test_speech+'..."')

election = speech_container.radio(
        'Speaker party:', ('Pass', 'Conservatives', 'Labour'),
        horizontal=True)

submit = result_container.button('Submit')

if submit:
    i+=1
    if election == 'Conservatives':
        election_label = 1
    elif election == 'Labour':
        election_label = 0
    else:
        election_label = -1
    if test_label == election_label:
        result_container.write('Congratulations! You got it right!')
        record.append(1)
    else:
        result_container.write("I'm sorry, you guessed wrong.")
        record.append(0)
    result_container.write(f'The speech was given by {test_speaker}, from the {test_party}.')
    result_container.write(f'Your record so far: {sum(record)} correct from {len(record)}\
         ({100*sum(record)/len(record):.0f}%)') 
    
    next = result_container.button('Next...')