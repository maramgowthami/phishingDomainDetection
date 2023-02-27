import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
st.write(""" # Phishing website Prediction""")
columns=[
'qty_dot_directory' ,
'qty_underline_directory',
'qty_slash_directory',
'qty_questionmark_directory',
'qty_equal_directory',
'qty_at_directory',
'qty_and_directory',
'qty_exclamation_directory',
'qty_space_directory',
'qty_tilde_directory',
'qty_comma_directory',
'qty_plus_directory',
'qty_asterisk_directory',
'qty_hashtag_directory',
'qty_dollar_directory',
'directory_length',
'qty_dot_file',
'qty_hyphen_file',
'qty_underline_file',
'qty_questionmark_file',
'qty_equal_file',
'qty_at_file',
'qty_and_file',
'qty_exclamation_file',
'qty_space_file',
'qty_tilde_file',
'qty_comma_file',
'qty_plus_file',
'qty_asterisk_file',
'qty_hashtag_file',
'qty_dollar_file',
'qty_slash_url',
'length_url'
]
url=st.text_input("Upload the url",value="https://internship.ineuron.ai/project/board/Phishing-Domain-Detection/62c5c56a66bce102bb9d46e7")

def prepare(url):
    url=url.split("/")
    directory="/".join(url[3:-1])
    domain=url[2]
    le=url[-1].split("?")
    file=le[0]
    parameter=""
    if(len(le)>=2):
        parameter=le[1]
    l=[]
    l.append(directory.count('.')-1)
    l.append(directory.count('_')-1)
    l.append(directory.count('/')-1)
    l.append(directory.count('?')-1)
    l.append(directory.count('=')-1)
    l.append(directory.count('@')-1)
    l.append(directory.count('&')-1)
    l.append(directory.count('!')-1)
    l.append(directory.count(' ')-1)
    l.append(directory.count('~')-1)
    l.append(directory.count(',')-1)
    l.append(directory.count('+')-1)
    l.append(directory.count('*')-1)
    l.append(directory.count('#')-1)
    l.append(directory.count('$')-1)
    l.append(len(directory))
    l.append(file.count('.')-1)
    l.append(file.count('-')-1)
    l.append(file.count('_')-1)
    l.append(file.count('?')-1)
    l.append(file.count('=')-1)
    l.append(file.count('@')-1)
    l.append(file.count('&')-1)
    l.append(file.count('!')-1)
    l.append(file.count(' ')-1)
    l.append(file.count('~')-1)
    l.append(file.count(',')-1)
    l.append(file.count('+')-1)
    l.append(file.count('*')-1)
    l.append(file.count('#')-1)
    l.append(file.count('$')-1)
    l.append(url.count('/')-1)
    l.append(len(url))
    return l

l=prepare(url)
l=np.array(l)
l=l.reshape(1,33)

df=pd.DataFrame(l,index=[1],columns=columns)

load_model = pickle.load(open('random_forest2', 'rb'))

prediction = load_model.predict(df)
prediction_proba = load_model.predict_proba(df)

st.subheader('Prediction Probability')
st.write(prediction_proba)






