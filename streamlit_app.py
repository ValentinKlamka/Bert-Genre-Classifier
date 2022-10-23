import streamlit as st
import pandas as pd
import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px

@st.experimental_singleton
def load_model(model_path):
    model =tf.saved_model.load(model_path, options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost'))
    return model

@st.experimental_singleton
def predict(input,model_path):
    model=load_model(model_path)
    prediction = model(tf.constant([input]))  
    return prediction

if __name__ == "__main__":

    if "load_state" not in st.session_state:
        st.session_state.load_state = False

    model_path = "./genre_bert"
    model = load_model(model_path)
    
    
    st.write("Write a movie description and the app will try to guess the genre:")
    labeldict={0: ' drama ', 1: ' thriller ', 2: ' adult ', 3: ' documentary ', 4: ' comedy ', 5: ' crime ', 6: ' reality-tv ', 7: ' horror ', 8: ' sport ', 9: ' animation ', 10: ' action ', 11: ' fantasy ', 12: ' short ', 13: ' sci-fi ', 14: ' music ', 15: ' adventure ', 16: ' talk-show ', 17: ' western ', 18: ' family ', 19: ' mystery ', 20: ' history ', 21: ' news ', 22: ' biography ', 23: ' romance ', 24: ' game-show ', 25: ' musical ', 26: ' war '}

    movie_description = st.text_input("Movie description")
    
        
    st.write(movie_description)

    if st.button("Predict") or st.session_state.load_state:
        st.session_state.load_state = True
        
        prediction = predict(movie_description,model_path)
        #make an interactive plot with the prediction

        st.session_state.load_state = True
        df = pd.DataFrame(prediction[0], columns=["probability"])
        df["genre"] = labeldict.values()
        fig = px.bar(df, x="genre", y="probability", color="probability", title="Prediction")
        st.plotly_chart(fig)

        #checkbox where I can select multiple genres
        #genres with a probability higher than 0.9 are selected by default
        #the genres are sorted by probability
        genres = st.multiselect("Select genres", df.sort_values(by="probability", ascending=False)["genre"].values, default=df[df["probability"]>0.9]["genre"].values)

        st.write(st.session_state.load_state)
        st.write(genres)
