import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## load ths lstm model
model=load_model('next_word_lstm.h5')


## load the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

## function to predict next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list= token_list[-(max_sequence_len-1):] ## ensure that sequence len matches
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


## streamlit app
st.title("Next Word Prediction using LSTM RNN and Early Stopping")
input_text = st.text_input("Enter a partial sentence:", "To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1 #Retreive the   max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Predicted Next Word: {next_word}")

