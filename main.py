from Model import predict,model,dictionary
import streamlit as st
st.title("Email Classification")
text = st.text_input("Enter your email content :")
button = st.button("Classify")
if button:
    output = predict(text,model,dictionary)
    st.text(f"Model Prediction:{output}")