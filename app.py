import streamlit as st

st.title("AI-Driven Smart Business Opportunity Analyzer")

st.write("Hello Mahmoud — your app is running successfully!")

uploaded_file = st.file_uploader("Upload Opportunity Excel/CSV", type=["xlsx", "xls", "csv"])

if uploaded_file:
    st.success("File uploaded successfully!")
    st.write("The full AI analysis will be added next.")
