import streamlit as st
import pickle

with open("model/model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

st.title("📰 Fake News Detection System")

st.write("Enter a news article below to check whether it is Real or Fake.")

news = st.text_area("Enter News Article")

if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter some news.")
    else:
        news_vector = vectorizer.transform([news])
        result = model.predict(news_vector)[0]

        if result == 0:
            st.error("❌ FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")