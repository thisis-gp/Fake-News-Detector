import streamlit as st
import requests
from streamlit_lottie import st_lottie

def load_animation(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

news_animation = load_animation("https://lottie.host/51c21153-c45b-4c30-b428-142bbc6a7aec/sph9K6z6Ec.json")

st.set_page_config(page_title="Fake News Detector", page_icon="assets/news_favicon.png", layout="wide")

st.title("Fake News Detector")
input_text = st.text_input("Enter news article: ")

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)

    with left_column:
        if input_text:
            st.write("This news is True")
        else:
            st.write("This news is Fake")

    with right_column:
        if news_animation:
            st_lottie(news_animation, height=300, key="news")
        else:
            st.write("Failed to load animation.")
