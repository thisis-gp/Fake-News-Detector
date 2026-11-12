import streamlit as st

from model.model import predict

st.set_page_config(page_title="Fake News Detector", page_icon="assets/news_favicon.png")

st.title("Fake News Detector")
st.caption(
    "A classroom classifier (TF-IDF + logistic regression) trained on the Kaggle Fake/True news "
    "dataset. It learns the writing style of that dataset. It does not check facts."
)

text = st.text_area("Paste a news article", height=240)

if st.button("Classify", disabled=not text.strip()):
    is_fake, confidence = predict(text)
    label = "Looks like the Fake set" if is_fake else "Looks like the True set"
    st.subheader(label)
    st.write(f"Model confidence: {confidence:.0%}")
    st.info("Short or non-US-politics text is outside what this model was trained on. Do not rely on it.")
