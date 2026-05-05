import streamlit as st
from transformers import pipeline
import time

st.set_page_config(
    page_title="Hinglish Sentiment Analyzer",
    page_icon="",
    layout="centered"
)

# -------- HEADER --------
st.title("Hinglish Sentiment Analyzer")
st.caption("Understand sentiment in Hinglish e-commerce reviews")

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="Arman-03/sentiment-analysis-hinglish"
    )

try:
    clf = load_model()
except Exception:
    st.error("Model failed to load. Please refresh.")
    st.stop()

# -------- SIDEBAR --------
with st.sidebar:
    st.header("Try Examples")

    examples = [
        "bhai product mast hai",
        "bilkul bekaar hai waste of money",
        "thik hai kuch khas nahi",
        "acha hai but expected better tha",
        "wah kya quality hai, 2 din me toot gaya"
    ]

    for ex in examples:
        if st.button(ex):
            st.session_state["input_text"] = ex

    st.markdown("---")
    st.write("Built with Transformers + Streamlit")

# -------- INPUT --------
text = st.text_area(
    "Enter Hinglish review:",
    value=st.session_state.get("input_text", ""),
    height=120,
    placeholder="e.g., bhai mast product hai but delivery late thi..."
)

# -------- ANALYZE --------
if st.button("Analyze", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing sentiment..."):
            time.sleep(0.3)  # small UX delay
            result = clf(text)[0]

        label = result["label"]
        score = result["score"]

        st.subheader("Result")

        # -------- EMOJI + COLOR --------
        if label.lower() == "positive":
            st.success(f"😊 Positive ({score:.2f})")
            explanation = "The review expresses satisfaction or approval."
        elif label.lower() == "negative":
            st.error(f"😠 Negative ({score:.2f})")
            explanation = "The review expresses dissatisfaction or complaint."
        else:
            st.info(f"😐 Neutral ({score:.2f})")
            explanation = "The review is mixed, balanced, or unclear."

        # -------- CONFIDENCE --------
        st.write("Confidence")
        st.progress(float(score))

        # -------- EXPLANATION --------
        st.markdown("### Interpretation")
        st.write(explanation)

        # -------- EXTRA INSIGHT --------
        st.markdown("### Observations")
        if "but" in text.lower():
            st.write("• Mixed sentiment detected (presence of 'but')")
        if "!" in text:
            st.write("• Strong emotion indicated by punctuation")
        if len(text.split()) < 4:
            st.write("• Very short review — may be ambiguous")

# -------- FOOTER --------
st.markdown("---")
st.caption("Demo app for Hinglish sentiment classification")