"""Streamlit UI for the AI-Based Fake News Detection project."""
import os
import sys
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_news, ModelNotFoundError, EmptyInputError, InsufficientTextError

st.set_page_config(page_title="AI-Based Fake News Detection", page_icon="📰", layout="centered")
st.title("📰 AI-Based Fake News Detection")
st.caption("TF-IDF + Logistic Regression with a conservative publication-attribution check")
st.divider()

st.subheader("Enter the news")
headline = st.text_input("Headline", placeholder="Paste the news headline here...")
article = st.text_area(
    "Article text",
    height=300,
    placeholder="Paste the full article here for a stronger prediction...",
)

check_clicked = st.button("Check News", type="primary", use_container_width=True)

if check_clicked:
    try:
        with st.spinner("Analyzing article..."):
            result = predict_news(headline, article)

        label = result["label"]
        if label == "REAL":
            st.success("Prediction: **REAL** ✅")
        elif label == "FAKE":
            st.error("Prediction: **FAKE** ⚠️")
        else:
            st.warning(
                f"Prediction: **UNCERTAIN** ⚠️ — raw ML prediction: "
                f"**{result['raw_label']}** ({result['ml_confidence']*100:.1f}%)."
            )

        st.metric("Final confidence", f"{result['confidence']*100:.1f}%")
        st.progress(min(max(result["confidence"], 0.0), 1.0))

        st.subheader("Model details")
        st.write(f"**Raw ML prediction:** {result['raw_label']}")
        st.write(f"**ML confidence:** {result['ml_confidence']*100:.1f}%")
        st.write(f"**Decision reason:** {result['reason']}")

        if result["source_signals"]:
            st.info("Publication signals detected: " + ", ".join(result["source_signals"]))

        st.subheader("Why the ML model predicted this")
        if result["top_words"]:
            st.write(f"Words contributing most toward the **{result['raw_label']}** ML prediction:")
            cols = st.columns(min(4, len(result["top_words"])))
            for i, (word, weight) in enumerate(result["top_words"]):
                with cols[i % len(cols)]:
                    st.markdown(f"**{word}**")
        else:
            st.info("No strong individual word contributions were found.")

        with st.expander("See cleaned text used by the model"):
            st.code(result["cleaned_text"] or "(empty after cleaning)")

    except (EmptyInputError, InsufficientTextError) as e:
        st.warning(str(e))
    except ModelNotFoundError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.divider()
st.caption(
    "⚠️ Academic project. The ML score is statistical and trained on a limited historical dataset. "
    "The publication-attribution signal is supporting evidence, not definitive fact-checking."
)
