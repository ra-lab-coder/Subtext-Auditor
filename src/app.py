import os
import streamlit as st
import joblib

# --- Paths ---
# Use "models" if that's your repo folder name
MODEL_DIR = "model"
MODEL_FILENAME = "deepsea_model_v1.pkl"  # change if your file is named differently
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


@st.cache_resource
def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def interpret_score(score: float):
    """
    Interpret probability that the text is Emotionally Dependent / Hot (class 1).
    Returns (verdict, description).
    """
    if score < 0.3:
        verdict = "🧊 Mostly Task-Oriented / Side-by-Side"
        desc = (
            "The model sees the interaction as primarily focused on tasks, ideas, or external goals. "
            "Signals of emotional dependency are low."
        )
    elif score < 0.7:
        verdict = "🟡 Mixed Orientation / Ambiguous"
        desc = (
            "The model detects a blend of task/idea focus and relational/emotional cues. "
            "This often happens in supportive friendships or high-frequency collaboration."
        )
    else:
        verdict = "🔥 Mostly Emotionally Dependent / Face-to-Face"
        desc = (
            "The model detects strong relational-focus signals such as emotional validation-seeking, "
            "prioritization of the interaction, dependency language, or boundary-blurring patterns."
        )

    return verdict, desc


def main():
    st.set_page_config(page_title="DeepSea Communication Orientation Auditor", page_icon="🌊")

    st.title("🌊 DeepSea Communication Orientation Auditor")
    st.write(
        "This app classifies a chat snippet by **communication orientation**:\n"
        "- **Task-Oriented / Cold (side-by-side)**\n"
        "- **Emotionally Dependent / Hot (face-to-face)**\n\n"
        "It does **not** determine what the relationship *is* (friends/partners/etc.). "
        "It only analyzes the *style* of interaction in the text you provide."
    )

    # Load model
    try:
        model = load_model()
    except FileNotFoundError:
        st.error(
            f"Model file not found at: `{MODEL_PATH}`\n\n"
            "Check that the model exists and that `MODEL_DIR` / `MODEL_FILENAME` match your repo."
        )
        return

    st.subheader("Input Conversation")

    default_example = (
        "A: Morning — quick question about my CV draft.\n"
        "B: Sure, send it over and I’ll give feedback.\n"
        "A: Thanks. Also… when you don’t reply I get a bit unsettled.\n"
        "B: I’m here. Message me whenever you need."
    )

    text = st.text_area(
        "Paste chat here:",
        value=default_example,
        height=220,
        help=(
            "You can include multiple lines with speaker labels. "
            "The model treats the entire block as one document."
        ),
    )

    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please paste some text first.")
            return

        # Predict probability for class 1
        proba = model.predict_proba([text])[0]

        # Safer than assuming proba[1] is class 1:
        # map by model.classes_
        class_to_proba = {int(c): float(p) for c, p in zip(model.classes_, proba)}
        p_hot = class_to_proba.get(1, float(proba[-1]))

        verdict, desc = interpret_score(p_hot)

        st.subheader("Result")
        st.markdown(f"**Orientation Score (Class 1 probability):** `{p_hot:.2f}`")
        st.progress(p_hot)
        st.markdown(f"### {verdict}")
        st.write(desc)

        st.markdown("#### Notes")
        st.write(
            "- This is a **pattern-based classifier**, not a moral judgment.\n"
            "- Use it for **reflection / research**, not as proof about a relationship.\n"
            "- Scores can be sensitive to context, sarcasm, and missing conversation history."
        )

    st.markdown("---")
    st.caption(
        "Prototype tool trained on synthetic data. Educational/research use only; "
        "not psychological, legal, or relationship advice."
    )


if __name__ == "__main__":
    main()
