import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from src.ai_detect import detect

st.set_page_config(page_title="AI-detect — Google Forms", layout="wide")
st.title("AI-detect — Google Forms responses")
st.write("Upload a Google Forms CSV of responses and get a per-answer AI-likelihood score (heuristic).")

# Upload
uploaded = st.file_uploader("Upload Google Forms CSV", type=["csv"])
answer_col = st.text_input("Answer column header (exact)", value="answer")
student_col = st.text_input("Student/ID column header (optional)", value="student")

use_sample = st.checkbox("Use sample CSV (data/sample_responses.csv) if no upload", value=False)

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    try:
        df = pd.read_csv("data/sample_responses.csv")
    except Exception:
        st.error("Sample CSV not found in data/sample_responses.csv")
        df = None

if df is not None:
    st.subheader("Preview")
    st.dataframe(df.head())

    if answer_col not in df.columns:
        st.warning(f"Column '{answer_col}' not found in uploaded data. Pick the correct header.")
    else:
        if st.button("Run AI detection"):
            with st.spinner("Analyzing answers..."):
                # compute scores
                df["_ai_score"] = df[answer_col].astype(str).apply(lambda t: detect(t)["score"])
                # flag
                def flag_score(s, hi=0.70, lo=0.40):
                    if s >= hi:
                        return "Likely AI"
                    if s <= lo:
                        return "Likely Human"
                    return "Ambiguous"
                df["_flag"] = df["_ai_score"].apply(flag_score)

                # explanations
                def explain(row):
                    s = row["_ai_score"]
                    reasons = []
                    if "—" in str(row[answer_col]) or "--" in str(row[answer_col]):
                        reasons.append("em-dash usage")
                    if len(str(row[answer_col]).split()) > 80:
                        reasons.append("very long answer")
                    if s >= 0.75:
                        reasons.append("high AI score")
                    return "; ".join(reasons) if reasons else "None detected"

                df["_explanation"] = df.apply(explain, axis=1)

            st.success("Analysis complete")
            st.subheader("Annotated responses")
            st.dataframe(df[[c for c in [student_col, answer_col, "_ai_score", "_flag", "_explanation"] if c in df.columns]])

            # download annotated CSV
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download annotated CSV", data=csv_bytes, file_name="annotated_responses.csv", mime="text/csv")

            # Charts
            st.subheader("Visualizations")

            # Bar chart
            fig1, ax1 = plt.subplots(figsize=(8,3))
            df_sorted = df.sort_values("_ai_score", ascending=False)
            ax1.bar(df_sorted.get(student_col, df_sorted.index).astype(str), df_sorted["_ai_score"])
            ax1.set_ylim(0,1)
            ax1.set_ylabel("AI score (0→1)")
            ax1.set_xlabel(student_col if student_col in df.columns else "row")
            ax1.set_title("AI-likelihood by respondent")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig1)

            # Pie chart
            fig2, ax2 = plt.subplots(figsize=(4,4))
            counts = df["_flag"].value_counts()
            ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
            ax2.set_title("Detection summary")
            ax2.axis("equal")
            st.pyplot(fig2)

else:
    st.info("Upload a Google Forms CSV (or enable 'Use sample CSV').")
