import streamlit as st
import pandas as pd
import re
import json
import time
from ollama_utils import run_ollama


def run_ollama_with_retries(prompt, model, temperature, max_tokens, retries, delay):
    """Call run_ollama with retries if it fails or returns an empty response."""
    for attempt in range(1, retries + 1):
        try:
            response = run_ollama(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            if response is not None and response != "":
                return response
            st.write(f"Attempt {attempt}: Empty response received.")
        except Exception as e:
            st.write(f"Attempt {attempt} failed with error: {e}")
        time.sleep(delay)

    st.write("All attempts failed. Please check your connection or try again later.")
    return None


def extract_json(response, key):
    """Extract a specific key from a JSON object in the LLM response."""
    if not isinstance(response, str):
        response = str(response) if response else ""

    cleaned = re.sub(r"```json|```", "", response).strip()
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if json_match:
        json_text = json_match.group(0)
        json_text = re.sub(r',\s*(\}|])', r'\1', json_text)
        try:
            parsed = json.loads(json_text)
            return parsed.get(key, None)
        except json.JSONDecodeError:
            return None
    return None


def display_results(results, title, filename):
    """Display results and provide a CSV download button."""
    results_df = pd.DataFrame(results)
    st.write(f"### {title}")
    st.dataframe(results_df)
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


def main():
    st.title("Local LLM Scoring for Spanish L2 Writing")

    st.sidebar.subheader("Ollama Settings")
    model_name = st.sidebar.text_input("Pick a model (e.g., llama3.1:8b)", "llama3.1:8b")

    st.sidebar.markdown(
        "**Temperature** controls how random the model is. "
        "Lower values make it more deterministic."
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.sidebar.markdown(
        "**Max Tokens** is the maximum number of tokens the model can generate."
    )
    max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, max_value=2048, value=512)

    st.sidebar.subheader("Retry Configuration")
    num_retries = st.sidebar.number_input("Number of Retries", min_value=1, max_value=10, value=3, step=1)
    retry_delay = st.sidebar.number_input("Delay Between Retries (seconds)", min_value=1.0, max_value=10.0, value=2.0, step=0.5)

    st.write("## 1. Upload CSV")
    uploaded_file = st.file_uploader("Upload a CSV with columns 'Name/ID' and 'Text'", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        required_cols = {"Name/ID", "Text"}
        if not required_cols.issubset(df.columns):
            st.error("CSV must contain 'Name/ID' and 'Text' columns.")
            return

        st.write("### Data Preview:")
        st.dataframe(df.head())

        st.write("## 2. Choose Operation")
        analysis_option = st.selectbox(
            "What do you want to do?",
            [
                "Experiment 1 - Analytic Rubric (Paragraph Level)",
                "Experiment 2 - Holistic Scoring (Student Level)",
            ]
        )

        if analysis_option == "Experiment 1 - Analytic Rubric (Paragraph Level)":
            st.write("This option analyzes each text line using a proficiency rubric and returns a score between 0 and 10.")

            if st.button("Run Experiment 1"):
                results = []
                for _, row in df.iterrows():
                    text = row["Text"]
                    prompt = f"""You are a Spanish language expert. Analyze the following text using the rubric below and provide a proficiency score between 0 (lowest) and 10 (highest).

Rubric:
Communication Effectiveness (3 points)
  1 Point: Minimal communication; frequent misunderstandings hinder meaning.
  2 Points: Basic communication; occasional misunderstandings but manages to convey main ideas.
  3 Points: Clear and effective communication; ideas are well expressed.

Grammatical Accuracy (4 points)
  1 Point: Frequent grammatical errors that disrupt understanding.
  2 Points: Occasional grammatical errors; mostly clear.
  3 Points: Mostly accurate; few errors.
  4 Points: Excellent grammatical control; complex structures used accurately.

Relevance and Completeness (3 points)
  1 Point: Partially answers; key aspects missing.
  2 Points: Adequately answers; most details included.
  3 Points: Thorough and comprehensive answer.

Calculate the total score out of 10 by summing the points.
Return ONLY a valid JSON object in the following format:
{{ "Proficiency Score": score }}

Text: "{text}"
"""
                    response = run_ollama_with_retries(
                        prompt, model_name, temperature, max_tokens, num_retries, retry_delay
                    )
                    if response is None:
                        continue

                    score = extract_json(response, "Proficiency Score")
                    results.append({
                        "Name/ID": row["Name/ID"],
                        "Text": text,
                        "Proficiency Score": score
                    })

                display_results(results, "Experiment 1 Results", "experiment1_analytic_results.csv")

        elif analysis_option == "Experiment 2 - Holistic Scoring (Student Level)":
            st.write("This option evaluates all paragraphs written by the same student as a single unit.")

            if st.button("Run Experiment 2"):
                grouped = df.groupby("Name/ID")["Text"].apply(lambda x: " ".join(map(str, x))).reset_index()
                results = []

                for _, row in grouped.iterrows():
                    text = row["Text"]
                    prompt = f"""You are a Spanish language expert evaluating a Spanish learner's proficiency. The student wrote multiple paragraphs using different tenses: present, past, future, and subjunctive.

Assess the overall proficiency based on:
1. Fluency and coherence
2. Grammatical accuracy
3. Proper use of different tenses
4. Vocabulary complexity

Provide a proficiency score from 0 to 10. Return ONLY a JSON object:
{{ "Proficiency Score": score }}

Text: "{text}"
"""
                    response = run_ollama_with_retries(
                        prompt, model_name, temperature, max_tokens, num_retries, retry_delay
                    )
                    if response is None:
                        continue

                    score = extract_json(response, "Proficiency Score")
                    results.append({
                        "Name/ID": row["Name/ID"],
                        "Text": text,
                        "Proficiency Score": score
                    })

                display_results(results, "Experiment 2 Results", "experiment2_holistic_results.csv")


if __name__ == "__main__":
    main()
