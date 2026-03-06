import streamlit as st
import pandas as pd
import re
import json
import time
from ollama_utils import run_ollama

def run_ollama_with_retries(prompt, model, temperature, max_tokens, retries, delay):
    """
    Calls run_ollama with a given prompt, retrying if an exception occurs or if no response is returned.
    :param prompt: The prompt to send to the model.
    :param model: The model name.
    :param temperature: Temperature setting.
    :param max_tokens: Maximum tokens to generate.
    :param retries: Number of attempts.
    :param delay: Delay (in seconds) between retries.
    :return: The response from run_ollama or None if unsuccessful.
    """
    for attempt in range(1, retries + 1):
        try:
            response = run_ollama(prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens)
            if response is not None and response != "":
                return response
            else:
                st.write(f"Attempt {attempt}: Empty response received.")
        except Exception as e:
            st.write(f"Attempt {attempt} failed with error: {e}")
        time.sleep(delay)
    st.write("All attempts failed. Please check your connection or try again later.")
    return None

def main():
    st.title("LocalG")

    # Sidebar: Model choice and configuration
    st.sidebar.subheader("Ollama Settings")
    model_name = st.sidebar.text_input("Pick a Model (e.g., llama2, llama2:13b, etc.)", "llama2")

    st.sidebar.markdown(
        "**Temperature** controls how random or 'creative' the model is. "
        "Lower values (e.g., 0.0) make it more deterministic, while higher values (closer to 1.0) increase randomness."
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

    st.sidebar.markdown(
        "**Max Tokens** is the maximum number of tokens the model can generate. "
        "Increasing this number allows for longer responses, but can be slower or use more resources."
    )
    max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, max_value=2048, value=512)

    # Retry configuration options
    st.sidebar.subheader("Retry Configuration")
    num_retries = st.sidebar.number_input("Number of Retries", min_value=1, max_value=10, value=3, step=1)
    retry_delay = st.sidebar.number_input("Delay Between Retries (seconds)", min_value=1.0, max_value=10.0, value=2.0, step=0.5)

    st.write("## 1. Upload CSV")
    uploaded_file = st.file_uploader("Upload a CSV with columns 'Name/ID' and 'Text'", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview:")
        st.dataframe(df.head())

        st.write("## 2. Choose Operation")
        analysis_option = st.selectbox(
            "What do you want to do?",
            [
                "Error Analysis",
                "Pronoun Analysis",
                "Politeness Analysis",
                "Proficiency Analysis (Numeric Score 0-10)",
                "Proficiency Analysis (Holistic - All Paragraphs Together)",
                "Proficiency Analysis (ACTFL Levels)",
                "Verb Analysis"
            ]
        )

        # -------------------------
        # Error ANALYSIS
        # -------------------------
        if analysis_option == "Error Analysis":
            st.write("This option analyzes the text and returns a list of errors with brief explanations.")
            if st.button("Run Error Analysis"):
                results = []
                for i, row in df.iterrows():
                    text = row["Text"]
                    prompt = f'''
You are a Spanish grammar expert. Analyze the following text and identify any errors (grammatical, punctuation, or usage errors). 
For each error, provide a brief explanation.

Return ONLY a valid JSON object in the following format:
{{
    "Errors": [
        {{
            "Error": "description of the error",
            "Explanation": "brief explanation of the error"
        }},
        ...
    ]
}}
If no errors are found, return an empty list for "Errors".

Text: "{text}"
                    '''
                    response = run_ollama_with_retries(prompt, model_name, temperature, max_tokens, num_retries, retry_delay)
                    if response is None:
                        continue
                    cleaned = re.sub(r"```json|```", "", response).strip()
                    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                    errors = None
                    if json_match:
                        json_text = json_match.group(0)
                        json_text = re.sub(r',\s*(\}|])', r'\1', json_text)
                        try:
                            parsed = json.loads(json_text)
                            errors = parsed.get("Errors", [])
                        except json.JSONDecodeError as e:
                            st.write(f"⚠️ JSON Decode Error for row {i}: {response}")
                            st.write(f"Error details: {str(e)}")
                    else:
                        st.write(f"⚠️ No valid JSON found in response for row {i}. Response: {response}")
                    results.append({
                        "Name/ID": row["Name/ID"],
                        "Text": text,
                        "Errors": errors
                    })
                max_errors = max((len(r["Errors"]) for r in results if r["Errors"]), default=0)
                flattened_results = []
                for r in results:
                    row_data = {"Name/ID": r["Name/ID"], "Text": r["Text"]}
                    errs = r["Errors"] or []
                    for j in range(max_errors):
                        if j < len(errs):
                            row_data[f"Error{j+1}"] = errs[j].get("Error", "")
                            row_data[f"Explanation{j+1}"] = errs[j].get("Explanation", "")
                        else:
                            row_data[f"Error{j+1}"] = ""
                            row_data[f"Explanation{j+1}"] = ""
                    flattened_results.append(row_data)
                results_df = pd.DataFrame(flattened_results)
                st.write("### Error Analysis Results")
                st.dataframe(results_df)
                csv = results_df.to_csv(index=False)
                st.download_button(label="Download Results as CSV", data=csv, file_name="error_analysis_results.csv", mime="text/csv")

        # -------------------------
        # Pronoun ANALYSIS
        # -------------------------
        elif analysis_option == "Pronoun Analysis":
            st.write("This option identifies the pronoun used (e.g., 'yo', 'ella', 'usted') in each sentence.")
            if st.button("Run Pronoun Analysis"):
                results = []
                for i, row in df.iterrows():
                    text = row["Text"]
                    prompt = f'''
You are a Spanish language expert. For each sentence in the following text, identify the pronoun used (such as "yo", "tú", "él", "ella", "usted", etc.).

Return ONLY a valid JSON object that:
- Uses keys in the format "Sentence 1 Pronoun", "Sentence 2 Pronoun", etc.
- If a sentence does not contain a pronoun, set its value to an empty string.

Example valid JSON format:
{{
"Sentence 1 Pronoun": "yo",
"Sentence 2 Pronoun": "él"
}}

Text: "{text}"
                    '''
                    response = run_ollama_with_retries(prompt, model_name, temperature, max_tokens, num_retries, retry_delay)
                    if response is None:
                        continue
                    cleaned = re.sub(r"```json|```", "", response).strip()
                    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                    pronoun_data = {}
                    if json_match:
                        json_text = json_match.group(0)
                        json_text = re.sub(r',\s*(\}|])', r'\1', json_text)
                        try:
                            pronoun_data = json.loads(json_text)
                        except json.JSONDecodeError as e:
                            st.write(f"⚠️ JSON Decode Error for row {i}: {response}")
                            st.write(f"Error details: {str(e)}")
                    else:
                        st.write(f"⚠️ No valid JSON found in response for row {i}. Response: {response}")
                    result_entry = {"Name/ID": row["Name/ID"], "Text": text}
                    result_entry.update(pronoun_data)
                    results.append(result_entry)
                results_df = pd.DataFrame(results)
                st.write("### Pronoun Analysis Results")
                st.dataframe(results_df)
                csv = results_df.to_csv(index=False)
                st.download_button(label="Download Results as CSV", data=csv, file_name="pronoun_analysis_results.csv", mime="text/csv")

        # -------------------------
        # Politeness ANALYSIS
        # -------------------------
        elif analysis_option == "Politeness Analysis":
            st.write("This option analyzes the text and determines its politeness level (e.g., Formal, Informal, or Neutral).")
            if st.button("Run Politeness Analysis"):
                results = []
                for i, row in df.iterrows():
                    text = row["Text"]
                    prompt = f'''
You are a Spanish language expert. Analyze the following text and determine its level of politeness. 
Is the language formal, informal, or neutral? Provide only one of these classifications: "Formal", "Informal", or "Neutral".
Return ONLY a valid JSON object in the following format:
{{ "Politeness": "classification" }}
Text: "{text}"
                    '''
                    response = run_ollama_with_retries(prompt, model_name, temperature, max_tokens, num_retries, retry_delay)
                    if response is None:
                        continue
                    cleaned = re.sub(r"```json|```", "", response).strip()
                    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                    politeness = None
                    if json_match:
                        json_text = json_match.group(0)
                        json_text = re.sub(r',\s*(\}|])', r'\1', json_text)
                        try:
                            parsed = json.loads(json_text)
                            politeness = parsed.get("Politeness", None)
                        except json.JSONDecodeError as e:
                            st.write(f"⚠️ JSON Decode Error for row {i}: {response}")
                            st.write(f"Error details: {str(e)}")
                    else:
                        st.write(f"⚠️ No valid JSON found in response for row {i}. Response: {response}")
                    results.append({"Name/ID": row["Name/ID"], "Text": text, "Politeness": politeness})
                results_df = pd.DataFrame(results)
                st.write("### Politeness Analysis Results")
                st.dataframe(results_df)
                csv = results_df.to_csv(index=False)
                st.download_button(label="Download Results as CSV", data=csv, file_name="politeness_analysis_results.csv", mime="text/csv")

        # -------------------------
        # Proficiency Analysis (Numeric Score 0-10)
        # -------------------------
        elif analysis_option == "Proficiency Analysis (Numeric Score 0-10)":
            st.write("This option analyzes each text line using a proficiency rubric and returns a score between 0 and 10.")
            if st.button("Run Proficiency Analysis"):
                results = []
                for i, row in df.iterrows():
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
                    response = run_ollama_with_retries(prompt, model_name, temperature, max_tokens, num_retries, retry_delay)
                    if response is None:
                        continue
                    score = extract_json(response, "Proficiency Score")
                    results.append({"Name/ID": row["Name/ID"], "Text": text, "Proficiency Score": score})
                display_results(results, "Proficiency Analysis Results")

        # -------------------------
        # Proficiency Analysis (Holistic - All Paragraphs Together)
        # -------------------------
        elif analysis_option == "Proficiency Analysis (Holistic - All Paragraphs Together)":
            st.write("This option evaluates all paragraphs written by the same student as a single unit.")
            if st.button("Run Holistic Proficiency Analysis"):
                grouped = df.groupby("Name/ID")["Text"].apply(lambda x: " ".join(map(str, x))).reset_index()
                results = []
                for i, row in grouped.iterrows():
                    text = row["Text"]
                    prompt = f"""
You are a Spanish language expert evaluating a Spanish learner's proficiency. The student wrote multiple paragraphs using different tenses: present, past, future, and subjunctive. 

Assess the overall proficiency based on:
1. Fluency and coherence
2. Grammatical accuracy
3. Proper use of different tenses
4. Vocabulary complexity

Provide a proficiency score from 0 to 10. Return ONLY a JSON object:
{{ "Proficiency Score": score }}

Text: "{text}"
"""
                    response = run_ollama_with_retries(prompt, model_name, temperature, max_tokens, num_retries, retry_delay)
                    if response is None:
                        continue
                    score = extract_json(response, "Proficiency Score")
                    results.append({"Name/ID": row["Name/ID"], "Text": text, "Proficiency Score": score})
                display_results(results, "Holistic Proficiency Analysis Results")

        # -------------------------
        # Proficiency Analysis (ACTFL Levels)
        # -------------------------
        elif analysis_option == "Proficiency Analysis (ACTFL Levels)":
            st.write("This option categorizes proficiency based on ACTFL levels.")
            if st.button("Run ACTFL Proficiency Analysis"):
                results = []
                for i, row in df.iterrows():
                    text = row["Text"]
                    prompt = f"""
You are an expert in Spanish language assessment. Evaluate the following text and assign an ACTFL proficiency level.

ACTFL Proficiency Levels:
- Novice Low
- Novice Mid
- Novice High
- Intermediate Low
- Intermediate Mid
- Intermediate High
- Advanced Low
- Advanced Mid
- Advanced High
- Superior

Consider fluency, accuracy, vocabulary, and complexity. Return a JSON object:
{{ "ACTFL Level": "Level Here" }}

Text: "{text}"
"""
                    response = run_ollama_with_retries(prompt, model_name, temperature, max_tokens, num_retries, retry_delay)
                    if response is None:
                        continue
                    level = extract_json(response, "ACTFL Level")
                    results.append({"Name/ID": row["Name/ID"], "Text": text, "ACTFL Level": level})
                display_results(results, "ACTFL Proficiency Analysis Results")

        # -------------------------
        # Verb ANALYSIS
        # -------------------------
        elif analysis_option == "Verb Analysis":
            st.write(
                "This option analyzes the first two verbs in each sentence, extracting the infinitive, tense, and correctness."
            )
            if st.button("Run Verb Analysis"):
                results = []
                for i, row in df.iterrows():
                    text = row["Text"]
                    prompt = f"""
You are a Spanish grammar expert. Analyze the following text to identify verbs.

**Instructions:**
1. Extract the **first two verbs** from the text (if present).
2. For each verb, provide:
   - **The infinitive form**.
   - **The verb tense** (e.g., present, past, future, etc.).
   - **Whether the verb is used correctly in context** (Correct/Incorrect).

Return ONLY a valid JSON object in the following format:
{{
    "Verb 1 Infinitive": "infinitive_verb_1",
    "Verb 1 Tense": "tense_verb_1",
    "Verb 1 Correctness": "Correct/Incorrect",
    "Verb 2 Infinitive": "infinitive_verb_2",
    "Verb 2 Tense": "tense_verb_2",
    "Verb 2 Correctness": "Correct/Incorrect"
}}

If fewer than two verbs are present, leave the missing values empty.

Text: "{text}"
"""
                    response = run_ollama_with_retries(prompt, model_name, temperature, max_tokens, num_retries, retry_delay)
                    if response is None:
                        continue
                    cleaned = re.sub(r"```json|```", "", response).strip()
                    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                    verb1_inf = verb1_tense = verb1_correct = None
                    verb2_inf = verb2_tense = verb2_correct = None  
                    if json_match:
                        json_text = json_match.group(0)
                        json_text = re.sub(r',\s*(\}|])', r'\1', json_text)
                        try:
                            parsed = json.loads(json_text)
                            verb1_inf = (parsed.get("Verb 1 Infinitive") or "").strip() or None
                            verb1_tense = (parsed.get("Verb 1 Tense") or "").strip() or None
                            verb1_correct = (parsed.get("Verb 1 Correctness") or "").strip() or None
                            verb2_inf = (parsed.get("Verb 2 Infinitive") or "").strip() or None
                            verb2_tense = (parsed.get("Verb 2 Tense") or "").strip() or None
                            verb2_correct = (parsed.get("Verb 2 Correctness") or "").strip() or None
                            if not verb2_inf:
                                verb2_inf, verb2_tense, verb2_correct = None, None, None
                        except json.JSONDecodeError as e:
                            st.write(f"⚠️ JSON Decode Error for row {i}: {response}")
                            st.write(f"Error details: {str(e)}")
                    else:
                        st.write(f"⚠️ No valid JSON found in response for row {i}. Response: {response}")
                    results.append({
                        "Name/ID": row["Name/ID"],
                        "Text": text,
                        "Verb 1 Infinitive": verb1_inf,
                        "Verb 1 Tense": verb1_tense,
                        "Verb 1 Correctness": verb1_correct,
                        "Verb 2 Infinitive": verb2_inf,
                        "Verb 2 Tense": verb2_tense,
                        "Verb 2 Correctness": verb2_correct
                    })
                results_df = pd.DataFrame(results)
                st.write("### Verb Analysis Results")
                st.dataframe(results_df)
                csv = results_df.to_csv(index=False)
                st.download_button(label="Download Results as CSV", data=csv, file_name="verb_analysis_results.csv", mime="text/csv")

def extract_json(response, key):
    """Extracts a specific key from a JSON object in the LLM response."""
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

def display_results(results, title):
    """Displays a DataFrame of results and provides a CSV download button."""
    results_df = pd.DataFrame(results)
    st.write(f"### {title}")
    st.dataframe(results_df)
    csv = results_df.to_csv(index=False)
    st.download_button(label="Download Results as CSV", data=csv, file_name="analysis_results.csv", mime="text/csv")

if __name__ == "__main__":
    main()
