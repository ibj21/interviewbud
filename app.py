import streamlit as st
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

st.title("📄 AI Interview Prep Buddy")

# Upload resume
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description Here", height=200)

resume_text = ""

if resume_file:
    with open("uploaded_resume.pdf", "wb") as f:
        f.write(resume_file.read())
    resume_text = extract_text("uploaded_resume.pdf")
    st.text_area("Extracted Resume Text", resume_text, height=200)

# Compare
if st.button("Compare Resume with JD"):
    if resume_text and jd_text:
        corpus = [resume_text, jd_text]
        vectorizer = TfidfVectorizer().fit_transform(corpus)
        score = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0] * 100
        st.metric("Match Score", f"{score:.2f}%")
    else:
        st.warning("Please upload resume and paste job description.")

# Optional: Suggestions using OpenAI (needs API key)
openai_api = st.text_input("Enter your OpenAI API key to get smart suggestions", type="password")
if openai_api:
    openai.api_key = openai_api
    if st.button("Get Resume Tips"):
        prompt = f"Analyze the resume below and suggest ways to make it more aligned with this job:\n\nResume:\n{resume_text}\n\nJob:\n{jd_text}"
        with st.spinner("Thinking..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            st.success("Done!")
            st.write(response['choices'][0]['message']['content'])
