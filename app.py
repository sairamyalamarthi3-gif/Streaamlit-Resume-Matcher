import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))"))

# ---------- Page Config ----------
st.set_page_config(
    page_title="AI Resume Enhancer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Matcher & GPT Enhancer")
st.caption("Upload resume → Analyze → Auto-rewrite with GPT → Export PDF")

# ---------- Utility Functions ----------
def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    return " ".join(page.extract_text() for page in reader.pages)

def similarity_score(resume, jd):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform([resume, jd])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

def gpt_rewrite_resume(resume, jd):
    prompt = f"""
You are an AI resume assistant.
Rewrite the resume to better match the job description.
Do NOT fabricate experience.
Improve skills, bullets, and wording professionally.

RESUME:
{resume}

JOB DESCRIPTION:
{jd}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content

def generate_pdf(text, filename="enhanced_resume.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y = height - 40

    for line in text.split("\n"):
        if y < 40:
            c.showPage()
            y = height - 40
        c.drawString(40, y, line)
        y -= 14

    c.save()
    return filename

# ---------- UI Tabs ----------
tab1, tab2, tab3 = st.tabs(
    ["📄 Upload Resume", "📊 Match Analysis", "✨ GPT Enhancement"]
)

with tab1:
    uploaded_pdf = st.file_uploader("Upload Resume (PDF)", type="pdf")
    job_desc = st.text_area("Paste Job Description", height=250)

    if uploaded_pdf:
        resume_text = extract_pdf_text(uploaded_pdf)
        st.success("Resume uploaded successfully")
        st.text_area("Extracted Resume Text", resume_text, height=250)

with tab2:
    if uploaded_pdf and job_desc:
        score = similarity_score(resume_text, job_desc)

        st.metric("Match Score", f"{score}%")
        st.progress(int(score))

        if score >= 70:
            st.success("Strong match")
        elif score >= 40:
            st.warning("Moderate match")
        else:
            st.error("Low match")

with tab3:
    if uploaded_pdf and job_desc:
        if st.button("🤖 Enhance Resume with GPT", use_container_width=True):
            with st.spinner("GPT is rewriting your resume..."):
                enhanced_resume = gpt_rewrite_resume(resume_text, job_desc)

            new_score = similarity_score(enhanced_resume, job_desc)

            st.subheader("✨ Enhanced Resume")
            st.text_area("", enhanced_resume, height=300)

            st.metric("Updated Match Score", f"{new_score}%")
            st.progress(int(new_score))

            pdf_file = generate_pdf(enhanced_resume)
            with open(pdf_file, "rb") as f:
                st.download_button(
                    "⬇️ Download Enhanced Resume (PDF)",
                    f,
                    file_name="Enhanced_Resume.pdf",
                    use_container_width=True
                )
