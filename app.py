import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.title("🤖 AI Resume Skill Matcher")
st.write("Match your resume against a job description in real time")

resume_text = st.text_area("📄 Paste your Resume", height=200)
job_text = st.text_area("🧾 Paste Job Description", height=200)

if st.button("🔍 Analyze Match"):
    if resume_text and job_text:
        documents = [resume_text, job_text]

        tfidf = TfidfVectorizer(stop_words="english")
        vectors = tfidf.fit_transform(documents)

        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        match_percentage = round(similarity * 100, 2)

        resume_words = set(resume_text.lower().split())
        job_words = set(job_text.lower().split())
        missing_skills = job_words - resume_words

        st.success(f"✅ Match Score: {match_percentage}%")

        if match_percentage > 70:
            st.balloons()
            st.write("🎉 Great match for the role!")
        elif match_percentage > 40:
            st.warning("⚠️ Moderate match. Improve skill alignment.")
        else:
            st.error("❌ Low match. Consider upskilling.")

        st.subheader("📌 Missing Keywords (Sample)")
        st.write(list(missing_skills)[:10])
    else:
        st.warning("Please paste both Resume and Job Description")
