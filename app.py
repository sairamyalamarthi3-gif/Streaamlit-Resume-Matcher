import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="AI Resume Skill Matcher",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Skill Matcher")
st.markdown(
    "Analyze how well your resume matches a job description and improve it intelligently."
)

st.divider()

# Layout
col1, col2 = st.columns(2)

with col1:
    resume_text = st.text_area(
        "📄 Paste Your Resume",
        height=300,
        placeholder="Paste your resume text here..."
    )

with col2:
    job_text = st.text_area(
        "🧾 Paste Job Description",
        height=300,
        placeholder="Paste job description here..."
    )

st.divider()

if st.button("🔍 Analyze Match", use_container_width=True):
    if resume_text and job_text:
        documents = [resume_text, job_text]

        tfidf = TfidfVectorizer(stop_words="english")
        vectors = tfidf.fit_transform(documents)

        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        match_percentage = round(similarity * 100, 2)

        resume_words = set(resume_text.lower().split())
        job_words = set(job_text.lower().split())
        missing_skills = list(job_words - resume_words)

        st.subheader("📊 Match Result")
        st.progress(int(match_percentage))
        st.metric("Match Score", f"{match_percentage}%")

        if match_percentage >= 70:
            st.success("🎉 Excellent match! Your resume aligns well.")
            st.balloons()

        elif 40 <= match_percentage < 70:
            st.warning("⚠️ Moderate match. Resume can be improved.")

        else:
            st.error("❌ Low match. Resume needs improvement.")

        # Missing Skills
        st.subheader("📌 Missing / Weak Keywords")
        if missing_skills:
            st.write(missing_skills[:15])
        else:
            st.write("No major missing keywords detected.")

        # Resume improvement section
        if match_percentage < 60:
            st.divider()
            st.subheader("✍️ Improve Your Resume")

            st.info(
                "Tips to improve your resume:\n"
                "- Add missing skills naturally\n"
                "- Use keywords from the job description\n"
                "- Mention tools, frameworks, and impact\n"
                "- Avoid generic sentences"
            )

            improved_resume = st.text_area(
                "🛠️ Edit Your Resume Below and Re-check",
                value=resume_text,
                height=250
            )

            if st.button("🔁 Re-Analyze Updated Resume"):
                new_docs = [improved_resume, job_text]
                new_vectors = tfidf.fit_transform(new_docs)
                new_similarity = cosine_similarity(
                    new_vectors[0:1], new_vectors[1:2]
                )[0][0]

                new_score = round(new_similarity * 100, 2)

                st.success(f"✅ Updated Match Score: {new_score}%")
                st.progress(int(new_score))

    else:
        st.warning("Please provide both Resume and Job Description.")
