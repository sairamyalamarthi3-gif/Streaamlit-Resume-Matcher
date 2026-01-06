import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="AI Resume Auto-Enhancer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Resume Matcher & Auto-Enhancer")
st.markdown(
    "Automatically improves your resume based on job description skill gaps."
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

def calculate_similarity(resume, job):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform([resume, job])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

def auto_enhance_resume(resume, missing_skills):
    enhancement = "\n\n🔹 Skills Enhancement (AI-Suggested):\n"
    enhancement += ", ".join(missing_skills[:10])
    enhancement += (
        "\n\nNote: Actively learning and applying these skills "
        "through hands-on projects and real-world use cases."
    )
    return resume + enhancement

if st.button("🔍 Analyze Resume", use_container_width=True):
    if resume_text and job_text:
        score = calculate_similarity(resume_text, job_text)

        resume_words = set(resume_text.lower().split())
        job_words = set(job_text.lower().split())
        missing_skills = sorted(list(job_words - resume_words))

        st.subheader("📊 Initial Match Result")
        st.metric("Match Score", f"{score}%")
        st.progress(int(score))

        if score >= 70:
            st.success("🎉 Strong resume match!")
            st.balloons()
        elif score >= 40:
            st.warning("⚠️ Moderate match – can be improved.")
        else:
            st.error("❌ Low match – improvement needed.")

        st.subheader("📌 Missing Keywords")
        st.write(missing_skills[:15] if missing_skills else "No major gaps found.")

        # Auto-enhancement
        if score < 65 and missing_skills:
            st.divider()
            st.subheader("🤖 Auto-Enhanced Resume (AI Generated)")

            enhanced_resume = auto_enhance_resume(
                resume_text, missing_skills
            )

            new_score = calculate_similarity(enhanced_resume, job_text)

            st.text_area(
                "🛠️ Enhanced Resume (Review & Edit)",
                value=enhanced_resume,
                height=300
            )

            st.subheader("📈 Improved Match Score")
            st.metric("Updated Score", f"{new_score}%")
            st.progress(int(new_score))

            if new_score > score:
                st.success("✅ Resume automatically improved successfully!")
            else:
                st.info("ℹ️ Minor improvement — manual refinement recommended.")

    else:
        st.warning("Please paste both Resume and Job Description.")
