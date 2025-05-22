import os
import glob
from docx import Document
from PyPDF2 import PdfReader
import requests
from sentence_transformers import SentenceTransformer
import joblib
import pandas as pd
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- CONFIG ---
RESUME_FOLDER       = r"C:/Users/hp/EDUVISION/Resumes"
COURSERA_CSV        = "Coursera_catalog.csv"
MODELS_DIR          = "models"
SENT_ENCODER_DIR    = os.path.join(MODELS_DIR, "sbert_all-MiniLM-L6-v2")
KNN_FILE            = os.path.join(MODELS_DIR, "knn_cosine.joblib")
TITLES_FILE         = os.path.join(MODELS_DIR, "knn_titles.joblib")
JOBS_DF_FILE        = os.path.join(MODELS_DIR, "jobs_df.joblib")
JOB_COUNT           = 2
COURSE_TOP_K        = 5

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="EduVision", layout="wide")

# --- LOGIN & API CREDENTIAL SETUP ---
st.title("🔐 EduVision Login & API Setup")
st.markdown(
    "To get your Lightcast API credentials (client_id and client_secret), visit "
    "[Lightcast Open Skills Access](https://lightcast.io/open-skills/access)."
)
client_id = st.text_input("Lightcast Client ID")
client_secret = st.text_input("Lightcast Client Secret")
if st.button("Save & Continue"):
    if not client_id or not client_secret:
        st.error("Both client_id and client_secret are required.")
    else:
        st.session_state['client_id'] = client_id
        st.session_state['secret'] = client_secret
        st.success("Credentials saved! Loading main app…")
if 'client_id' not in st.session_state or 'secret' not in st.session_state:
    st.stop()
CLIENT_ID = st.session_state['client_id']
SECRET = st.session_state['secret']

# --- AUTH TOKEN ---
@st.cache_resource
def get_access_token():
    resp = requests.post(
        "https://auth.emsicloud.com/connect/token",
        data={
            "client_id": CLIENT_ID,
            "client_secret": SECRET,
            "grant_type": "client_credentials",
            "scope": "emsi_open"
        }
    )
    resp.raise_for_status()
    return resp.json().get("access_token", "")

# --- LOAD COURSES & MODELS ---
@st.cache_data
def load_courses():
    df = pd.read_csv(COURSERA_CSV)
    # Ensure title column exists and fill nulls
    if 'course_title' in df.columns:
        df['course_title'] = df['course_title'].fillna('')
    else:
        df['course_title'] = ''
    # Handle optional description column
    if 'course_description' in df.columns:
        df['course_description'] = df['course_description'].fillna('')
    else:
        df['course_description'] = ''
    return df

@st.cache_resource
def build_course_tfidf(courses_df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))
    corpus = (courses_df['course_title'] + ' ' + courses_df['course_description']).tolist()
    matrix = tfidf.fit_transform(corpus)
    return tfidf, matrix

@st.cache_resource
def load_job_models():
    encoder = SentenceTransformer.load(SENT_ENCODER_DIR)
    knn = joblib.load(KNN_FILE)
    titles = joblib.load(TITLES_FILE)
    jobs_df = joblib.load(JOBS_DF_FILE)
    return encoder, knn, titles, jobs_df

# Initialize data
COURSES_DF = load_courses()
tfidf_vec, tfidf_mat = build_course_tfidf(COURSES_DF)
encoder, knn, titles, jobs_df = load_job_models()

# --- UTILITIES ---
def extract_text(path: str) -> str:
    if path.lower().endswith('.pdf'):
        text = ''
        with open(path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ''
        return text
    elif path.lower().endswith(('.doc', '.docx')):
        doc = Document(path)
        return '\n'.join(para.text for para in doc.paragraphs)
    return ''

def clean_text(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).lower().strip()

def extract_skills_api(text: str) -> set:
    token = get_access_token()
    resp = requests.post(
        "https://emsiservices.com/skills/versions/latest/extract",
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
        params={'language': 'en'}, json={'text': text, 'confidenceThreshold': 0.6}
    )
    if resp.status_code == 200:
        return {item['skill']['name'] for item in resp.json().get('data', [])}
    return set()

# --- RECOMMENDERS ---
def recommend_jobs(skills: set, text: str):
    if skills:
        embs = encoder.encode(list(skills), convert_to_numpy=True, show_progress_bar=False)
        q_vec = embs.mean(axis=0, keepdims=True)
    else:
        q_vec = encoder.encode([text], convert_to_numpy=True)
    dists, idxs = knn.kneighbors(q_vec, n_neighbors=JOB_COUNT)
    return [(titles[i], round((1 - dists[0][j]) * 100, 1)) for j, i in enumerate(idxs[0])]

def recommend_courses_for_job(job_title: str, top_k: int = COURSE_TOP_K):
    vec = tfidf_vec.transform([job_title])
    sims = linear_kernel(vec, tfidf_mat).flatten()
    idxs = sims.argsort()[-top_k:][::-1]
    recs = COURSES_DF.iloc[idxs].copy()
    recs['similarity_score'] = sims[idxs]
    return recs

# --- STREAMLIT UI ---
st.title("🎓 AI Career Advisor")
uploaded = st.file_uploader("Upload your resume (PDF/DOCX)", type=["pdf", "docx"])
if uploaded:
    temp_path = f"temp_{uploaded.name}"
    with open(temp_path, 'wb') as f: f.write(uploaded.getbuffer())
    raw = extract_text(temp_path)
    clean = clean_text(raw)
    skills = extract_skills_api(clean)
    st.write("**Extracted Skills:**", ", ".join(skills) if skills else "None found")

    st.subheader("💼 Top Job Recommendations")
    jobs = recommend_jobs(skills, clean)
    for i, (title, score) in enumerate(jobs, 1):
        st.markdown(f"{i}. **{title}** (Match: {score}%)")

    if jobs:
        top_job = jobs[0][0]
        st.subheader(f"📚 Top Courses for '{top_job}'")
        courses = recommend_courses_for_job(top_job)
        if courses.empty:
            st.info("No courses found. Showing popular courses.")
            courses = COURSES_DF.head(top_k)
        for _, row in courses.iterrows():
            st.markdown(
                f"**{row['course_title']}**  \n"
                f"*By {row.get('course_organization','N/A')}*  \n"
                f"Rating: {row.get('course_rating','N/A')}  \n"
                f"Similarity: {row['similarity_score']:.2f}  \n"
            )
    os.remove(temp_path)
else:
    st.write("Please upload a resume to get started.")

if st.button("Analyze Sample Resumes"):
    st.info("Processing sample resumes…")
    results = []
    for resume in glob.glob(os.path.join(RESUME_FOLDER, "*.*")):
        txt = clean_text(extract_text(resume))
        skills = extract_skills_api(txt)
        results.append([os.path.basename(resume), len(skills)])
    st.dataframe(pd.DataFrame(results, columns=["File", "Skills Count"]))
