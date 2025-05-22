# EDUVISION
## Description

EduVision is an AI-powered career advisor designed to help users find job recommendations and relevant courses based on their resumes. It uses natural language processing (NLP) and machine learning techniques to analyze resumes, extract skills, and match them with suitable jobs and educational courses. The project consists of two main components:

A Jupyter notebook (job_recommend.ipynb) that demonstrates the core logic for job recommendations using TF-IDF and SBERT.

A Streamlit app (app2.py) that provides a user-friendly interface for uploading resumes and obtaining personalized recommendations.

## Features

Extract skills from uploaded resumes (PDF or DOCX) using the Lightcast API.

Recommend top jobs based on extracted skills and resume content using TF-IDF with cosine similarity and SBERT with Nearest Neighbors.

Suggest relevant courses from a dataset (e.g., Coursera courses) related to the recommended jobs.

Analyze sample resumes in batch for demonstration purposes.

## Installation

Install the required dependencies using pip:
```bash
pip install pandas scikit-learn sentence-transformers streamlit docx PyPDF2 requests joblib numpy
```
## Setup

To run this project, follow these setup steps:

### Datasets:

Ensure you have the following datasets in the project directory:

linkedin_title_skills.csv: Contains job titles and associated skills.

Coursera_catalog.csv: Contains course information for course recommendations.

linkedin_title_skills.csv dataset is not provided, you can obtain it from (https://www.kaggle.com/code/rochajose/linkedin-top-skills-in-i-t-dev-da-ds-ml).

### Models:

Due to GitHub's file size limits, the pre-trained models (including sbert_all-MiniLM-L6-v2) is not included in this repository.

You can run the job_recommend.ipynb notebook to generate the models.

**Alternatively, the SBERT model can be downloaded automatically:**
```bash
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### Lightcast API Credentials:

Register on the Lightcast Open Skills Access website to obtain your client_id and client_secret.

These credentials are required for the Streamlit app to extract skills from resumes.

## Running the Streamlit App

Make sure the models are set up (either by running the notebook or downloading models.zip).

Run the app using:
```bash
streamlit run app2.py
```

In the app:

Enter your Lightcast API credentials on the login page.

Upload your resume (PDF or DOCX) to get job and course recommendations.
