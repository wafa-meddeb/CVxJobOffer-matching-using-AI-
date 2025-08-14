# AI-Powered CV–Job Matching Platform

# 📌 Overview

This project is an AI-powered recruitment platform developed during my summer internship at Nouvelair Tunisie in 2025.
It automates the process of matching resumes (CVs) with job offers using Large Language Models (LLMs), embeddings, and grammar scoring to provide ranked candidate lists for recruiters.

The system is built on Django’s MVT architecture, integrates Groq-hosted LLaMA 3 models for parsing, and uses semantic similarity scoring with BERT embeddings.

# 🎯 Project Scope

Automate CV screening for HR departments.

Extract structured data from unstructured PDF resumes and job offers.

Match candidates to job postings using a scoring system.

Provide exportable ranked lists for recruiters.

# 🔍 Features
Core Functionalities

CV Upload: Bulk PDF uploads with AI parsing.

Job Offer Management: Create, edit, delete, and view job postings.

Information Extraction: Use LLMs to parse CVs and job descriptions into JSON.

Matching & Scoring:

Semantic similarity via BERT embeddings.

Grammar checking with penalty scoring.

Weighted final match score.

Results & Reporting:

Ranked candidate lists.

Export to Excel.

Analytics Dashboard:

KPIs, match score distribution, and trends.


# 🛠 Technology Stack

Backend:

Python 3

Django (MVT framework)

Groq API with LLaMA 3–8B model

PyMuPDF (PDF text extraction)

BERT / Sentence Transformers (semantic similarity)

LanguageTool API (grammar checking)

Frontend:

HTML, CSS, JavaScript

Django Templates

Database:

PostgreSQL (default, can use SQLite for dev)


# 📊 AI Matching Pipeline

Text Extraction – Convert PDFs to text using PyMuPDF.

LLM Parsing – Extract structured fields (skills, experience, etc.) with Groq-hosted LLaMA 3–8B.

Embedding Generation – Encode CVs and job offers into vector space using BERT.

Similarity Scoring – Calculate cosine similarity, adjust with grammar penalty.

Ranking & Export – Produce a ranked list of candidates per job offer.

# 🚀 Installation & Setup

# Clone the repository
git clone https://github.com/yourusername/cv-job-matching.git
cd cv-job-matching

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file
@"
SECRET_KEY=your_django_secret_key
DEBUG=True
DATABASE_URL=postgres://user:password@localhost:5432/dbname
GROQ_API_KEY=your_groq_api_key
"@ | Out-File -Encoding UTF8 .env

# Apply migrations
python manage.py migrate

# Run the development server
python manage.py runserver


# 👩‍💻 Author

Wafa Meddeb – AI Engineering Student at ESPRIT School of Engineering
Developed during a Summer Internship at Nouvelair Tunisie (2025).
