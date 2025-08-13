# Enhanced ai_utils.py with embedding-based matching - FIXED VERSION
import os
import re
import json
import requests
import base64
import io
import uuid
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertModel
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF
import language_tool_python
from typing import List, Dict, Tuple

import time
from collections import defaultdict
from datetime import datetime

import logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to .env or your deployment environment.")

# ---- Groq API Configuration ----
ROUTER_URL = "https://api.groq.com/openai/v1/chat/completions"
TEXT_MODEL = "llama3-8b-8192"  

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

# ---- Embedding Model Configuration ----
class EmbeddingModel:
    _instance = None
    _initialized = False

    def __init__(self):
        if EmbeddingModel._instance is not None:
            raise Exception("This class is a singleton!")
        
        try:
            logger.info("Initializing DistilBERT embedding model...")
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.model.eval()
            self._initialized = True
            EmbeddingModel._instance = self
            logger.info("DistilBERT embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # Fallback to simple TF-IDF based similarity
            self.tokenizer = None
            self.model = None
            self._initialized = False
            EmbeddingModel._instance = self

    @staticmethod
    def get_instance():
        if EmbeddingModel._instance is None:
            EmbeddingModel()
        return EmbeddingModel._instance
    
    def is_initialized(self) -> bool:
        return self._initialized and self.model is not None
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using DistilBERT
        Falls back to TF-IDF if model is not available
        """
        if not self.is_initialized():
            logger.warning("Embedding model not initialized, falling back to TF-IDF similarity")
            return self._fallback_tfidf_embeddings(texts)
        
        embeddings = []
        
        try:
            with torch.no_grad():
                for text in texts:
                    # Tokenize and encode
                    inputs = self.tokenizer(
                        text, 
                        return_tensors='pt', 
                        max_length=512, 
                        truncation=True, 
                        padding=True
                    )
                    
                    # Get model outputs
                    outputs = self.model(**inputs)
                    
                    # Use the [CLS] token embedding (first token)
                    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
                    embeddings.append(cls_embedding[0])
            
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return self._fallback_tfidf_embeddings(texts)
    
    def _fallback_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Fallback TF-IDF based embeddings when DistilBERT is not available
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            return tfidf_matrix.toarray()
        except ImportError:
            logger.error("Neither DistilBERT nor scikit-learn available for embeddings")
            # Return simple word count based embeddings
            return self._simple_word_embeddings(texts)
    
    def _simple_word_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Very simple word-based embeddings as last resort
        """
        # Create a vocabulary from all texts
        all_words = set()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.update(words)
        
        vocab = list(all_words)
        
        # Create embeddings based on word presence
        embeddings = []
        for text in texts:
            words = set(re.findall(r'\b\w+\b', text.lower()))
            embedding = [1 if word in words else 0 for word in vocab]
            embeddings.append(embedding)
        
        return np.array(embeddings)

# Global embedding model instance (lazy loaded)
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = EmbeddingModel.get_instance()
        except Exception as e:
            logger.error(f"Failed to get embedding model: {e}")
            # Create a dummy model for fallback
            class DummyEmbeddingModel:
                def is_initialized(self):
                    return False
                def get_embeddings(self, texts):
                    return np.random.rand(len(texts), 100)  # Random embeddings as fallback
            embedding_model = DummyEmbeddingModel()
    return embedding_model

# ---- Grammar Checker Configuration ----
class GrammarChecker:
    def __init__(self):
        self.use_textblob = False
        self.use_regex = True
        
        # Try to initialize TextBlob
        try:
            from textblob import TextBlob
            # Test with simple text to ensure it works
            test = TextBlob("test sentence")
            test.correct()  # Test correction function
            self.use_textblob = True
            logger.info("TextBlob grammar checker initialized successfully")
        except ImportError:
            logger.warning("TextBlob not installed. Install with: pip install textblob")
        except Exception as e:
            logger.warning(f"TextBlob initialization failed: {e}")
        
        # Fallback regex patterns (same as your original)
        if not self.use_textblob:
            logger.info("Using regex-based grammar checking as fallback")
            self._init_regex_patterns()
    
    def _init_regex_patterns(self):
        """Initialize regex patterns for grammar checking"""
        self.error_patterns = [
            # Subject-verb disagreement patterns
            (r'\b(he|she|it)\s+(are|were)\b', 'Subject-verb disagreement'),
            (r'\b(they|we|you)\s+(is|was)\b', 'Subject-verb disagreement'),
            
            # Double negatives
            (r"(don't|doesn't|didn't|won't|can't|shouldn't)\s+\w*\s+(no|nothing|nobody|nowhere|never)\b", 'Double negative'),
            
            # Wrong word forms
            (r'\bthere\s+are\s+\d+\s+person\b', 'Singular/plural mismatch'),
            (r'\bmuch\s+\w*s\b', 'Much with plural'),
            (r'\bmany\s+\w+(?<!s)\b', 'Many with singular'),
            
            # Common typos/wrong words
            (r'\byour\s+(going|coming|running)', 'Your/you\'re confusion'),
            (r'\bits\s+a\b', 'Its/it\'s confusion'),
            
            # Capitalization errors
            (r'\bi\s+(?!am|was|were|will|would|should|could|might|must)', 'Uncapitalized I'),
            
            # Article errors
            (r'\ba\s+([aeiouAEIOU])', 'Article error'),
            
            # Sentence fragments (basic check)
            (r'^[a-z][^.!?]*[^.!?]', 'Possible sentence fragment'),
        ]
    
    def _textblob_check(self, text: str) -> Tuple[int, int]:
        """Check grammar using TextBlob"""
        try:
            from textblob import TextBlob
            
            if not text.strip():
                return 0, 0
            
            # Create TextBlob object
            blob = TextBlob(text)
            words = len(blob.words)
            
            if words == 0:
                return 0, 0
            
            # Get corrected version
            corrected_blob = blob.correct()
            
            # Count differences (indicates potential errors)
            original_words = [word.lower() for word in blob.words]
            corrected_words = [word.lower() for word in corrected_blob.words]
            
            # Count spelling errors
            spelling_errors = 0
            min_length = min(len(original_words), len(corrected_words))
            
            for i in range(min_length):
                if original_words[i] != corrected_words[i]:
                    spelling_errors += 1
            
            # Add length difference as potential errors
            length_diff = abs(len(original_words) - len(corrected_words))
            spelling_errors += length_diff
            
            # Also run basic regex checks for grammar
            grammar_errors = self._regex_check_simple(text)
            
            total_errors = spelling_errors + grammar_errors
            return total_errors, words
            
        except Exception as e:
            logger.error(f"TextBlob check failed: {e}")
            # Fallback to regex
            return self._regex_check_full(text)
    
    def _regex_check_simple(self, text: str) -> int:
        """Simple regex check for common grammar errors"""
        error_count = 0
        
        # Basic patterns only
        simple_patterns = [
            r'\b(he|she|it)\s+(are|were)\b',
            r'\b(they|we|you)\s+(is|was)\b',
            r'\byour\s+(going|coming|running)',
            r'\bi\s+(?!am|was|were|will)',
        ]
        
        for pattern in simple_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            error_count += len(matches)
        
        return error_count
    
    def _regex_check_full(self, text: str) -> Tuple[int, int]:
        """Full regex-based grammar checking"""
        if not text.strip():
            return 0, 0
        
        error_count = 0
        words = len(text.split())
        
        for pattern, _ in self.error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            error_count += len(matches)
        
        return error_count, max(words, 1)
    
    def check_grammar(self, text: str) -> Tuple[int, int]:
        """
        Main interface: Check grammar and return (errors_count, total_words)
        """
        if not text or not text.strip():
            return 0, 0
        
        if self.use_textblob:
            return self._textblob_check(text)
        else:
            return self._regex_check_full(text)
    
    def calculate_grammar_penalty(self, text: str, max_penalty: float = 0.3) -> float:
        """
        Calculate grammar penalty as a fraction (0 to max_penalty)
        """
        if not text.strip():
            return 0.0
        
        errors, total_words = self.check_grammar(text)
        
        if total_words == 0:
            return 0.0
        
        # Calculate error rate
        error_rate = errors / total_words
        
        # Apply penalty scaling based on method used
        if self.use_textblob:
            # TextBlob is more accurate, so we can be more aggressive
            penalty = min(error_rate * 1.8, max_penalty)
        else:
            # Regex is less accurate, so be more conservative
            penalty = min(error_rate * 2.0, max_penalty)
        
        logger.debug(f"Grammar penalty: {errors} errors in {total_words} words = {penalty:.3f}")
        
        return penalty
    
    def get_correction_suggestions(self, text: str) -> str:
        """
        Get corrected version of text (if available)
        """
        if not self.use_textblob:
            return text
        
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            corrected = blob.correct()
            return str(corrected)
        except Exception as e:
            logger.error(f"Error getting corrections: {e}")
            return text
        
        
# Global grammar checker instance
grammar_checker = GrammarChecker()

# ---------------- PDF Text Extraction Helper ----------------

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using PyMuPDF (fitz).
    Returns the full text content of the PDF.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page in doc:
            full_text += page.get_text()
        
        doc.close()
        return full_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

# ---------------- Text Processing Helpers ----------------

def prepare_text_for_embedding(data: dict, is_job: bool = False) -> str:
    """
    Prepare structured data (job or CV) for embedding generation
    """
    if is_job:
        # Combine job offer fields into a comprehensive text
        parts = []
        
        if data.get('title'):
            parts.append(f"Job Title: {data['title']}")
        
        if data.get('description'):
            parts.append(f"Description: {data['description']}")
        
        if data.get('requirements'):
            reqs = data['requirements']
            if isinstance(reqs, list):
                parts.append(f"Requirements: {' '.join(reqs)}")
            else:
                parts.append(f"Requirements: {reqs}")
        
        if data.get('nice_to_have'):
            nice = data['nice_to_have']
            if isinstance(nice, list):
                parts.append(f"Nice to have: {' '.join(nice)}")
            else:
                parts.append(f"Nice to have: {nice}")
        
        if data.get('technologies'):
            techs = data['technologies']
            if isinstance(techs, list):
                parts.append(f"Technologies: {' '.join(techs)}")
            else:
                parts.append(f"Technologies: {techs}")
        
        if data.get('languages'):
            langs = data['languages']
            if isinstance(langs, list):
                parts.append(f"Languages: {' '.join(langs)}")
            else:
                parts.append(f"Languages: {langs}")
        
        return ' '.join(parts)
    
    else:
        # Combine CV fields into a comprehensive text
        parts = []
        
        if data.get('name'):
            parts.append(f"Name: {data['name']}")
        
        if data.get('summary'):
            parts.append(f"Summary: {data['summary']}")
        
        if data.get('skills'):
            skills = data['skills']
            if isinstance(skills, list):
                parts.append(f"Skills: {' '.join(skills)}")
            else:
                parts.append(f"Skills: {skills}")
        
        if data.get('experience'):
            exp = data['experience']
            if isinstance(exp, list):
                exp_texts = []
                for item in exp:
                    if isinstance(item, dict):
                        exp_text = f"{item.get('title', '')} at {item.get('company', '')} - {item.get('description', '')}"
                        exp_texts.append(exp_text)
                    else:
                        exp_texts.append(str(item))
                parts.append(f"Experience: {' '.join(exp_texts)}")
            else:
                parts.append(f"Experience: {exp}")
        
        if data.get('education'):
            edu = data['education']
            if isinstance(edu, list):
                edu_texts = []
                for item in edu:
                    if isinstance(item, dict):
                        edu_text = f"{item.get('degree', '')} from {item.get('school', '')}"
                        edu_texts.append(edu_text)
                    else:
                        edu_texts.append(str(item))
                parts.append(f"Education: {' '.join(edu_texts)}")
            else:
                parts.append(f"Education: {edu}")
        
        if data.get('languages'):
            langs = data['languages']
            if isinstance(langs, list):
                parts.append(f"Languages: {' '.join(langs)}")
            else:
                parts.append(f"Languages: {langs}")
        
        return ' '.join(parts)

def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract skill keywords from text for skill matching analysis
    """
    # Common technical skills and keywords
    skill_patterns = [
        r'\b(?:python|java|javascript|react|node\.js|angular|vue\.js|django|flask|spring|html|css|sql|mongodb|postgresql|mysql|git|docker|kubernetes|aws|azure|gcp|machine learning|ai|data science|tensorflow|pytorch)\b',
        r'\b(?:project management|scrum|agile|leadership|communication|teamwork|problem solving|analytical|creative|strategic|planning)\b'
    ]
    
    skills = set()
    for pattern in skill_patterns:
        matches = re.findall(pattern, text.lower())
        skills.update(matches)
    
    return list(skills)

# ---------------- Matching Functions ----------------

def calculate_cosine_similarity(job_text: str, cv_text: str) -> float:
    """
    Calculate cosine similarity between job and CV texts using embeddings
    """
    try:
        # Get embedding model
        model = get_embedding_model()
        
        # Generate embeddings
        embeddings = model.get_embeddings([job_text, cv_text])
        
        if embeddings.shape[0] < 2:
            logger.error("Failed to generate embeddings for both texts")
            return 0.0
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
        similarity_score = similarity_matrix[0][0]
        
        # Ensure score is between 0 and 1
        similarity_score = max(0.0, min(1.0, float(similarity_score)))
        
        logger.debug(f"Cosine similarity calculated: {similarity_score:.3f}")
        return similarity_score
    
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        # Fallback to simple text overlap
        return calculate_simple_text_similarity(job_text, cv_text)

def calculate_simple_text_similarity(job_text: str, cv_text: str) -> float:
    """
    Simple text similarity as fallback when embeddings fail
    """
    try:
        # Convert to lowercase and split into words
        job_words = set(re.findall(r'\b\w+\b', job_text.lower()))
        cv_words = set(re.findall(r'\b\w+\b', cv_text.lower()))
        
        if not job_words or not cv_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = job_words.intersection(cv_words)
        union = job_words.union(cv_words)
        
        similarity = len(intersection) / len(union) if union else 0.0
        
        logger.debug(f"Simple text similarity: {similarity:.3f}")
        return similarity
    
    except Exception as e:
        logger.error(f"Error in simple text similarity: {e}")
        return 0.0

def calculate_skill_overlap(job_skills: List[str], cv_skills: List[str]) -> Tuple[List[str], float]:
    """
    Calculate skill overlap between job requirements and CV skills
    Returns (matched_skills, overlap_ratio)
    """
    if not job_skills or not cv_skills:
        return [], 0.0
    
    # Convert to lowercase for case-insensitive matching
    job_skills_lower = [skill.lower().strip() for skill in job_skills]
    cv_skills_lower = [skill.lower().strip() for skill in cv_skills]
    
    # Find matches
    matched_skills = []
    for job_skill in job_skills:
        if job_skill.lower().strip() in cv_skills_lower:
            matched_skills.append(job_skill)
    
    # Calculate overlap ratio
    overlap_ratio = len(matched_skills) / len(job_skills) if job_skills else 0.0
    
    logger.debug(f"Skill overlap: {len(matched_skills)}/{len(job_skills)} = {overlap_ratio:.3f}")
    return matched_skills, overlap_ratio

def match_resume_to_job(job_data: dict, cv_data: dict) -> dict:
    """
    Match a single resume to a job offer using embedding-based similarity and grammar analysis
    """
    try:
        logger.info(f"Matching CV {cv_data.get('cv_id', 'Unknown')} to job {job_data.get('job_id', 'Unknown')}")
        
        # Prepare texts for embedding
        job_text = prepare_text_for_embedding(job_data, is_job=True)
        cv_text = prepare_text_for_embedding(cv_data, is_job=False)
        
        logger.debug(f"Job text length: {len(job_text)} chars")
        logger.debug(f"CV text length: {len(cv_text)} chars")
        
        # Calculate cosine similarity
        # similarity_score = calculate_cosine_similarity(job_text, cv_text)
        similarity_score = calculate_cosine_similarity_with_tracking(job_text, cv_text)
        
        # Calculate grammar penalty based on CV text quality
        grammar_penalty = grammar_checker.calculate_grammar_penalty(cv_text)
        
        # Calculate final score (similarity minus grammar penalty)
        final_score = max(0.0, similarity_score - grammar_penalty)
        
        # Extract and match skills
        job_skills = extract_skills_from_text(job_text)
        cv_skills = extract_skills_from_text(cv_text)
        matched_skills, skill_overlap = calculate_skill_overlap(job_skills, cv_skills)
        
        result = {
            'cv_id': cv_data.get('cv_id', ''),
            'name': cv_data.get('name', 'Unknown'),
            'title': cv_data.get('title', ''),
            'summary': cv_data.get('summary', ''),
            'similarity_score': similarity_score,
            'grammar_penalty': grammar_penalty,
            'final_score': final_score,
            'matched_skills': matched_skills,
            'skill_overlap_ratio': skill_overlap,
            'job_skills_count': len(job_skills),
            'cv_skills_count': len(cv_skills),
            'matched_skills_count': len(matched_skills)
        }
        
        logger.info(f"Matching completed for {cv_data.get('name', 'Unknown')}: final_score={final_score:.3f}")
        return result
        
    except Exception as e:
        logger.error(f"Error in match_resume_to_job: {e}")
        # Return error result
        return {
            'cv_id': cv_data.get('cv_id', ''),
            'name': cv_data.get('name', 'Unknown'),
            'title': 'Error',
            'summary': f'Error processing resume: {str(e)}',
            'similarity_score': 0.0,
            'grammar_penalty': 0.0,
            'final_score': 0.0,
            'matched_skills': [],
            'skill_overlap_ratio': 0.0,
            'job_skills_count': 0,
            'cv_skills_count': 0,
            'matched_skills_count': 0
        }

# ---------------- Existing Functions (Updated) ----------------

def build_prompt_cv(cv_text: str) -> str:
    return f"""You are an HR assistant. Extract the following fields in pure JSON.
Fields:
- name, email, phone
- summary
- skills (list of strings)
- education (list of objects with degree, school, years)
- experience (list of objects with title, company, years)
- languages (list of strings)
- certificates (list of strings)

Resume:
{cv_text}

Return ONLY valid JSON (no prose, no markdown fences).
"""

def build_prompt_job_offer(text: str) -> str:
    return f"""You are an HR assistant. Extract this job offer into JSON with:
- title, company, location, description
- requirements (list of strings)
- nice_to_have (list of strings)
- contract_type, experience_level
- languages (list), technologies (list), salary_range

Job offer:
{text}

Return ONLY valid JSON (no prose, no markdown fences).
"""

# ---------------- JSON Extraction ----------------

def extract_json_from_text(text: str) -> dict:
    """
    Find first balanced JSON object in text and parse it.
    Handles markdown code blocks and multiline descriptions.
    """
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Remove any text after the JSON (like "Note: If the job offer...")
    lines = text.split('\n')
    json_lines = []
    brace_count = 0
    started = False
    
    for line in lines:
        if '{' in line and not started:
            started = True
        
        if started:
            json_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            
            # Stop when braces are balanced
            if brace_count == 0 and '}' in line:
                break
    
    json_text = '\n'.join(json_lines)
    
    if not json_text.strip():
        raise ValueError(f"No JSON found in model output: {text[:200]}")
    
    # Clean up the JSON
    try:
        # First attempt: try to parse as-is
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Second attempt: fix common issues
        try:
            # Replace problematic characters in description field
            json_text = re.sub(r'include:\s*\n\s*-', 'include:', json_text)
            json_text = re.sub(r'\n\s*-\s*', ' • ', json_text)  # Replace bullet points
            json_text = re.sub(r'\n\s+', ' ', json_text)        # Replace newlines with spaces
            json_text = re.sub(r',\s*([\]}])', r'\1', json_text) # Remove trailing commas
            
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            # Third attempt: extract field by field using regex
            try:
                result = {}
                
                # Extract each field using regex
                title_match = re.search(r'"title":\s*"([^"]*)"', json_text)
                result['title'] = title_match.group(1) if title_match else ''
                
                company_match = re.search(r'"company":\s*"([^"]*)"', json_text)
                result['company'] = company_match.group(1) if company_match else ''
                
                location_match = re.search(r'"location":\s*"([^"]*)"', json_text)
                result['location'] = location_match.group(1) if location_match else ''
                
                # For description, extract everything between quotes, handling multiline
                desc_match = re.search(r'"description":\s*"(.*?)"(?=,\s*"|\s*})', json_text, re.DOTALL)
                if desc_match:
                    desc = desc_match.group(1)
                    desc = re.sub(r'\s*include:\s*', ' include: ', desc)
                    desc = re.sub(r'\s*-\s*', ' • ', desc)
                    desc = re.sub(r'\s+', ' ', desc)
                    result['description'] = desc.strip()
                else:
                    result['description'] = ''
                
                # Extract arrays
                req_match = re.search(r'"requirements":\s*\[(.*?)\]', json_text, re.DOTALL)
                if req_match:
                    req_content = req_match.group(1)
                    requirements = re.findall(r'"([^"]+)"', req_content)
                    result['requirements'] = requirements
                else:
                    result['requirements'] = []
                
                nice_match = re.search(r'"nice_to_have":\s*\[(.*?)\]', json_text, re.DOTALL)
                if nice_match:
                    nice_content = nice_match.group(1)
                    nice_to_have = re.findall(r'"([^"]+)"', nice_content)
                    result['nice_to_have'] = nice_to_have
                else:
                    result['nice_to_have'] = []
                
                contract_match = re.search(r'"contract_type":\s*"([^"]*)"', json_text)
                result['contract_type'] = contract_match.group(1) if contract_match else ''
                
                exp_match = re.search(r'"experience_level":\s*"([^"]*)"', json_text)
                result['experience_level'] = exp_match.group(1) if exp_match else ''
                
                # Extract languages and technologies arrays
                lang_match = re.search(r'"languages":\s*\[(.*?)\]', json_text, re.DOTALL)
                if lang_match:
                    lang_content = lang_match.group(1)
                    languages = re.findall(r'"([^"]+)"', lang_content)
                    result['languages'] = languages
                else:
                    result['languages'] = []
                
                tech_match = re.search(r'"technologies":\s*\[(.*?)\]', json_text, re.DOTALL)
                if tech_match:
                    tech_content = tech_match.group(1)
                    technologies = re.findall(r'"([^"]+)"', tech_content)
                    result['technologies'] = technologies
                else:
                    result['technologies'] = []
                
                salary_match = re.search(r'"salary_range":\s*"([^"]*)"', json_text)
                result['salary_range'] = salary_match.group(1) if salary_match else ''
                
                return result
                
            except Exception as regex_error:
                raise ValueError(f"Error parsing JSON: {e}\nRegex fallback failed: {regex_error}\nRaw JSON: {json_text[:500]}")

# ---------------- Router Call Helpers ----------------

def _chat_completion(messages, model=TEXT_MODEL, temperature: float = 0.2, max_tokens: int = 800, timeout_s: int = 60) -> str:
    """
    Call Groq API chat completions and return the assistant message content (string).
    """
    payload = {
        "model": model,
        "messages": messages,  # list of {"role": "user"/"system"/"assistant", "content": "..."}
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(ROUTER_URL, headers=HEADERS, json=payload, timeout=timeout_s)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Router request failed: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"Router error {resp.status_code}: {resp.text[:400]}")

    data = resp.json()
    # OpenAI-compatible structure: choices[0].message.content
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected Router response format: {data}")

# ---------------- Public Parsing Functions (Text) ----------------

def parse_cv_with_llm(cv_text: str) -> dict:
    """
    Build CV prompt, call Router, and return parsed JSON dict.
    """
    prompt = build_prompt_cv(cv_text)
    content = _chat_completion([
        {"role": "system", "content": "You are a precise JSON extractor for HR."},
        {"role": "user", "content": prompt},
    ])
    return extract_json_from_text(content.strip())

def parse_job_offer_with_llm(job_text: str) -> dict:
    """
    Build Job Offer prompt, call Router, and return parsed JSON dict.
    """
    prompt = build_prompt_job_offer(job_text)
    content = _chat_completion([
        {"role": "system", "content": "You are a precise JSON extractor for HR."},
        {"role": "user", "content": prompt},
    ])
    return extract_json_from_text(content.strip())

# ---------------- PDF Text-Based Parsing Functions ----------------

def parse_cv_from_pdf(pdf_path: str) -> dict:
    """
    Parse CV from PDF by extracting text and processing with text LLM.
    """
    try:
        # Extract text from PDF
        cv_text = extract_text_from_pdf(pdf_path)
        
        if not cv_text.strip():
            raise ValueError("No text could be extracted from PDF")
        
        logger.info(f"Extracted text length: {len(cv_text)} characters")
        logger.debug(f"Text preview: {cv_text[:200]}...")
        
        # Process with text LLM
        return parse_cv_with_llm(cv_text)
        
    except Exception as e:
        logger.error(f"Error parsing CV from PDF: {e}")
        raise

def parse_job_offer_from_pdf(pdf_path: str) -> dict:
    """
    Parse job offer from PDF by extracting text and processing with text LLM.
    """
    try:
        # Extract text from PDF
        job_text = extract_text_from_pdf(pdf_path)
        
        if not job_text.strip():
            raise ValueError("No text could be extracted from PDF")
        
        logger.info(f"Extracted text length: {len(job_text)} characters")
        logger.debug(f"Text preview: {job_text[:200]}...")
        
        # Process with text LLM
        return parse_job_offer_with_llm(job_text)
        
    except Exception as e:
        logger.error(f"Error parsing job offer from PDF: {e}")
        raise

# Initialize embedding model on module load (optional - can be lazy loaded)
def initialize_models():
    """
    Initialize AI models at startup
    """
    try:
        logger.info("Initializing AI models...")
        get_embedding_model()
        logger.info("AI models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing AI models: {e}")
        logger.warning("System will use fallback methods for matching")
        
        
        
        
        
        
# ---- Runtime Detection System ----

class EmbeddingMethodTracker:
    """
    Tracks which embedding method is being used at runtime
    """
    
    def __init__(self):
        self.method_usage = defaultdict(int)
        self.method_timings = defaultdict(list)
        self.current_session_methods = []
        self.last_method_used = None
        self.session_start = datetime.now()
    
    def record_method_usage(self, method_name: str, duration: float = 0.0, success: bool = True):
        """Record which method was used and its performance"""
        timestamp = datetime.now()
        
        # Update counters
        self.method_usage[method_name] += 1
        self.method_timings[method_name].append(duration)
        self.last_method_used = method_name
        
        # Record for current session
        self.current_session_methods.append({
            'method': method_name,
            'timestamp': timestamp.isoformat(),
            'duration': duration,
            'success': success
        })
        
        # Log the usage
        logger.info(f"🔧 EMBEDDING METHOD: {method_name} | Duration: {duration:.3f}s | Success: {success}")
    
    def get_primary_method(self) -> str:
        """Get the most frequently used method"""
        if not self.method_usage:
            return "None"
        return max(self.method_usage.items(), key=lambda x: x[1])[0]
    
    def get_current_status(self) -> dict:
        """Get current runtime status"""
        total_calls = sum(self.method_usage.values())
        
        status = {
            'session_duration': str(datetime.now() - self.session_start),
            'total_embedding_calls': total_calls,
            'last_method_used': self.last_method_used,
            'primary_method': self.get_primary_method(),
            'method_distribution': dict(self.method_usage),
            'method_percentages': {},
            'average_timings': {},
            'current_capability_level': self._get_capability_level()
        }
        
        # Calculate percentages
        if total_calls > 0:
            for method, count in self.method_usage.items():
                status['method_percentages'][method] = round((count / total_calls) * 100, 2)
        
        # Calculate average timings
        for method, timings in self.method_timings.items():
            if timings:
                status['average_timings'][method] = round(sum(timings) / len(timings), 3)
        
        return status
    
    def _get_capability_level(self) -> str:
        """Determine current system capability level"""
        primary = self.get_primary_method()
        
        if primary == "DistilBERT":
            return "🟢 OPTIMAL - Using transformer embeddings"
        elif primary == "TF-IDF":
            return "🟡 GOOD - Using statistical embeddings" 
        elif primary == "SimpleWord":
            return "🟠 BASIC - Using word-based embeddings"
        elif primary == "Random":
            return "🔴 EMERGENCY - Using random embeddings"
        else:
            return "⚪ UNKNOWN - No embedding calls detected"
    
    def print_status_report(self):
        """Print a comprehensive status report"""
        status = self.get_current_status()
        
        print("\n" + "="*60)
        print("🔍 EMBEDDING METHOD RUNTIME DETECTION REPORT")
        print("="*60)
        
        print(f"📊 Session Duration: {status['session_duration']}")
        print(f"🔢 Total Embedding Calls: {status['total_embedding_calls']}")
        print(f"🎯 Current Capability: {status['current_capability_level']}")
        print(f"⚡ Last Method Used: {status['last_method_used'] or 'None'}")
        print(f"👑 Primary Method: {status['primary_method']}")
        
        print("\n📈 METHOD USAGE DISTRIBUTION:")
        for method, percentage in status['method_percentages'].items():
            bar_length = int(percentage / 2)  # Scale bar to 50 chars max
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {method:12} │{bar}│ {percentage:5.1f}% ({status['method_distribution'][method]} calls)")
        
        print("\n⏱️  AVERAGE PROCESSING TIMES:")
        for method, avg_time in status['average_timings'].items():
            print(f"  {method:12} │ {avg_time:.3f} seconds")
        
        print("\n🔧 RECENT ACTIVITY:")
        recent_methods = self.current_session_methods[-5:]  # Last 5 calls
        for call in recent_methods:
            success_icon = "✅" if call['success'] else "❌"
            print(f"  {call['timestamp'][:19]} │ {call['method']:12} │ {call['duration']:.3f}s {success_icon}")
        
        print("="*60)

# Global tracker instance
embedding_tracker = EmbeddingMethodTracker()

# ---- Enhanced EmbeddingModel with Detection ----

class EmbeddingModelWithTracking(EmbeddingModel):
    """
    Enhanced EmbeddingModel that tracks which method is being used
    """
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings with runtime method tracking
        """
        start_time = time.time()
        method_used = None
        success = False
        
        try:
            if not self.is_initialized():
                logger.warning("🔴 DistilBERT not available, falling back to TF-IDF")
                result = self._fallback_tfidf_embeddings(texts)
                method_used = "TF-IDF"
                success = True
                return result
            
            # Try DistilBERT
            embeddings = []
            
            with torch.no_grad():
                for text in texts:
                    # Tokenize and encode
                    inputs = self.tokenizer(
                        text, 
                        return_tensors='pt', 
                        max_length=512, 
                        truncation=True, 
                        padding=True
                    )
                    
                    # Get model outputs
                    outputs = self.model(**inputs)
                    
                    # Use the [CLS] token embedding (first token)
                    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
                    embeddings.append(cls_embedding[0])
            
            method_used = "DistilBERT"
            success = True
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"🔴 DistilBERT failed: {e}, falling back to TF-IDF")
            try:
                result = self._fallback_tfidf_embeddings(texts)
                method_used = "TF-IDF"
                success = True
                return result
            except Exception as e2:
                logger.error(f"🟠 TF-IDF failed: {e2}, falling back to simple word embeddings")
                try:
                    result = self._simple_word_embeddings(texts)
                    method_used = "SimpleWord"
                    success = True
                    return result
                except Exception as e3:
                    logger.error(f"🔴 All methods failed: {e3}, using random embeddings")
                    method_used = "Random"
                    success = False
                    return np.random.rand(len(texts), 100)
        
        finally:
            duration = time.time() - start_time
            if method_used:
                embedding_tracker.record_method_usage(method_used, duration, success)

    def _fallback_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Enhanced TF-IDF with tracking
        """
        start_time = time.time()
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            logger.info("🟡 Using TF-IDF vectorization")
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            duration = time.time() - start_time
            logger.debug(f"TF-IDF completed in {duration:.3f} seconds")
            
            return tfidf_matrix.toarray()
            
        except ImportError:
            logger.error("🔴 scikit-learn not available for TF-IDF")
            raise
    
    def _simple_word_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Enhanced simple embeddings with tracking
        """
        start_time = time.time()
        
        logger.info("🟠 Using simple word-based embeddings")
        
        # Create a vocabulary from all texts
        all_words = set()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.update(words)
        
        vocab = list(all_words)
        logger.debug(f"Created vocabulary with {len(vocab)} words")
        
        # Create embeddings based on word presence
        embeddings = []
        for text in texts:
            words = set(re.findall(r'\b\w+\b', text.lower()))
            embedding = [1 if word in words else 0 for word in vocab]
            embeddings.append(embedding)
        
        duration = time.time() - start_time
        logger.debug(f"Simple word embeddings completed in {duration:.3f} seconds")
        
        return np.array(embeddings)

# ---- Enhanced Similarity Calculation with Detection ----

def calculate_cosine_similarity_with_tracking(job_text: str, cv_text: str) -> float:
    """
    Enhanced cosine similarity calculation with method tracking
    """
    try:
        logger.debug("🔍 Starting similarity calculation...")
        
        # Get embedding model
        model = get_embedding_model()
        
        # Generate embeddings (this will automatically track the method used)
        embeddings = model.get_embeddings([job_text, cv_text])
        
        if embeddings.shape[0] < 2:
            logger.error("Failed to generate embeddings for both texts")
            return 0.0
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
        similarity_score = similarity_matrix[0][0]
        
        # Ensure score is between 0 and 1
        similarity_score = max(0.0, min(1.0, float(similarity_score)))
        
        logger.info(f"✅ Similarity calculated: {similarity_score:.3f} using {embedding_tracker.last_method_used}")
        return similarity_score
    
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        # Fallback to simple text overlap
        return calculate_simple_text_similarity(job_text, cv_text)

# ---- Diagnostic Functions ----

def diagnose_embedding_capabilities():
    """
    Run diagnostic tests to determine what embedding methods are available
    """
    print("\n🔬 RUNNING EMBEDDING CAPABILITIES DIAGNOSTIC")
    print("="*50)
    
    results = {
        'distilbert': False,
        'scikit_learn': False,
        'pytorch': False,
        'transformers': False
    }
    
    # Test DistilBERT
    try:
        from transformers import DistilBertTokenizer, DistilBertModel
        import torch
        
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Test with sample text
        inputs = tokenizer("test", return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        results['distilbert'] = True
        results['pytorch'] = True
        results['transformers'] = True
        print("✅ DistilBERT: Available and functional")
        
    except ImportError as e:
        print(f"❌ DistilBERT: Not available - {e}")
    except Exception as e:
        print(f"⚠️  DistilBERT: Available but error - {e}")
    
    # Test scikit-learn
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(["test text", "another test"])
        similarity = cosine_similarity(matrix)
        
        results['scikit_learn'] = True
        print("✅ Scikit-learn: Available and functional")
        
    except ImportError as e:
        print(f"❌ Scikit-learn: Not available - {e}")
    except Exception as e:
        print(f"⚠️  Scikit-learn: Available but error - {e}")
    
    # Determine expected runtime behavior
    print(f"\n🎯 EXPECTED RUNTIME BEHAVIOR:")
    if results['distilbert']:
        print("   Primary Method: DistilBERT embeddings 🟢")
    elif results['scikit_learn']:
        print("   Primary Method: TF-IDF vectorization 🟡")
    else:
        print("   Primary Method: Simple word counting 🟠")
    
    return results

# ---- Integration Functions ----

def test_embedding_methods():
    """
    Test all embedding methods with sample data
    """
    print("\n🧪 TESTING ALL EMBEDDING METHODS")
    print("="*40)
    
    sample_texts = [
        "Software engineer with Python and machine learning experience",
        "Data scientist skilled in statistical analysis and deep learning"
    ]
    
    model = get_embedding_model()
    
    # This will automatically test and track which method gets used
    embeddings = model.get_embeddings(sample_texts)
    
    print(f"✅ Embeddings generated with shape: {embeddings.shape}")
    print(f"🎯 Method used: {embedding_tracker.last_method_used}")
    
    # Test similarity calculation
    similarity = calculate_cosine_similarity_with_tracking(sample_texts[0], sample_texts[1])
    print(f"📊 Similarity score: {similarity:.3f}")
    
    return embeddings

def get_runtime_status():
    """
    Get current runtime status - call this from your views or API
    """
    return embedding_tracker.get_current_status()

def print_embedding_status():
    """
    Print current embedding status - call this anytime for debugging
    """
    embedding_tracker.print_status_report()

# ---- Startup Detection ----

def initialize_with_detection():
    """
    Initialize system and detect capabilities
    """
    print("\n🚀 INITIALIZING EMBEDDING SYSTEM WITH DETECTION")
    print("="*55)
    
    # Run diagnostics
    capabilities = diagnose_embedding_capabilities()
    
    # Initialize models
    try:
        model = get_embedding_model()
        print(f"✅ Embedding model initialized")
    except Exception as e:
        print(f"❌ Embedding model initialization failed: {e}")
    
    # Run test
    try:
        test_embedding_methods()
        print("✅ All tests completed")
    except Exception as e:
        print(f"❌ Tests failed: {e}")
    
    # Print initial status
    embedding_tracker.print_status_report()
    
    return capabilities

# ---- Replace your existing functions ----

# Replace the existing calculate_cosine_similarity function
calculate_cosine_similarity = calculate_cosine_similarity_with_tracking

# Update the EmbeddingModel class
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = EmbeddingModelWithTracking.get_instance()
        except Exception as e:
            logger.error(f"Failed to get embedding model: {e}")
            # Create a dummy model for fallback
            class DummyEmbeddingModel:
                def is_initialized(self):
                    return False
                def get_embeddings(self, texts):
                    embedding_tracker.record_method_usage("Random", 0.001, False)
                    return np.random.rand(len(texts), 100)
            embedding_model = DummyEmbeddingModel()
    return embedding_model

def test_embedding_methods():
    """
    Test all embedding methods with sample data
    """
    print("\n🧪 TESTING ALL EMBEDDING METHODS")
    print("="*40)
    
    sample_texts = [
        "Software engineer with Python and machine learning experience",
        "Data scientist skilled in statistical analysis and deep learning"
    ]
    
    try:
        model = get_embedding_model()
        
        # This will automatically test and track which method gets used
        embeddings = model.get_embeddings(sample_texts)
        
        print(f"✅ Embeddings generated with shape: {embeddings.shape}")
        print(f"🎯 Method used: {embedding_tracker.last_method_used}")
        
        # Test similarity calculation
        similarity = calculate_cosine_similarity(sample_texts[0], sample_texts[1])
        print(f"📊 Similarity score: {similarity:.3f}")
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return None
    
    
# Initialize tracking on module load
try:
    embedding_tracker = EmbeddingMethodTracker()
    print("🔧 Embedding tracker initialized")
except Exception as e:
    print(f"⚠️ Embedding tracker initialization failed: {e}")
    
    
# ---- Runtime Detection System ----
class EmbeddingMethodTracker:
    """Tracks which embedding method is being used at runtime"""
    
    def __init__(self):
        self.method_usage = defaultdict(int)
        self.method_timings = defaultdict(list)
        self.current_session_methods = []
        self.last_method_used = None
        self.session_start = datetime.now()
    
    def record_method_usage(self, method_name: str, duration: float = 0.0, success: bool = True):
        """Record which method was used and its performance"""
        timestamp = datetime.now()
        
        # Update counters
        self.method_usage[method_name] += 1
        self.method_timings[method_name].append(duration)
        self.last_method_used = method_name
        
        # Record for current session
        self.current_session_methods.append({
            'method': method_name,
            'timestamp': timestamp.isoformat(),
            'duration': duration,
            'success': success
        })
        
        # Log the usage
        logger.info(f"🔧 EMBEDDING METHOD: {method_name} | Duration: {duration:.3f}s | Success: {success}")
    
    def get_current_status(self) -> dict:
        """Get current runtime status"""
        total_calls = sum(self.method_usage.values())
        
        status = {
            'session_duration': str(datetime.now() - self.session_start),
            'total_embedding_calls': total_calls,
            'last_method_used': self.last_method_used,
            'primary_method': max(self.method_usage.items(), key=lambda x: x[1])[0] if self.method_usage else "None",
            'method_distribution': dict(self.method_usage),
            'method_percentages': {},
            'current_capability_level': self._get_capability_level()
        }
        
        # Calculate percentages
        if total_calls > 0:
            for method, count in self.method_usage.items():
                status['method_percentages'][method] = round((count / total_calls) * 100, 2)
        
        return status
    
    def _get_capability_level(self) -> str:
        """Determine current system capability level"""
        primary = max(self.method_usage.items(), key=lambda x: x[1])[0] if self.method_usage else "None"
        
        if primary == "DistilBERT":
            return "🟢 OPTIMAL - Using transformer embeddings"
        elif primary == "TF-IDF":
            return "🟡 GOOD - Using statistical embeddings" 
        elif primary == "SimpleWord":
            return "🟠 BASIC - Using word-based embeddings"
        elif primary == "Random":
            return "🔴 EMERGENCY - Using random embeddings"
        else:
            return "⚪ UNKNOWN - No embedding calls detected"

# Initialize tracker
embedding_tracker = EmbeddingMethodTracker()


def test_embedding_methods():
    """Test all embedding methods with sample data"""
    print("\n🧪 TESTING ALL EMBEDDING METHODS")
    print("="*40)
    
    sample_texts = [
        "Software engineer with Python and machine learning experience",
        "Data scientist skilled in statistical analysis and deep learning"
    ]
    
    try:
        # Test similarity calculation (this will trigger tracking)
        similarity = calculate_cosine_similarity(sample_texts[0], sample_texts[1])
        
        print(f"✅ Test completed!")
        print(f"📊 Similarity score: {similarity:.3f}")
        print(f"🎯 Method used: {embedding_tracker.last_method_used}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

def get_runtime_status():
    """Get current runtime status"""
    return embedding_tracker.get_current_status()

def print_embedding_status():
    """Print current embedding status"""
    status = get_runtime_status()
    
    print("\n" + "="*60)
    print("🔍 EMBEDDING METHOD RUNTIME DETECTION REPORT")
    print("="*60)
    print(f"📊 Total Embedding Calls: {status['total_embedding_calls']}")
    print(f"🎯 Current Capability: {status['current_capability_level']}")
    print(f"⚡ Last Method Used: {status['last_method_used'] or 'None'}")
    print(f"👑 Primary Method: {status['primary_method']}")
    print("="*60)