# import osimport os
import re
import json
import requests
import base64
import io
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import os

# Load .env for local dev. In production, set GROQ_API_KEY as a real env var.
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Add this to test your token
print(f"API Key exists: {bool(GROQ_API_KEY)}")
print(f"API Key starts with: {GROQ_API_KEY[:10] if GROQ_API_KEY else 'None'}...")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to .env or your deployment environment.")

# ---- Groq API Configuration ----
ROUTER_URL = "https://api.groq.com/openai/v1/chat/completions"

# Text models
TEXT_MODEL = "llama3-8b-8192"  # Your existing text model
# Vision model
VISION_MODEL = "llava-v1.5-7b-4096-preview"  # For image/PDF processing

HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

# ---------------- Image/PDF Helpers ----------------

def encode_image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def pil_image_to_base64(pil_image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def pdf_to_images(pdf_path, dpi=150, max_pages=5):
    """Convert PDF to list of PIL images"""
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Converted PDF to {len(images)} images (processing first {max_pages})")
        return images[:max_pages]  # Limit pages to save tokens
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return []

# ---------------- Prompt Builders ----------------

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

def build_prompt_cv_vision() -> str:
    return """You are an HR assistant. Extract the following fields from this CV/resume image in pure JSON format:
Fields:
- name, email, phone
- summary
- skills (list of strings)
- education (list of objects with degree, school, years)
- experience (list of objects with title, company, years)
- languages (list of strings)
- certificates (list of strings)

Read all text carefully from the image and return ONLY valid JSON (no prose, no markdown fences).
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

def build_prompt_job_offer_vision() -> str:
    return """You are an HR assistant. Extract this job offer from the image into JSON with:
- title, company, location, description
- requirements (list of strings)
- nice_to_have (list of strings)  
- contract_type, experience_level
- languages (list), technologies (list), salary_range

Read all text carefully from the image and return ONLY valid JSON (no prose, no markdown fences).
"""

# ---------------- JSON Extraction (Your existing function) ----------------

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

def _vision_completion(prompt, base64_image, temperature: float = 0.2, max_tokens: int = 1000, timeout_s: int = 90) -> str:
    """
    Call Groq Vision API with image and text prompt.
    """
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    try:
        resp = requests.post(ROUTER_URL, headers=HEADERS, json=payload, timeout=timeout_s)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Vision API request failed: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"Vision API error {resp.status_code}: {resp.text[:400]}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected Vision API response format: {data}")

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

# ---------------- NEW: Vision-Based Parsing Functions ----------------

def parse_cv_from_image(image_path: str) -> dict:
    """
    Parse CV from an image file using vision model.
    """
    base64_image = encode_image_to_base64(image_path)
    prompt = build_prompt_cv_vision()
    
    content = _vision_completion(prompt, base64_image)
    return extract_json_from_text(content.strip())

def parse_cv_from_pdf(pdf_path: str, max_pages: int = 3) -> dict:
    """
    Parse CV from PDF by converting to images and processing with vision model.
    Combines results from multiple pages.
    """
    images = pdf_to_images(pdf_path, max_pages=max_pages)
    if not images:
        raise ValueError("Failed to convert PDF to images")
    
    # Process first page (usually contains main info)
    first_image_b64 = pil_image_to_base64(images[0])
    prompt = build_prompt_cv_vision()
    
    print(f"Processing CV from PDF page 1/{len(images)}...")
    content = _vision_completion(prompt, first_image_b64)
    
    # For multi-page PDFs, you might want to process additional pages
    # and merge the results, but for now we'll use just the first page
    # to stay within token limits
    
    return extract_json_from_text(content.strip())

def parse_job_offer_from_image(image_path: str) -> dict:
    """
    Parse job offer from an image file using vision model.
    """
    base64_image = encode_image_to_base64(image_path)
    prompt = build_prompt_job_offer_vision()
    
    content = _vision_completion(prompt, base64_image)
    return extract_json_from_text(content.strip())

def parse_job_offer_from_pdf(pdf_path: str, max_pages: int = 3) -> dict:
    """
    Parse job offer from PDF by converting to images and processing with vision model.
    """
    images = pdf_to_images(pdf_path, max_pages=max_pages)
    if not images:
        raise ValueError("Failed to convert PDF to images")
    
    # Process first page
    first_image_b64 = pil_image_to_base64(images[0])
    prompt = build_prompt_job_offer_vision()
    
    print(f"Processing job offer from PDF page 1/{len(images)}...")
    content = _vision_completion(prompt, first_image_b64)
    
    return extract_json_from_text(content.strip())