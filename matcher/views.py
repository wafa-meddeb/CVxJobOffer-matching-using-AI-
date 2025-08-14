import uuid, json
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods,require_GET, require_POST
from django.contrib import messages
from django.urls import reverse
from django.db.models import Case, When, F, Count, IntegerField, FloatField, Value


from django.db.models import Count, Avg, Max, Min, Q
from collections import Counter


from django.db.models import Q, Count
from django.utils import timezone
from datetime import datetime, timedelta
from django.utils import timezone
from django.core.paginator import Paginator
from .models import JobOffer, CV, MatchResult
from .ai_utils import match_resume_to_job, prepare_text_for_embedding

# AI imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .ai_utils import (
    build_prompt_job_offer, 
    parse_cv_with_llm,
    parse_job_offer_with_llm,
    parse_cv_from_pdf,  # Updated import
    extract_text_from_pdf,
    embedding_tracker, 
    print_embedding_status, 
    get_runtime_status,
    diagnose_embedding_capabilities
)




import os
from .models import CV
from .utils.resume_filters import resume_filter_manager
from django.conf import settings
import tempfile
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

import logging
logger = logging.getLogger(__name__)

# Initialize AI models at startup
try:
    initialize_models()
except Exception as e:
    logger.error(f"Failed to initialize AI models at startup: {e}")

# [Keep all your existing view functions for job offers unchanged]
@csrf_exempt
def create_job_offer(request):
    if request.method == 'POST':
        try:
            # Log the incoming request for debugging
            logger.info(f"Received POST request to create job offer")
            logger.info(f"Content-Type: {request.content_type}")
            logger.info(f"Request body (first 200 chars): {request.body[:200]}")
            
            # Parse JSON with error handling
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                return JsonResponse({
                    'error': 'Invalid JSON format', 
                    'details': f'JSON parsing failed: {str(e)}'
                }, status=400)
            
            # Get the text with proper key handling
            raw_text = data.get('text', '').strip()
            print(raw_text)
            
            # Also try alternative key names in case frontend sends different key
            if not raw_text:
                raw_text = data.get('job_text', '').strip()
            if not raw_text:
                raw_text = data.get('description', '').strip()
            
            logger.info(f"Extracted text length: {len(raw_text) if raw_text else 0}")
            
            # Validate input
            if not raw_text:
                logger.warning("No job text provided in request")
                return JsonResponse({
                    'error': 'Job text is required', 
                    'details': 'Please provide job offer text in the "text" field'
                }, status=400)
            
            if len(raw_text) < 10:
                logger.warning(f"Job text too short: {len(raw_text)} characters")
                return JsonResponse({
                    'error': 'Job text too short', 
                    'details': 'Please provide more detailed job offer text'
                }, status=400)
            
            if len(raw_text) > 50000:
                logger.warning(f"Job text too long: {len(raw_text)} characters")
                return JsonResponse({
                    'error': 'Job text too long', 
                    'details': 'Please limit job text to 50,000 characters'
                }, status=400)

            # Parse with LLM
            try:
                logger.info("Starting LLM parsing...")
                parsed = parse_job_offer_with_llm(raw_text)
                print("Parsed result:", parsed)
                logger.info("LLM parsing completed successfully")
                logger.info(f"Parsed data keys: {list(parsed.keys()) if parsed else 'None'}")
                
            except ValueError as e:
                logger.error(f"LLM parsing validation error: {e}")
                # Log the raw LLM response for debugging
                try:
                    # Try to get the raw response from the LLM to debug
                    from .ai_utils import _chat_completion, build_prompt_job_offer
                    prompt = build_prompt_job_offer(raw_text)
                    raw_response = _chat_completion([
                        {"role": "system", "content": "You are a precise JSON extractor for HR."},
                        {"role": "user", "content": prompt},
                    ])
                    logger.error(f"Raw LLM response: {raw_response[:500]}")
                    print(f"Raw LLM response: {raw_response}")
                except Exception as debug_e:
                    logger.error(f"Could not get raw LLM response: {debug_e}")
                
                return JsonResponse({
                    'error': 'LLM parsing failed', 
                    'details': f'JSON validation error: {str(e)}'
                }, status=422)
                
            except RuntimeError as e:
                logger.error(f"LLM API error: {e}")
                return JsonResponse({
                    'error': 'LLM API error', 
                    'details': f'API connection failed: {str(e)}'
                }, status=503)
                
            except Exception as e:
                logger.error(f"Unexpected LLM error: {e}")
                return JsonResponse({
                    'error': 'LLM parsing failed', 
                    'details': f'Unexpected error: {str(e)}'
                }, status=500)

            # Validate parsed data
            if not parsed or not isinstance(parsed, dict):
                logger.error(f"Invalid parsed data: {type(parsed)}")
                return JsonResponse({
                    'error': 'Invalid parsing result', 
                    'details': 'LLM returned invalid data format'
                }, status=500)

            # Create job offer with error handling
            try:
                job_uuid = str(uuid.uuid4())  # Generate UUID once
                
                job = JobOffer.objects.create(
                    id=job_uuid,                      # ✅ Set the primary key
                    job_id=job_uuid,                  # ✅ Set job_id (using same UUID)
                    title=parsed.get('title', 'Untitled Position'),
                    company=parsed.get('company', 'Unknown Company'),
                    location=parsed.get('location', ''),
                    description=parsed.get('description', ''),
                    requirements=parsed.get('requirements', []),
                    nice_to_have=parsed.get('nice_to_have', []),
                    contract_type=parsed.get('contract_type', ''),
                    experience_level=parsed.get('experience_level', ''),
                    languages=parsed.get('languages', []),
                    technologies=parsed.get('technologies', []),
                    salary_range = parsed.get('salary_range') or ''

                )
                
                logger.info(f"Successfully created job offer: {job.job_id}")
                
                return JsonResponse({
                    'success': True,
                    'message': 'Job offer created successfully', 
                    'job_id': job.job_id,
                    'title': job.title,
                    'company': job.company,
                    'parsed_data': parsed  # Include parsed data for debugging
                }, status=201)
                
            except Exception as e:
                logger.error(f"Database error creating job offer: {e}")
                return JsonResponse({
                    'error': 'Database error', 
                    'details': f'Failed to save job offer: {str(e)}'
                }, status=500)

        except Exception as e:
            logger.error(f"Unexpected error in create_job_offer: {e}")
            return JsonResponse({
                'error': 'Internal server error', 
                'details': str(e) if settings.DEBUG else 'Please contact support'
            }, status=500)

    else:
        logger.warning(f"Invalid request method: {request.method}")
        return JsonResponse({
            'error': 'Method not allowed', 
            'details': 'Use POST method to create job offers'
        }, status=405)

# [Keep all other job-related views unchanged - jobs_view, manage_jobs_view, job_list, job_detail, job_edit, job_delete, etc.]

def jobs_view(request):
    return render(request, 'jobs.html')

def matching_view(request):
    return render(request, 'matching.html')

def analytics_view(request):
    return render(request, 'analytics.html')

def manage_jobs_view(request):
    # Fetch all job offers from database, ordered by most recent first
    jobs = JobOffer.objects.all().order_by('-created_at')
    
    context = {
        'jobs': jobs,
        'total_jobs': jobs.count()
    }
    
    return render(request, 'manage_jobs.html', context)

def job_list(request):
    """Display list of jobs"""
    jobs = JobOffer.objects.all().order_by('-created_at')
    return render(request, 'manage_jobs.html', {'jobs': jobs})

def job_detail(request, job_id):
    """View job details"""
    job = get_object_or_404(JobOffer, job_id=job_id)
    return render(request, 'job_detail.html', {'job': job})

def job_edit(request, job_id):
    """Edit job"""
    job = get_object_or_404(JobOffer, job_id=job_id)
    
    if request.method == 'POST':
        # Update job fields based on form data
        job.title = request.POST.get('title', job.title)
        job.company = request.POST.get('company', job.company)
        job.location = request.POST.get('location', job.location)
        job.description = request.POST.get('description', job.description)
        job.contract_type = request.POST.get('contract_type', job.contract_type)
        job.experience_level = request.POST.get('experience_level', job.experience_level)
        job.salary_range = request.POST.get('salary_range', job.salary_range)
        
        # Handle technologies (JSONField)
        technologies = request.POST.get('technologies', '')
        if technologies:
            job.technologies = [tech.strip() for tech in technologies.split(',') if tech.strip()]
        else:
            job.technologies = []
        
        # Handle languages (JSONField)
        languages = request.POST.get('languages', '')
        if languages:
            job.languages = [lang.strip() for lang in languages.split(',') if lang.strip()]
        else:
            job.languages = []
        
        # Handle requirements (JSONField)
        requirements = request.POST.get('requirements', '')
        if requirements:
            job.requirements = [req.strip() for req in requirements.split('\n') if req.strip()]
        else:
            job.requirements = []
            
        # Handle nice_to_have (JSONField)
        nice_to_have = request.POST.get('nice_to_have', '')
        if nice_to_have:
            job.nice_to_have = [item.strip() for item in nice_to_have.split('\n') if item.strip()]
        else:
            job.nice_to_have = []
        
        job.save()
        messages.success(request, f'Job "{job.title}" updated successfully!')
        return redirect('job_list')  # Adjust URL name as needed
    
    return render(request, 'job_edit.html', {'job': job})

@require_http_methods(["POST"])
@csrf_exempt
def job_delete(request, job_id):
    """Delete job via AJAX"""
    try:
        job = get_object_or_404(JobOffer, job_id=job_id)
        job_title = job.title
        job.delete()
        return JsonResponse({
            'success': True, 
            'message': f'Job "{job_title}" deleted successfully!'
        })
    except Exception as e:
        return JsonResponse({
            'success': False, 
            'message': f'Error deleting job: {str(e)}'
        }, status=500)

def job_delete_confirm(request, job_id):
    """Delete job with confirmation page"""
    job = get_object_or_404(JobOffer, job_id=job_id)
    
    if request.method == 'POST':
        job_title = job.title
        job.delete()
        messages.success(request, f'Job "{job_title}" deleted successfully!')
        return redirect('job_list')
    
    return render(request, 'job_delete_confirm.html', {'job': job})


################## manage resumes views ##########################################

def resumes_view(request):
    resumes = CV.objects.all().order_by('-uploaded_at')
    return render(request, 'resumes.html', {'resumes': resumes})



def manage_resumes_view(request):
    """
    Enhanced resume management view with functional filters
    """
    # Start with all resumes
    resumes = CV.objects.all()
    total_resumes = resumes.count()
    
    # Apply search filter
    search_query = request.GET.get('search', '').strip()
    if search_query:
        resumes = resume_filter_manager.filter_by_search(resumes, search_query)
    
    # Apply date filter
    date_filter = request.GET.get('date_filter', '')
    if date_filter:
        resumes = apply_date_filter(resumes, date_filter)
    
    # Apply experience filter
    experience_filter = request.GET.get('experience_filter', '')
    if experience_filter:
        resumes = resume_filter_manager.filter_by_experience_level(resumes, experience_filter)
    
    # Apply education filter
    education_filter = request.GET.get('education_filter', '')
    if education_filter:
        resumes = resume_filter_manager.filter_by_education_level(resumes, education_filter)
    
    # Order by upload date (newest first)
    resumes = resumes.order_by('-uploaded_at')
    
    # Add pagination (optional)
    paginator = Paginator(resumes, 12)  # 12 resumes per page to fit grid layout
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'resumes': page_obj if paginator.num_pages > 1 else resumes,
        'total_resumes': total_resumes,
        'filtered_count': resumes.count(),
        'search_query': search_query,
        'date_filter': date_filter,
        'experience_filter': experience_filter,
        'education_filter': education_filter,
        'page_obj': page_obj if paginator.num_pages > 1 else None,
    }
    
    return render(request, 'manage_resumes.html', context)


def apply_date_filter(queryset, date_filter):
    """
    Apply date-based filters
    """
    now = timezone.now()
    
    if date_filter == 'today':
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return queryset.filter(uploaded_at__gte=start_date)
    
    elif date_filter == 'this_week':
        start_date = now - timedelta(days=7)
        return queryset.filter(uploaded_at__gte=start_date)
    
    elif date_filter == 'this_month':
        start_date = now - timedelta(days=30)
        return queryset.filter(uploaded_at__gte=start_date)
    
    elif date_filter == 'last_3_months':
        start_date = now - timedelta(days=90)
        return queryset.filter(uploaded_at__gte=start_date)
    
    return queryset


# API endpoint for AJAX filtering (optional for better UX)
def resume_filter_api(request):
    """
    API endpoint for AJAX-based filtering without page reload
    """
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    try:
        # Apply the same filtering logic
        resumes = CV.objects.all()
        total_resumes = resumes.count()
        
        # Apply filters
        search_query = request.GET.get('search', '').strip()
        if search_query:
            resumes = resume_filter_manager.filter_by_search(resumes, search_query)
        
        date_filter = request.GET.get('date_filter', '')
        if date_filter:
            resumes = apply_date_filter(resumes, date_filter)
        
        experience_filter = request.GET.get('experience_filter', '')
        if experience_filter:
            resumes = resume_filter_manager.filter_by_experience_level(resumes, experience_filter)
        
        education_filter = request.GET.get('education_filter', '')
        if education_filter:
            resumes = resume_filter_manager.filter_by_education_level(resumes, education_filter)
        
        resumes = resumes.order_by('-uploaded_at')
        
        # Serialize data for JSON response
        resume_data = []
        for resume in resumes[:50]:  # Limit to first 50 for performance
            # Process skills safely
            skills = []
            if resume.skills:
                if isinstance(resume.skills, list):
                    skills = resume.skills
                elif isinstance(resume.skills, str):
                    try:
                        skills = json.loads(resume.skills)
                    except:
                        skills = []
            
            # Process experience safely
            experience = []
            if resume.experience:
                if isinstance(resume.experience, list):
                    experience = resume.experience
                elif isinstance(resume.experience, str):
                    try:
                        experience = json.loads(resume.experience)
                    except:
                        experience = []
            
            # Process education safely
            education = []
            if resume.education:
                if isinstance(resume.education, list):
                    education = resume.education
                elif isinstance(resume.education, str):
                    try:
                        education = json.loads(resume.education)
                    except:
                        education = []
            
            resume_data.append({
                'cv_id': resume.cv_id,
                'name': resume.name,
                'email': resume.email,
                'phone': resume.phone,
                'uploaded_at': resume.uploaded_at.strftime('%b %d, %Y'),
                'summary': resume.summary[:150] if resume.summary else '',
                'skills': skills[:8],  # First 8 skills
                'skills_total': len(skills),
                'experience': experience[:2],  # First 2 experiences
                'experience_total': len(experience),
                'education': education[:2],  # First 2 educations
                'education_total': len(education),
                'languages': resume.languages[:4] if resume.languages else [],
                'certificates': resume.certificates if resume.certificates else [],
            })
        
        return JsonResponse({
            'success': True,
            'resumes': resume_data,
            'total_count': total_resumes,
            'filtered_count': resumes.count()
        })
        
    except Exception as e:
        logger.error(f"Error in resume_filter_api: {e}")
        return JsonResponse({
            'success': False,
            'error': 'Filter operation failed'
        }, status=500)


def search_suggestions_api(request):
    """
    API endpoint to provide search suggestions based on existing resume data
    """
    query = request.GET.get('q', '').strip().lower()
    
    if len(query) < 2:
        return JsonResponse({'suggestions': []})
    
    suggestions = set()
    
    # Get suggestions from various fields
    resumes = CV.objects.all()[:100]  # Limit for performance
    
    for resume in resumes:
        # Check skills
        if resume.skills:
            skills = resume.skills if isinstance(resume.skills, list) else []
            if isinstance(resume.skills, str):
                try:
                    skills = json.loads(resume.skills)
                except:
                    skills = []
            
            for skill in skills:
                if query in str(skill).lower():
                    suggestions.add(str(skill))
        
        # Check name
        if query in resume.name.lower():
            suggestions.add(resume.name)
        
        # Check summary keywords
        if resume.summary and query in resume.summary.lower():
            words = resume.summary.split()
            for word in words:
                if query in word.lower() and len(word) > 3:
                    suggestions.add(word)
    
    # Limit and sort suggestions
    suggestions = sorted(list(suggestions))[:10]
    
    return JsonResponse({'suggestions': suggestions})


# Keep your existing upload_resumes, delete_resume, etc. functions as they are
@csrf_exempt
def upload_resumes(request):
    """Handle multiple resume uploads using text extraction from PDF"""
    if request.method == 'POST':
        try:
            uploaded_files = request.FILES.getlist('resumes')
            
            if not uploaded_files:
                return JsonResponse({
                    'success': False,
                    'error': 'No files uploaded'
                }, status=400)
            
            results = []
            processed_count = 0
            
            for uploaded_file in uploaded_files:
                try:
                    logger.info(f"Processing file: {uploaded_file.name}")
                    
                    # Validate file type
                    if not uploaded_file.name.lower().endswith('.pdf'):
                        results.append({
                            'name': uploaded_file.name,
                            'status': 'Skipped - Not a PDF file'
                        })
                        logger.warning(f"Skipped non-PDF file: {uploaded_file.name}")
                        continue
                    
                    # Validate file size (e.g., max 10MB)
                    if uploaded_file.size > 10 * 1024 * 1024:
                        results.append({
                            'name': uploaded_file.name,
                            'status': 'Skipped - File too large (max 10MB)'
                        })
                        logger.warning(f"Skipped large file: {uploaded_file.name} ({uploaded_file.size} bytes)")
                        continue
                    
                    # Save file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        for chunk in uploaded_file.chunks():
                            temp_file.write(chunk)
                        temp_file_path = temp_file.name
                    
                    try:
                        # Extract text from PDF and process with text LLM
                        logger.info(f"Extracting text from PDF: {uploaded_file.name}")
                        
                        # First, extract text to validate PDF content
                        extracted_text = extract_text_from_pdf(temp_file_path)
                        
                        if not extracted_text or len(extracted_text.strip()) < 50:
                            results.append({
                                'name': uploaded_file.name,
                                'status': 'Failed - Could not extract readable text from PDF'
                            })
                            logger.error(f"No readable text found in: {uploaded_file.name}")
                            continue
                        
                        logger.info(f"Extracted {len(extracted_text)} characters from {uploaded_file.name}")
                        
                        # Process with text-based LLM parsing
                        logger.info(f"Processing with LLM: {uploaded_file.name}")
                        parsed_data = parse_cv_from_pdf(temp_file_path)
                        
                        # Validate parsed data
                        if not parsed_data or not isinstance(parsed_data, dict):
                            results.append({
                                'name': uploaded_file.name,
                                'status': 'Failed - Could not parse CV data'
                            })
                            logger.error(f"Invalid parsed data for: {uploaded_file.name}")
                            continue
                        
                        # Generate unique CV ID
                        cv_id = str(uuid.uuid4())
                        
                        # Save file to Django's media storage
                        file_name = f"resumes/{cv_id}_{uploaded_file.name}"
                        saved_path = default_storage.save(file_name, uploaded_file)
                        
                        # Create CV record in database
                        cv = CV.objects.create(
                            id=cv_id,  # Set primary key
                            cv_id=cv_id,
                            name=parsed_data.get('name', 'Unknown'),
                            email=parsed_data.get('email', ''),
                            phone=parsed_data.get('phone', ''),
                            summary=parsed_data.get('summary', ''),
                            skills=parsed_data.get('skills', []),
                            education=parsed_data.get('education', []),
                            experience=parsed_data.get('experience', []),
                            languages=parsed_data.get('languages', []),
                            certificates=parsed_data.get('certificates', []),
                            file=saved_path
                        )
                        
                        results.append({
                            'name': parsed_data.get('name', uploaded_file.name),
                            'status': 'Successfully processed and saved',
                            'cv_id': cv.cv_id,
                            'extracted_text_length': len(extracted_text)
                        })
                        processed_count += 1
                        
                        logger.info(f"Successfully processed CV: {cv.name} (ID: {cv.cv_id})")
                        
                    except ValueError as ve:
                        logger.error(f"Validation error processing {uploaded_file.name}: {ve}")
                        results.append({
                            'name': uploaded_file.name,
                            'status': f'Failed - Text extraction or LLM parsing error: {str(ve)}'
                        })
                    except RuntimeError as re:
                        logger.error(f"Runtime error processing {uploaded_file.name}: {re}")
                        results.append({
                            'name': uploaded_file.name,
                            'status': f'Failed - API connection error: {str(re)}'
                        })
                    except Exception as processing_error:
                        logger.error(f"Unexpected error processing {uploaded_file.name}: {processing_error}")
                        results.append({
                            'name': uploaded_file.name,
                            'status': f'Failed - Processing error: {str(processing_error)}'
                        })
                    
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_file_path)
                            logger.debug(f"Cleaned up temp file: {temp_file_path}")
                        except Exception as cleanup_error:
                            logger.warning(f"Could not clean up temp file {temp_file_path}: {cleanup_error}")
                            
                except Exception as file_error:
                    logger.error(f"Error handling file {uploaded_file.name}: {file_error}")
                    results.append({
                        'name': uploaded_file.name,
                        'status': f'Failed - File handling error: {str(file_error)}'
                    })
            
            # Prepare response
            success_message = f"Successfully processed {processed_count} out of {len(uploaded_files)} resume(s)"
            logger.info(f"Upload batch completed: {success_message}")
            
            return JsonResponse({
                'success': True,
                'message': success_message,
                'processed_count': processed_count,
                'total_files': len(uploaded_files),
                'results': results
            })
            
        except Exception as e:
            logger.error(f"Unexpected error in upload_resumes: {e}")
            return JsonResponse({
                'success': False,
                'error': f'Upload failed: {str(e)}'
            }, status=500)
    
    else:
        return JsonResponse({
            'success': False,
            'error': 'Method not allowed. Use POST to upload resumes.'
        }, status=405)


@require_http_methods(["DELETE"])
def delete_resume(request, cv_id):
    try:
        resume = CV.objects.get(cv_id=cv_id)
        resume.delete()
        return JsonResponse({'success': True})
    except CV.DoesNotExist:
        return JsonResponse({'success': False, 'message': 'Resume not found'}, status=404)


def resume_detail(request, cv_id):
    resume = get_object_or_404(CV, cv_id=cv_id)
    return render(request, 'resume_detail.html', {'resume': resume})


def resume_delete_confirm(request, cv_id):
    resume = get_object_or_404(CV, cv_id=cv_id)

    if request.method == 'POST':
        resume.delete()
        return redirect('manage_resumes')

    return render(request, 'resume_delete_confirmation.html', {'resume': resume})


def manage_resumes_view(request):
    """
    Enhanced resume management view with functional filters
    """
    # Start with all resumes
    resumes = CV.objects.all()
    total_resumes = resumes.count()
    
    # Apply search filter
    search_query = request.GET.get('search', '').strip()
    if search_query:
        resumes = resume_filter_manager.filter_by_search(resumes, search_query)
    
    # Apply date filter
    date_filter = request.GET.get('date_filter', '')
    if date_filter:
        resumes = apply_date_filter(resumes, date_filter)
    
    # Apply experience filter
    experience_filter = request.GET.get('experience_filter', '')
    if experience_filter:
        resumes = resume_filter_manager.filter_by_experience_level(resumes, experience_filter)
    
    # Apply education filter
    education_filter = request.GET.get('education_filter', '')
    if education_filter:
        resumes = resume_filter_manager.filter_by_education_level(resumes, education_filter)
    
    # Order by upload date (newest first)
    resumes = resumes.order_by('-uploaded_at')
    
    # Add pagination (optional)
    paginator = Paginator(resumes, 12)  # 12 resumes per page to fit grid layout
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'resumes': page_obj if paginator.num_pages > 1 else resumes,
        'total_resumes': total_resumes,
        'filtered_count': resumes.count(),
        'search_query': search_query,
        'date_filter': date_filter,
        'experience_filter': experience_filter,
        'education_filter': education_filter,
        'page_obj': page_obj if paginator.num_pages > 1 else None,
    }
    
    return render(request, 'manage_resumes.html', context)

def apply_date_filter(queryset, date_filter):
    """
    Apply date-based filters
    """
    now = timezone.now()
    
    if date_filter == 'today':
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return queryset.filter(uploaded_at__gte=start_date)
    
    elif date_filter == 'this_week':
        start_date = now - timedelta(days=7)
        return queryset.filter(uploaded_at__gte=start_date)
    
    elif date_filter == 'this_month':
        start_date = now - timedelta(days=30)
        return queryset.filter(uploaded_at__gte=start_date)
    
    elif date_filter == 'last_3_months':
        start_date = now - timedelta(days=90)
        return queryset.filter(uploaded_at__gte=start_date)
    
    return queryset

# Quick test view for debugging
def test_filters_view(request):
    """
    Quick test view - access via URL to test filters
    """
    output = []
    output.append("<h2>Resume Filter Test Results</h2>")
    
    try:
        from .utils.resume_filters import resume_filter_manager
        
        # Get all resumes
        all_resumes = CV.objects.all()
        total_count = all_resumes.count()
        output.append(f"<p><strong>Total resumes:</strong> {total_count}</p>")
        
        if total_count == 0:
            output.append("<p style='color: red;'>No resumes found. Please upload some resumes first.</p>")
            output.append("<p><a href='/upload-resumes/'>Go to Upload Page</a></p>")
        else:
            # Test search filter
            output.append("<h3>Search Filter Test:</h3>")
            search_queries = ['python', 'engineer', 'java', 'developer']
            
            for query in search_queries:
                try:
                    filtered = resume_filter_manager.filter_by_search(all_resumes, query)
                    count = filtered.count()
                    output.append(f"<p>&nbsp;&nbsp;'{query}': {count} results</p>")
                    if count > 0:
                        output.append(f"<p>&nbsp;&nbsp;&nbsp;&nbsp;First: {filtered.first().name}</p>")
                except Exception as e:
                    output.append(f"<p>&nbsp;&nbsp;'{query}': Error - {str(e)}</p>")
            
            # Test experience filter
            output.append("<h3>Experience Filter Test:</h3>")
            experience_levels = ['0-1', '2-5', '6-10', '10+']
            
            for level in experience_levels:
                try:
                    filtered = resume_filter_manager.filter_by_experience_level(all_resumes, level)
                    count = filtered.count()
                    output.append(f"<p>&nbsp;&nbsp;'{level} years': {count} results</p>")
                    if count > 0:
                        first_resume = filtered.first()
                        calculated_years = resume_filter_manager._calculate_total_experience(first_resume)
                        output.append(f"<p>&nbsp;&nbsp;&nbsp;&nbsp;First: {first_resume.name} ({calculated_years} years)</p>")
                except Exception as e:
                    output.append(f"<p>&nbsp;&nbsp;'{level}': Error - {str(e)}</p>")
            
            # Sample data inspection
            output.append("<h3>Sample Resume Data:</h3>")
            sample = all_resumes.first()
            output.append(f"<p><strong>Name:</strong> {sample.name}</p>")
            output.append(f"<p><strong>Skills:</strong> {sample.skills}</p>")
            output.append(f"<p><strong>Experience type:</strong> {type(sample.experience)}</p>")
            if sample.experience:
                exp_str = str(sample.experience)[:300] + "..." if len(str(sample.experience)) > 300 else str(sample.experience)
                output.append(f"<p><strong>Experience preview:</strong> {exp_str}</p>")
        
    except Exception as e:
        output.append(f"<p style='color: red;'>Error: {str(e)}</p>")
    
    output.append("<p><strong>Test Complete!</strong></p>")
    output.append("<p><a href='/manage-resumes/'>Go to Resume Management</a></p>")
    
    return HttpResponse(''.join(output))



# ================== MATCHING API ENDPOINTS ==================

@csrf_exempt
@require_http_methods(["POST"])
def match_resumes_api(request):
    """
    API endpoint to match selected resumes against a selected job offer
    """
    try:
        logger.info("Received matching request")
        
        # Parse request data
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in request: {e}")
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON format in request body'
            }, status=400)
        
        job_id = data.get('job_id')
        resume_ids = data.get('resume_ids', [])
        
        logger.info(f"Matching request: job_id={job_id}, resume_ids={resume_ids}")
        
        # Validate input
        if not job_id:
            return JsonResponse({
                'success': False,
                'error': 'Job ID is required'
            }, status=400)
        
        if not resume_ids or not isinstance(resume_ids, list):
            return JsonResponse({
                'success': False,
                'error': 'At least one resume ID is required'
            }, status=400)
        
        # Get job offer
        try:
            # Handle both 'id' and 'job_id' fields
            job = JobOffer.objects.get(Q(id=job_id) | Q(job_id=job_id))
            logger.info(f"Found job: {job.title} at {job.company}")
        except JobOffer.DoesNotExist:
            logger.error(f"Job offer with ID {job_id} not found")
            return JsonResponse({
                'success': False,
                'error': f'Job offer with ID {job_id} not found'
            }, status=404)
        except Exception as e:
            logger.error(f"Error fetching job: {e}")
            return JsonResponse({
                'success': False,
                'error': f'Error fetching job: {str(e)}'
            }, status=500)
        
        # Get resumes
        try:
            # Handle both 'id' and 'cv_id' fields
            resumes = CV.objects.filter(Q(id__in=resume_ids) | Q(cv_id__in=resume_ids))
            found_resume_ids = list(resumes.values_list('cv_id', flat=True))
            
            # Check for missing resumes
            missing_ids = [rid for rid in resume_ids if rid not in found_resume_ids]
            if missing_ids:
                logger.warning(f"Missing resume IDs: {missing_ids}")
            
            if not resumes.exists():
                logger.error("No valid resumes found")
                return JsonResponse({
                    'success': False,
                    'error': 'No valid resumes found for provided IDs'
                }, status=404)
            
            logger.info(f"Found {resumes.count()} resumes to match")
            
        except Exception as e:
            logger.error(f"Error fetching resumes: {e}")
            return JsonResponse({
                'success': False,
                'error': f'Error fetching resumes: {str(e)}'
            }, status=500)
        
        # Prepare job data for matching
        job_data = {
            'job_id': job.job_id or job.id,
            'title': job.title or '',
            'company': job.company or '',
            'location': job.location or '',
            'description': job.description or '',
            'requirements': job.requirements or [],
            'nice_to_have': job.nice_to_have or [],
            'contract_type': job.contract_type or '',
            'experience_level': job.experience_level or '',
            'languages': job.languages or [],
            'technologies': job.technologies or [],
            'salary_range': job.salary_range or ''
        }
        
        logger.debug(f"Job data prepared: {job_data['title']} with {len(job_data['technologies'])} technologies")
        
        # Process each resume
        matching_results = []
        
        for resume in resumes:
            try:
                logger.info(f"Processing resume: {resume.name}")
                
                # Prepare CV data for matching
                cv_data = {
                    'cv_id': resume.cv_id or resume.id,
                    'name': resume.name or 'Unknown',
                    'email': resume.email or '',
                    'phone': resume.phone or '',
                    'summary': resume.summary or '',
                    'skills': resume.skills or [],
                    'education': resume.education or [],
                    'experience': resume.experience or [],
                    'languages': resume.languages or [],
                    'certificates': resume.certificates or []
                }
                
                # Perform matching
                logger.debug(f"Starting matching for {resume.name}")
                match_result = match_resume_to_job(job_data, cv_data)
                
                # Ensure all required fields are present
                match_result.setdefault('cv_id', cv_data['cv_id'])
                match_result.setdefault('name', cv_data['name'])
                match_result.setdefault('title', '')
                match_result.setdefault('summary', cv_data['summary'])
                match_result.setdefault('similarity_score', 0.0)
                match_result.setdefault('grammar_penalty', 0.0)
                match_result.setdefault('final_score', 0.0)
                match_result.setdefault('matched_skills', [])
                match_result.setdefault('skill_overlap_ratio', 0.0)
                
                # Save match result to database
                try:
                    match_uuid = str(uuid.uuid4())
                    match_record, created = MatchResult.objects.update_or_create(
                        cv=resume,
                        job_offer=job,
                        defaults={
                            'uuid': match_uuid,
                            'score': match_result['final_score']
                        }
                    )
                    logger.debug(f"Saved match result to database: {match_record.uuid}")
                except Exception as db_error:
                    logger.error(f"Error saving match result to database: {db_error}")
                    # Continue processing even if DB save fails
                
                matching_results.append(match_result)
                logger.info(f"Match completed for {resume.name}: final_score={match_result['final_score']:.3f}")
                
            except Exception as e:
                logger.error(f"Error matching resume {resume.cv_id}: {e}")
                # Add error result
                error_result = {
                    'cv_id': resume.cv_id or resume.id,
                    'name': resume.name or 'Unknown',
                    'title': 'Processing Error',
                    'summary': f'Error processing resume: {str(e)}',
                    'similarity_score': 0.0,
                    'grammar_penalty': 0.0,
                    'final_score': 0.0,
                    'matched_skills': [],
                    'skill_overlap_ratio': 0.0
                }
                matching_results.append(error_result)
        
        # Sort results by final score (highest first)
        matching_results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        logger.info(f"Matching completed. Processed {len(matching_results)} resumes")
        
        # Prepare response
        response_data = {
            'success': True,
            'message': f'Successfully matched {len(matching_results)} resumes',
            'results': matching_results,
            'job': {
                'id': job.job_id or job.id,
                'title': job.title,
                'company': job.company
            },
            'processed_count': len(matching_results),
            'requested_count': len(resume_ids)
        }
        
        logger.info(f"Returning response with {len(matching_results)} results")
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Unexpected error in match_resumes_api: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)

@require_http_methods(["GET"])
def get_jobs_api(request):
    """
    API endpoint to get all job offers for the matching interface
    """
    try:
        logger.info("Fetching jobs for matching interface")
        jobs = JobOffer.objects.all().order_by('-created_at')
        
        jobs_data = []
        for job in jobs:
            try:
                # Safely handle potentially None fields
                title = job.title or 'Untitled Position'
                company = job.company or 'Unknown Company'
                location = job.location or 'Location not specified'
                description = job.description or 'No description available'
                
                # Truncate long descriptions
                if len(description) > 200:
                    description = description[:200] + '...'
                
                # Safely handle JSON fields
                skills = job.technologies or []
                if isinstance(skills, str):
                    try:
                        skills = json.loads(skills)
                    except:
                        skills = []
                
                requirements = job.requirements or []
                if isinstance(requirements, str):
                    try:
                        requirements = json.loads(requirements)
                    except:
                        requirements = []
                
                nice_to_have = job.nice_to_have or []
                if isinstance(nice_to_have, str):
                    try:
                        nice_to_have = json.loads(nice_to_have)
                    except:
                        nice_to_have = []
                
                job_item = {
                    'id': job.job_id or job.id,  # Use job_id if available, fallback to id
                    'title': title,
                    'company': company,
                    'location': location,
                    'type': job.contract_type or 'Type not specified',
                    'description': description,
                    'skills': skills[:10],  # Limit to first 10 skills
                    'experience_level': job.experience_level or 'Not specified',
                    'posted': job.created_at.strftime('%Y-%m-%d') if job.created_at else 'Date not available',
                    'requirements': requirements[:5],  # Limit requirements
                    'nice_to_have': nice_to_have[:5]  # Limit nice-to-have
                }
                
                jobs_data.append(job_item)
                
            except Exception as job_error:
                logger.error(f"Error processing job {job.id}: {job_error}")
                # Add minimal job data
                jobs_data.append({
                    'id': job.job_id or job.id,
                    'title': job.title or 'Error loading job',
                    'company': job.company or 'Unknown',
                    'location': job.location or 'Unknown',
                    'type': 'Unknown',
                    'description': 'Error loading job description',
                    'skills': [],
                    'experience_level': 'Unknown',
                    'posted': 'Unknown',
                    'requirements': [],
                    'nice_to_have': []
                })
        
        logger.info(f"Returning {len(jobs_data)} jobs")
        return JsonResponse({
            'success': True,
            'jobs': jobs_data,
            'count': len(jobs_data)
        })
        
    except Exception as e:
        logger.error(f"Error in get_jobs_api: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Failed to fetch jobs: {str(e)}',
            'jobs': [],
            'count': 0
        }, status=500)

@require_http_methods(["GET"])
def get_resumes_api(request):
    """
    API endpoint to get all resumes for the matching interface
    """
    try:
        logger.info("Fetching resumes for matching interface")
        resumes = CV.objects.all().order_by('-uploaded_at')
        
        resumes_data = []
        for resume in resumes:
            try:
                # Safely handle potentially None fields
                name = resume.name or 'Unknown Candidate'
                
                # Calculate years of experience (simple heuristic)
                experience_years = "Not specified"
                if resume.experience:
                    try:
                        if isinstance(resume.experience, list) and len(resume.experience) > 0:
                            experience_years = f"~{len(resume.experience)} positions"
                        elif isinstance(resume.experience, str):
                            # Try to extract years from text
                            import re
                            years_match = re.search(r'(\d+)\s*years?', resume.experience.lower())
                            if years_match:
                                experience_years = f"{years_match.group(1)} years"
                    except Exception as exp_error:
                        logger.debug(f"Error parsing experience for {name}: {exp_error}")
                
                # Get top skills (limit to 5)
                top_skills = []
                if resume.skills:
                    try:
                        if isinstance(resume.skills, list):
                            top_skills = resume.skills[:5]
                        elif isinstance(resume.skills, str):
                            # Try to parse as JSON first
                            try:
                                skills_list = json.loads(resume.skills)
                                if isinstance(skills_list, list):
                                    top_skills = skills_list[:5]
                                else:
                                    top_skills = resume.skills.split(',')[:5]
                            except:
                                top_skills = resume.skills.split(',')[:5]
                    except Exception as skills_error:
                        logger.debug(f"Error parsing skills for {name}: {skills_error}")
                
                # Safely handle summary
                summary = resume.summary or 'No summary available'
                if len(summary) > 150:
                    summary = summary[:150] + '...'
                
                resume_item = {
                    'id': resume.cv_id or resume.id,  # Use cv_id if available, fallback to id
                    'name': name,
                    'title': experience_years,  # Using experience as title for now
                    'experience': experience_years,
                    'location': 'Not specified',  # Add location field to CV model if needed
                    'skills': [str(skill).strip() for skill in top_skills if skill],
                    'matchScore': None,  # Default score, will be calculated during matching
                    'summary': summary,
                    'email': resume.email or 'Email not provided',
                    'phone': resume.phone or 'Phone not provided',
                    'uploaded_at': resume.uploaded_at.strftime('%Y-%m-%d') if resume.uploaded_at else 'Date not available'
                }
                
                resumes_data.append(resume_item)
                
            except Exception as resume_error:
                logger.error(f"Error processing resume {resume.id}: {resume_error}")
                # Add minimal resume data
                resumes_data.append({
                    'id': resume.cv_id or resume.id,
                    'name': resume.name or 'Error loading resume',
                    'title': 'Error',
                    'experience': 'Unknown',
                    'location': 'Unknown',
                    'skills': [],
                    'matchScore': 0,
                    'summary': 'Error loading resume summary',
                    'email': 'Unknown',
                    'phone': 'Unknown',
                    'uploaded_at': 'Unknown'
                })
        
        logger.info(f"Returning {len(resumes_data)} resumes")
        return JsonResponse({
            'success': True,
            'resumes': resumes_data,
            'count': len(resumes_data)
        })
        
    except Exception as e:
        logger.error(f"Error in get_resumes_api: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Failed to fetch resumes: {str(e)}',
            'resumes': [],
            'count': 0
        }, status=500)

@require_http_methods(["GET"])
def get_match_history_api(request):
    """
    API endpoint to get matching history for analytics
    """
    try:
        logger.info("Fetching match history")
        # Get recent matches
        matches = MatchResult.objects.select_related('cv', 'job_offer').order_by('-matched_at')[:100]
        
        matches_data = []
        for match in matches:
            try:
                match_item = {
                    'id': match.uuid,
                    'cv_name': match.cv.name if match.cv else 'Unknown',
                    'job_title': match.job_offer.title if match.job_offer else 'Unknown',
                    'company': match.job_offer.company if match.job_offer else 'Unknown',
                    'score': float(match.score) if match.score else 0.0,
                    'matched_at': match.matched_at.strftime('%Y-%m-%d %H:%M:%S') if match.matched_at else 'Unknown'
                }
                matches_data.append(match_item)
            except Exception as match_error:
                logger.error(f"Error processing match {match.uuid}: {match_error}")
        
        logger.info(f"Returning {len(matches_data)} match records")
        return JsonResponse({
            'success': True,
            'matches': matches_data,
            'count': len(matches_data)
        })
        
    except Exception as e:
        logger.error(f"Error in get_match_history_api: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Failed to fetch match history: {str(e)}',
            'matches': [],
            'count': 0
        }, status=500)

# ================== ENHANCED MATCHING VIEW ==================

def matching_view(request):
    """
    Enhanced matching view with real-time data loading
    """
    context = {
        'page_title': 'Resume Matching',
        'api_endpoints': {
            'jobs': '/api/jobs/',
            'resumes': '/api/resumes/',
            'match': '/api/match-resumes/',
            'history': '/api/match-history/'
        }
    }
    return render(request, 'matching.html', context)

# ================== ANALYTICS ENDPOINTS ==================

@require_http_methods(["GET"])
def matching_analytics_api(request):
    """
    API endpoint for matching analytics and statistics
    """
    try:
        # Get basic statistics
        total_jobs = JobOffer.objects.count()
        total_resumes = CV.objects.count()
        total_matches = MatchResult.objects.count()
        
        # Get average scores
        from django.db.models import Avg, Max, Min
        score_stats = MatchResult.objects.aggregate(
            avg_score=Avg('score'),
            max_score=Max('score'),
            min_score=Min('score')
        )
        
        SCORES_ARE_FRACTIONS = True  # change to False if you store 0–100

        avg_score = (score_stats['avg_score'] or 0) * (100 if SCORES_ARE_FRACTIONS else 1)
        max_score = (score_stats['max_score'] or 0) * (100 if SCORES_ARE_FRACTIONS else 1)
        min_score = (score_stats['min_score'] or 0) * (100 if SCORES_ARE_FRACTIONS else 1)
        
        # Get top matches
        top_matches = MatchResult.objects.select_related('cv', 'job_offer').order_by('-score')[:10]
        top_matches_data = []
        for match in top_matches:
            try:
                match_data = {
                    'cv_name': match.cv.name if match.cv else 'Unknown',
                    'job_title': match.job_offer.title if match.job_offer else 'Unknown',
                    'company': match.job_offer.company if match.job_offer else 'Unknown',
                    'score': float(match.score) if match.score else 0.0,
                    'matched_at': match.matched_at.strftime('%Y-%m-%d') if match.matched_at else 'Unknown'
                }
                top_matches_data.append(match_data)
            except Exception as match_error:
                logger.error(f"Error processing top match: {match_error}")
        
        # Get matching trends (last 30 days)
        from datetime import datetime, timedelta
        thirty_days_ago = timezone.now() - timedelta(days=30)
        recent_matches = MatchResult.objects.filter(matched_at__gte=thirty_days_ago)
        
        # Group by date
        daily_matches = {}
        for match in recent_matches:
            try:
                date_key = match.matched_at.strftime('%Y-%m-%d')
                if date_key not in daily_matches:
                    daily_matches[date_key] = {'count': 0, 'avg_score': 0, 'total_score': 0}
                daily_matches[date_key]['count'] += 1
                daily_matches[date_key]['total_score'] += float(match.score) if match.score else 0
            except Exception as trend_error:
                logger.debug(f"Error processing trend data: {trend_error}")
        
        # Calculate averages
        for date_key in daily_matches:
            if daily_matches[date_key]['count'] > 0:
                daily_matches[date_key]['avg_score'] = daily_matches[date_key]['total_score'] / daily_matches[date_key]['count']
        
        logger.info("Analytics data prepared successfully")
        
        return JsonResponse({
            'success': True,
            'statistics': {
                'total_jobs': total_jobs,
                'total_resumes': total_resumes,
                'total_matches': total_matches,
                'avg_score': round(avg_score, 3),
                'max_score': round(max_score, 3),
                'min_score': round(min_score, 3)

            },
            'top_matches': top_matches_data,
            'daily_trends': daily_matches
        })
        
    except Exception as e:
        logger.error(f"Error in matching_analytics_api: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Failed to fetch analytics: {str(e)}'
        }, status=500)
        
        
        
##################### Analytics views ##############################

# Update your existing analytics_view function
def analytics_view(request):
    """
    Enhanced analytics view with comprehensive data
    """
    context = {
        'page_title': 'Analytics Dashboard',
        'api_endpoints': {
            'analytics': '/api/analytics/',
        }
    }
    return render(request, 'analytics.html', context)


def safe_analytics_operation(operation_name):
    """Decorator for safe analytics operations with consistent error handling"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {operation_name}: {str(e)}")
                return [] if 'list' in operation_name.lower() else {}
        return wrapper
    return decorator

# Add this new API endpoint to your views.py
@require_http_methods(["GET"])
def analytics_api(request):
    """
    Comprehensive analytics API endpoint that matches the front-end contract.
    Returns:
      - statistics (all percent-scaled)
      - score_distribution (buckets)
      - top_matches (scores in percent)
      - recent_activity (scores in percent)
      - top_companies
      - top_skills
      - experience_levels
      - weekly_trends (array; scores in percent)
    """
    try:
        logger.info("Fetching comprehensive analytics data")

        total_jobs = JobOffer.objects.count()
        total_resumes = CV.objects.count()
        total_matches = MatchResult.objects.count()

        # Period filter
        period = request.GET.get('period', 'all')  # all, 30, 7
        matches_queryset = MatchResult.objects.all()
        if period != 'all':
            days = int(period)
            cutoff_date = timezone.now() - timedelta(days=days)
            matches_queryset = matches_queryset.filter(matched_at__gte=cutoff_date)

        # Scale detection
        uses_percent = matches_queryset.filter(score__gt=1).exists()
        threshold = 70 if uses_percent else 0.70

        # Stats
        score_stats = matches_queryset.aggregate(
            avg_score=Avg('score'),
            max_score=Max('score'),
            min_score=Min('score')
        )

        # Distribution (bucketed; scale-aware)
        score_distribution = calculate_score_distribution(matches_queryset)
        


        # Top / recent (ensure percent-scaled)
        top_matches = get_top_matches(10)          # make sure helper scales with _scale_score
        recent_activity = get_recent_activity(10)  # make sure helper scales with _scale_score

        # Insights
        top_companies = get_top_companies(10)
        top_skills = get_top_skills(15)
        experience_levels = get_experience_levels()

        # Trends as array (what the FE expects)
        weekly_trends = get_weekly_trends()

        # Success rate (threshold respects storage scale)
        high_score_matches = matches_queryset.filter(score__gte=threshold).count()
        success_rate = (high_score_matches / total_matches * 100) if total_matches else 0

        response_data = {
            'success': True,
            'statistics': {
                'total_jobs': total_jobs,
                'total_resumes': total_resumes,
                'total_matches': total_matches,
                'avg_score': round(_scale_score(score_stats['avg_score']), 2) if score_stats['avg_score'] is not None else 0.0,
                'max_score': round(_scale_score(score_stats['max_score']), 2) if score_stats['max_score'] is not None else 0.0,
                'min_score': round(_scale_score(score_stats['min_score']), 2) if score_stats['min_score'] is not None else 0.0,
                'success_rate': round(success_rate, 1),
            },
            'api_signature': 'analytics_v3', 
            'score_distribution': score_distribution,
            'top_matches': top_matches,
            'recent_activity': recent_activity,
            'top_companies': top_companies,
            'top_skills': top_skills,
            'experience_levels': experience_levels,
            'weekly_trends': weekly_trends,      # <<< important: array key
            'period': period
        }

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error in analytics_api: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Failed to fetch analytics: {str(e)}',
            'statistics': {
                'total_jobs': 0, 'total_resumes': 0, 'total_matches': 0,
                'avg_score': 0, 'max_score': 0, 'min_score': 0, 'success_rate': 0
            },
            'score_distribution': {},
            'top_matches': [],
            'recent_activity': [],
            'top_companies': [],
            'top_skills': [],
            'experience_levels': [],
            'weekly_trends': []
        }, status=500)


def calculate_score_distribution(matches_queryset):
    """
    Works for scores stored as 0–1 or 0–100.
    Normalizes per row to percent and aggregates in one query.
    """
    try:
        qs = matches_queryset.annotate(
            pct=Case(
                When(score__isnull=True, then=Value(0.0)),
                When(score__lte=1.0, then=F('score') * 100.0),
                default=F('score'),
                output_field=FloatField(),
            )
        )

        agg = qs.aggregate(
            b90_100=Count(Case(When(pct__gte=90, then=1), output_field=IntegerField())),
            b80_89 =Count(Case(When(pct__gte=80, pct__lt=90, then=1), output_field=IntegerField())),
            b70_79 =Count(Case(When(pct__gte=70, pct__lt=80, then=1), output_field=IntegerField())),
            b60_69 =Count(Case(When(pct__gte=60, pct__lt=70, then=1), output_field=IntegerField())),
            blt60  =Count(Case(When(pct__lt=60, then=1), output_field=IntegerField())),
        )

        return {
            '90-100': agg['b90_100'] or 0,
            '80-89':  agg['b80_89']  or 0,
            '70-79':  agg['b70_79']  or 0,
            '60-69':  agg['b60_69']  or 0,
            '<60':    agg['blt60']   or 0,
        }
    except Exception as e:
        logger.error(f"Error calculating score distribution: {e}")
        return {'90-100': 0, '80-89': 0, '70-79': 0, '60-69': 0, '<60': 0}



def get_top_matches(limit=10):
    """
    Get highest scoring matches with details
    """
    try:
        top_matches = MatchResult.objects.select_related('cv', 'job_offer').order_by('-score')[:limit]
        
        matches_data = []
        for match in top_matches:
            try:
                match_data = {
                    'cv_name': match.cv.name if match.cv else 'Unknown Candidate',
                    'job_title': match.job_offer.title if match.job_offer else 'Unknown Position',
                    'company': match.job_offer.company if match.job_offer else 'Unknown Company',
                    'score': round(_scale_score(m.score), 1) if m.score is not None else 0.0,
                    'matched_at': match.matched_at.strftime('%Y-%m-%d') if match.matched_at else 'Unknown',
                    'cv_id': match.cv.cv_id if match.cv else None,
                    'job_id': match.job_offer.job_id if match.job_offer else None
                }
                matches_data.append(match_data)
            except Exception as match_error:
                logger.error(f"Error processing top match: {match_error}")
        
        return matches_data
        
    except Exception as e:
        logger.error(f"Error getting top matches: {e}")
        return []

def get_recent_activity(limit=10):
    """
    Get most recent matching activity
    """
    try:
        recent_matches = MatchResult.objects.select_related('cv', 'job_offer').order_by('-matched_at')[:limit]
        
        activity_data = []
        for match in recent_matches:
            try:
                activity_item = {
                    'cv_name': match.cv.name if match.cv else 'Unknown Candidate',
                    'job_title': match.job_offer.title if match.job_offer else 'Unknown Position',
                    'company': match.job_offer.company if match.job_offer else 'Unknown Company',
                    'score': round(_scale_score(m.score), 1) if m.score is not None else 0.0,
                    'date': match.matched_at.strftime('%Y-%m-%d') if match.matched_at else 'Unknown',
                    'time_ago': get_time_ago(match.matched_at) if match.matched_at else 'Unknown'
                }
                activity_data.append(activity_item)
            except Exception as activity_error:
                logger.error(f"Error processing recent activity: {activity_error}")
        
        return activity_data
        
    except Exception as e:
        logger.error(f"Error getting recent activity: {e}")
        return []

def get_top_companies(limit=10):
    """
    Get companies with most job postings
    """
    try:
        companies = JobOffer.objects.values('company').annotate(
            count=Count('id')
        ).order_by('-count')[:limit]
        
        companies_data = []
        for company in companies:
            if company['company']:  # Skip empty company names
                companies_data.append({
                    'name': company['company'],
                    'count': company['count']
                })
        
        return companies_data
        
    except Exception as e:
        logger.error(f"Error getting top companies: {e}")
        return []

def get_top_skills(limit=15):
    """
    Analyze most frequently mentioned skills across jobs and resumes
    """
    try:
        skills_counter = Counter()
        
        # Analyze skills from job offers (technologies field)
        job_skills = JobOffer.objects.exclude(technologies__isnull=True).values_list('technologies', flat=True)
        for skills_data in job_skills:
            try:
                if isinstance(skills_data, list):
                    skills_list = skills_data
                elif isinstance(skills_data, str):
                    try:
                        skills_list = json.loads(skills_data)
                    except:
                        skills_list = [skill.strip() for skill in skills_data.split(',') if skill.strip()]
                else:
                    continue
                
                for skill in skills_list:
                    if skill and isinstance(skill, str):
                        skills_counter[skill.strip().title()] += 1
                        
            except Exception as skill_error:
                logger.debug(f"Error processing job skill: {skill_error}")
        
        # Analyze skills from resumes
        resume_skills = CV.objects.exclude(skills__isnull=True).values_list('skills', flat=True)
        for skills_data in resume_skills:
            try:
                if isinstance(skills_data, list):
                    skills_list = skills_data
                elif isinstance(skills_data, str):
                    try:
                        skills_list = json.loads(skills_data)
                    except:
                        skills_list = [skill.strip() for skill in skills_data.split(',') if skill.strip()]
                else:
                    continue
                
                for skill in skills_list:
                    if skill and isinstance(skill, str):
                        skills_counter[skill.strip().title()] += 1
                        
            except Exception as skill_error:
                logger.debug(f"Error processing resume skill: {skill_error}")
        
        # Get top skills
        top_skills = [
            {'name': skill, 'count': count}
            for skill, count in skills_counter.most_common(limit)
        ]
        
        return top_skills
        
    except Exception as e:
        logger.error(f"Error getting top skills: {e}")
        return []

import re
from collections import Counter

# Friendly labels for common raw values
_LEVEL_MAP = {
    'junior': 'Junior (0-2 years)',
    'jr': 'Junior (0-2 years)',
    'mid': 'Mid-level (2-5 years)',
    'mid-level': 'Mid-level (2-5 years)',
    'middle': 'Mid-level (2-5 years)',
    'intermediate': 'Mid-level (2-5 years)',
    'senior': 'Senior (5+ years)',
    'sr': 'Senior (5+ years)',
    'lead': 'Lead/Principal (7+ years)',
    'principal': 'Lead/Principal (7+ years)',
    'entry': 'Entry Level',
    'entry-level': 'Entry Level',
    'intern': 'Internship',
    'internship': 'Internship',
}

# Heuristics when the DB field is empty — guess from the job title
_TITLE_PATTERNS = [
    (re.compile(r'\b(intern|internship)\b', re.I), 'Internship'),
    (re.compile(r'\b(entry[-\s]?level)\b', re.I), 'Entry Level'),
    (re.compile(r'\b(junior|jr\.?)\b', re.I), 'Junior (0-2 years)'),
    (re.compile(r'\b(mid[-\s]?level|middle|intermediate|mid)\b', re.I), 'Mid-level (2-5 years)'),
    (re.compile(r'\b(senior|sr\.?)\b', re.I), 'Senior (5+ years)'),
    (re.compile(r'\b(lead|principal)\b', re.I), 'Lead/Principal (7+ years)'),
]

def _normalize_level(raw: str | None) -> str | None:
    if not raw:
        return None
    k = str(raw).strip().lower()
    if not k:  # handles empty strings (column is NOT NULL but may be '')
        return None
    return _LEVEL_MAP.get(k) or _LEVEL_MAP.get(k.replace(' ', '-'))  # try with dash

def _infer_from_title(title: str | None) -> str | None:
    if not title:
        return None
    for pat, label in _TITLE_PATTERNS:
        if pat.search(title):
            return label
    return None

def get_experience_levels():
    """
    Build experience level distribution from JobOffer.experience_level.
    Falls back to inferring the level from JobOffer.title if the field is blank.
    Returns a list of {'name': ..., 'count': ...} sorted by count desc.
    """
    try:
        # Pull both fields; the column is NOT NULL but may contain empty strings
        rows = JobOffer.objects.values_list('experience_level', 'title')
        counter = Counter()

        for raw_level, title in rows:
            label = _normalize_level(raw_level) or _infer_from_title(title) or 'Unspecified'
            counter[label] += 1

        # If you prefer to hide 'Unspecified', filter it out here:
        items = [
            {'name': name, 'count': cnt}
            for name, cnt in counter.items()
            if cnt > 0
        ]

        items.sort(key=lambda x: x['count'], reverse=True)
        return items

    except Exception as e:
        logger.error(f"Error getting experience levels: {e}")
        return []
    
@require_GET
def experience_levels_api(request):
    items = get_experience_levels()  # the helper you showed
    return JsonResponse({'success': True, 'items': items})


def get_weekly_trends():
    try:
        out = []
        now = timezone.now()
        uses_percent = MatchResult.objects.filter(score__gt=1).exists()

        for i in range(8):
            start = now - timedelta(weeks=i+1)
            end   = now - timedelta(weeks=i)

            qs = MatchResult.objects.filter(matched_at__gte=start, matched_at__lt=end)
            cnt = qs.count()
            avg = qs.aggregate(avg_score=Avg('score'))['avg_score'] or 0.0
            avg = avg if uses_percent else (avg * 100.0)

            out.append({
                'week': f'W{i+1}',
                'date': start.date().isoformat(),   # ISO date → always parsable
                'matches': cnt,
                'avg_score': round(avg, 1)
            })

        return list(reversed(out))
    except Exception as e:
        logger.error(f"Error getting weekly trends: {e}")
        return []


def get_time_ago(datetime_obj):
    """
    Get human-readable time difference
    """
    try:
        if not datetime_obj:
            return 'Unknown'
        
        now = timezone.now()
        if datetime_obj.tzinfo:
            from django.utils import timezone
            now = timezone.now()
        
        diff = now - datetime_obj
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"
            
    except Exception as e:
        logger.debug(f"Error calculating time ago: {e}")
        return "Unknown"

# Additional helper function for getting match statistics by date range
def get_match_statistics_by_period(days=30):
    """
    Get detailed match statistics for a specific period
    """
    try:
        cutoff_date = timezone.now() - timedelta(days=days)
        recent_matches = MatchResult.objects.filter(matched_at__gte=cutoff_date)
        
        total_recent = recent_matches.count()
        if total_recent == 0:
            return {
                'total_matches': 0,
                'avg_score': 0,
                'high_quality_matches': 0,
                'success_rate': 0
            }
        
        avg_score = recent_matches.aggregate(avg_score=Avg('score'))['avg_score'] or 0
        high_quality = recent_matches.filter(score__gte=80).count()
        success_rate = (high_quality / total_recent) * 100
        
        return {
            'total_matches': total_recent,
            'avg_score': round(_scale_score(avg_score), 2),
            'high_quality_matches': high_quality,
            'success_rate': round(success_rate, 1)
        }
        
    except Exception as e:
        logger.error(f"Error getting period statistics: {e}")
        return {
            'total_matches': 0,
            'avg_score': 0,
            'high_quality_matches': 0,
            'success_rate': 0
        }        
        
        
        
######## chat views ##################
# ---- helpers ---------------------------------------------------------------

def _parse_period(request):
    """Return (start_dt or None) based on ?period=all|30|7"""
    period = request.GET.get('period', 'all')
    if period == 'all':
        return None
    try:
        days = int(period)
    except ValueError:
        days = 30
    return timezone.now() - timezone.timedelta(days=days)

def _scale_score(value):
    """
    If score is stored as 0–1, convert to 0–100.
    If it's already 0–100, keep it.
    """
    if value is None:
        return 0.0
    return float(value) * 100 if value <= 1.0 else float(value)

# ---- /api/analytics/matches/top/ ------------------------------------------

@require_GET
def get_top_matches_api(request):
    """
    Returns highest scoring matches.
    Optional params:
      - limit (int, default 10)
      - period: all|30|7
    """
    try:
        limit = int(request.GET.get('limit', 10))
        start_dt = _parse_period(request)

        qs = (MatchResult.objects
              .select_related('cv', 'job_offer')
              .order_by('-score'))
        if start_dt:
            qs = qs.filter(matched_at__gte=start_dt)

        items = []
        for m in qs[:limit]:
            items.append({
                'cv_name': m.cv.name if m.cv else 'Unknown Candidate',
                'job_title': m.job_offer.title if m.job_offer else 'Unknown Position',
                'company': m.job_offer.company if m.job_offer else 'Unknown Company',
                'score': round(_scale_score(m.score), 1),
                'matched_at': m.matched_at.isoformat() if m.matched_at else None,
                'cv_id': m.cv.cv_id if m.cv else None,
                'job_id': m.job_offer.job_id if m.job_offer else None,
            })

        return JsonResponse({'success': True, 'items': items, 'count': len(items)})
    except Exception as e:
        logger.exception("get_top_matches_api failed")
        return JsonResponse({'success': False, 'error': str(e), 'items': []}, status=500)

# ---- /api/analytics/activity/recent/ --------------------------------------

@require_GET
def get_recent_activity_api(request):
    """
    Returns latest match events (sorted by matched_at desc).
    Optional params:
      - limit (int, default 10)
      - period: all|30|7
    """
    try:
        limit = int(request.GET.get('limit', 10))
        start_dt = _parse_period(request)

        qs = (MatchResult.objects
              .select_related('cv', 'job_offer')
              .filter(matched_at__isnull=False)            # avoid null dates
              .order_by('-matched_at'))

        if start_dt:
            qs = qs.filter(matched_at__gte=start_dt)

        def _time_ago(dt):
            if not dt:
                return 'Unknown'
            now = timezone.now()
            diff = now - dt
            if diff.days > 0:
                d = diff.days
                return f"{d} day{'s' if d > 1 else ''} ago"
            if diff.seconds >= 3600:
                h = diff.seconds // 3600
                return f"{h} hour{'s' if h > 1 else ''} ago"
            if diff.seconds >= 60:
                m = diff.seconds // 60
                return f"{m} minute{'s' if m > 1 else ''} ago"
            return "Just now"

        items = []
        for m in qs[:limit]:
            items.append({
                'cv_name'   : m.cv.name if m.cv else 'Unknown Candidate',
                'job_title' : m.job_offer.title if m.job_offer else 'Unknown Position',
                'company'   : m.job_offer.company if m.job_offer else 'Unknown Company',
                # always percent-scale (0–100) for the UI
                'score'     : round(_scale_score(m.score), 1) if m.score is not None else 0.0,
                # IMPORTANT: provide matched_at because the JS uses it
                'matched_at': m.matched_at.isoformat() if m.matched_at else None,
                # keep these for convenience/backward-compat
                'date'      : m.matched_at.date().isoformat() if m.matched_at else None,
                'time_ago'  : _time_ago(m.matched_at) if m.matched_at else 'Unknown',
            })

        return JsonResponse({'success': True, 'items': items, 'count': len(items)})
    except Exception as e:
        logger.exception("get_recent_activity_api failed")
        return JsonResponse({'success': False, 'error': str(e), 'items': []}, status=500)
# ---- /api/analytics/companies/ --------------------------------------------

@require_GET
def get_top_companies_api(request):
    """
    Returns companies with most job postings.
    Optional param:
      - limit (int, default 10)
    """
    try:
        limit = int(request.GET.get('limit', 10))
        companies = (JobOffer.objects
                     .values('company')
                     .annotate(count=Count('id'))
                     .order_by('-count')[:limit])

        items = [{'name': c['company'], 'count': c['count']}
                 for c in companies if c['company']]

        return JsonResponse({'success': True, 'items': items, 'count': len(items)})
    except Exception as e:
        logger.exception("get_top_companies_api failed")
        return JsonResponse({'success': False, 'error': str(e), 'items': []}, status=500)

# ---- /api/analytics/skills/ -----------------------------------------------

@require_GET
def get_top_skills_api(request):
    """
    Returns most frequent skills across job offers and resumes.
    Optional param:
      - limit (int, default 15)
    """
    try:
        limit = int(request.GET.get('limit', 15))

        # Aggregate skills from both JobOffer.technologies and CV.skills (JSONField or CSV)
        counter = {}

        def bump(skill):
            if not skill:
                return
            s = skill.strip()
            if not s:
                return
            key = s.title()
            counter[key] = counter.get(key, 0) + 1

        # Job offers
        for raw in JobOffer.objects.exclude(technologies__isnull=True).values_list('technologies', flat=True):
            if isinstance(raw, list):
                for s in raw:
                    if isinstance(s, str):
                        bump(s)
            elif isinstance(raw, str):
                # try JSON first
                try:
                    lst = json.loads(raw)
                    if isinstance(lst, list):
                        for s in lst:
                            if isinstance(s, str):
                                bump(s)
                    else:
                        bump(raw)
                except Exception:
                    for s in [p.strip() for p in raw.split(',') if p.strip()]:
                        bump(s)

        # Resumes
        for raw in CV.objects.exclude(skills__isnull=True).values_list('skills', flat=True):
            if isinstance(raw, list):
                for s in raw:
                    if isinstance(s, str):
                        bump(s)
            elif isinstance(raw, str):
                try:
                    lst = json.loads(raw)
                    if isinstance(lst, list):
                        for s in lst:
                            if isinstance(s, str):
                                bump(s)
                    else:
                        bump(raw)
                except Exception:
                    for s in [p.strip() for p in raw.split(',') if p.strip()]:
                        bump(s)

        # Top N
        items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        items = [{'name': k, 'count': v} for k, v in items]

        return JsonResponse({'success': True, 'items': items, 'count': len(items)})
    except Exception as e:
        logger.exception("get_top_skills_api failed")
        return JsonResponse({'success': False, 'error': str(e), 'items': []}, status=500)        
    
    
    
# ================== EMBEDDING MONITORING VIEWS ==================

def embedding_status_view(request):
    """
    View to check current embedding method status
    """
    try:
        status = get_runtime_status()
        
        # Add some additional context
        status['recommendations'] = []
        
        primary_method = status.get('primary_method', 'None')
        
        if primary_method == 'Random':
            status['recommendations'].append({
                'level': 'critical',
                'message': 'System is using random embeddings. Install transformers and torch for better performance.',
                'action': 'pip install transformers torch'
            })
        elif primary_method == 'SimpleWord':
            status['recommendations'].append({
                'level': 'warning', 
                'message': 'System is using basic word embeddings. Install scikit-learn for TF-IDF or transformers for best performance.',
                'action': 'pip install scikit-learn transformers torch'
            })
        elif primary_method == 'TF-IDF':
            status['recommendations'].append({
                'level': 'info',
                'message': 'System is using TF-IDF embeddings. Install transformers for optimal semantic understanding.',
                'action': 'pip install transformers torch'
            })
        elif primary_method == 'DistilBERT':
            status['recommendations'].append({
                'level': 'success',
                'message': 'System is using optimal DistilBERT embeddings. Performance is at maximum.',
                'action': 'No action needed'
            })
        
        return JsonResponse({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"Error getting embedding status: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def embedding_diagnostics_view(request):
    """
    Run diagnostic tests and return results
    """
    try:
        logger.info("Running embedding diagnostics...")
        
        # Run diagnostics
        capabilities = diagnose_embedding_capabilities()
        
        # Test embedding methods
        try:
            test_embeddings = test_embedding_methods()
            test_success = True
            embedding_shape = test_embeddings.shape if hasattr(test_embeddings, 'shape') else None
        except Exception as e:
            test_success = False
            embedding_shape = None
            logger.error(f"Embedding test failed: {e}")
        
        # Get current status
        current_status = get_runtime_status()
        
        return JsonResponse({
            'success': True,
            'diagnostics': {
                'capabilities': capabilities,
                'test_success': test_success,
                'embedding_shape': embedding_shape,
                'current_status': current_status
            }
        })
        
    except Exception as e:
        logger.error(f"Error running diagnostics: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def embedding_initialize_view(request):
    """
    Initialize embedding system with full detection
    """
    try:
        logger.info("Initializing embedding system with detection...")
        
        # Initialize with detection
        capabilities = initialize_with_detection()
        
        # Get status after initialization
        status = get_runtime_status()
        
        return JsonResponse({
            'success': True,
            'message': 'Embedding system initialized successfully',
            'capabilities': capabilities,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"Error initializing embedding system: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

# ================== CONSOLE OUTPUT CAPTURE VIEW ==================

import io
import sys
from contextlib import redirect_stdout

def embedding_status_report_view(request):
    """
    Get the full status report as formatted text
    """
    try:
        # Capture the console output
        output_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer):
            print_embedding_status()
        
        report_text = output_buffer.getvalue()
        
        return JsonResponse({
            'success': True,
            'report': report_text,
            'status': get_runtime_status()
        })
        
    except Exception as e:
        logger.error(f"Error generating status report: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

# ================== MONITORING MIDDLEWARE ==================

class EmbeddingMonitoringMiddleware:
    """
    Middleware to automatically track embedding usage on each request
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Store initial state
        initial_calls = sum(embedding_tracker.method_usage.values())
        
        response = self.get_response(request)
        
        # Check if embeddings were used in this request
        final_calls = sum(embedding_tracker.method_usage.values())
        
        if final_calls > initial_calls:
            # Embeddings were used, add header with method info
            response['X-Embedding-Method'] = embedding_tracker.last_method_used or 'Unknown'
            response['X-Embedding-Calls'] = str(final_calls)
        
        return response

# ================== TEMPLATE VIEW FOR MONITORING DASHBOARD ==================

def embedding_dashboard_view(request):
    """
    Render a dashboard for monitoring embedding methods
    """
    try:
        status = get_runtime_status()
        
        context = {
            'status': status,
            'page_title': 'Embedding Method Monitor'
        }
        
        return render(request, 'embedding_dashboard.html', context)
        
    except Exception as e:
        logger.error(f"Error rendering embedding dashboard: {e}")
        context = {
            'error': str(e),
            'page_title': 'Embedding Method Monitor - Error'
        }
        return render(request, 'embedding_dashboard.html', context)