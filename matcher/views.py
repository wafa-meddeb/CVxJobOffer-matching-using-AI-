import uuid, json
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.urls import reverse
import logging

from .models import JobOffer

# AI imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .ai_utils import build_prompt_job_offer, parse_cv_with_llm,parse_job_offer_with_llm


logger = logging.getLogger(__name__)

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
        
        
# @csrf_exempt
# def create_job_offer(request):
#     if request.method == 'POST':
#         data = json.loads(request.body)
#         raw_text = data.get('text', '')

#         try:
#             parsed = parse_job_offer_with_llm(raw_text)  # ✅ just pass the raw job offer text
#         except Exception as e:
#             return JsonResponse({'error': 'LLM parsing failed', 'details': str(e)}, status=400)

#         job = JobOffer.objects.create(
#             job_id=str(uuid.uuid4()),
#             title=parsed.get('title', ''),
#             company=parsed.get('company', ''),
#             location=parsed.get('location', ''),
#             description=parsed.get('description', ''),
#             requirements=parsed.get('requirements', []),
#             nice_to_have=parsed.get('nice_to_have', []),
#             contract_type=parsed.get('contract_type', ''),
#             experience_level=parsed.get('experience_level', ''),
#             languages=parsed.get('languages', []),
#             technologies=parsed.get('technologies', []),
#             salary_range=parsed.get('salary_range', '')
#         )

#         return JsonResponse({'message': 'Job offer created', 'title': job.title})

#     return JsonResponse({'error': 'Invalid request method'}, status=400)



def jobs_view(request):
    return render(request, 'jobs.html')

def resumes_view(request):
    return render(request, 'resumes.html')

def matching_view(request):
    return render(request, 'matching.html')

def analytics_view(request):
    return render(request, 'analytics.html')



# MANAGE JOBS INTERFACE

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

# @require_http_methods(["DELETE"])
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

# Alternative non-AJAX delete view
def job_delete_confirm(request, job_id):
    """Delete job with confirmation page"""
    job = get_object_or_404(JobOffer, job_id=job_id)
    
    if request.method == 'POST':
        job_title = job.title
        job.delete()
        messages.success(request, f'Job "{job_title}" deleted successfully!')
        return redirect('job_list')
    
    return render(request, 'job_delete_confirm.html', {'job': job})