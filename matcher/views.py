from django.shortcuts import render
import uuid, json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import JobOffer

# AI imports
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .ai_utils import build_prompt_job_offer, parse_with_llm


@csrf_exempt
def create_job_offer(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        raw_text = data.get('text', '')

        # Build prompt and parse using Zephyr
        prompt = build_prompt_job_offer(raw_text)
        try:
            parsed = parse_with_llm(prompt)
        except Exception as e:
            return JsonResponse({'error': 'LLM parsing failed', 'details': str(e)}, status=400)

        # Create and save JobOffer object
        job = JobOffer.objects.create(
            job_id=str(uuid.uuid4()),
            title=parsed.get('title', ''),
            company=parsed.get('company', ''),
            location=parsed.get('location', ''),
            description=parsed.get('description', ''),
            requirements=parsed.get('requirements', []),
            nice_to_have=parsed.get('nice_to_have', []),
            contract_type=parsed.get('contract_type', ''),
            experience_level=parsed.get('experience_level', ''),
            languages=parsed.get('languages', []),
            technologies=parsed.get('technologies', []),
            salary_range=parsed.get('salary_range', '')
        )

        return JsonResponse({'message': 'Job offer created', 'title': job.title})

    return JsonResponse({'error': 'Invalid request method'}, status=400)


def jobs_view(request):
    return render(request, 'jobs.html')

def resumes_view(request):
    return render(request, 'resumes.html')

def matching_view(request):
    return render(request, 'matching.html')

def analytics_view(request):
    return render(request, 'analytics.html')

def manage_jobs_view(request):
    return render(request, 'manage_jobs.html')
