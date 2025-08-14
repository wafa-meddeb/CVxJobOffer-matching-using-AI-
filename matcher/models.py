from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.db.models import JSONField

class JobOffer(models.Model):
    id = models.CharField(max_length=255, primary_key=True)
    title = models.CharField(max_length=255)
    company = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    contract_type = models.CharField(max_length=255)
    experience_level = models.CharField(max_length=255)
    job_id = models.CharField(max_length=255, unique=True)
    languages = JSONField(default=list)
    nice_to_have = JSONField(default=list)
    requirements = JSONField(default=list)
    salary_range = models.CharField(max_length=255)
    technologies = JSONField(default=list)


class CV(models.Model):
    id = models.CharField(max_length=255, primary_key=True)  # Add this if needed
    cv_id = models.CharField(max_length=255, unique=True)
    name = models.CharField(max_length=255)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=50, blank=True)
    summary = models.TextField(blank=True)
    skills = JSONField(default=list, blank=True)
    education = JSONField(default=list, blank=True)
    experience = JSONField(default=list, blank=True)
    languages = JSONField(default=list, blank=True)
    certificates = JSONField(default=list, blank=True)
    file = models.FileField(upload_to='resumes/')
    uploaded_at = models.DateTimeField(auto_now_add=True)



class MatchResult(models.Model):
    uuid = models.CharField(max_length=255, unique=True)
    cv = models.ForeignKey(CV, on_delete=models.CASCADE)
    job_offer = models.ForeignKey(JobOffer, on_delete=models.CASCADE)
    score = models.FloatField()
    matched_at = models.DateTimeField(auto_now_add=True)

    
class HRUser(models.Model):
    user_id = models.CharField(max_length=255, unique=True)

   


