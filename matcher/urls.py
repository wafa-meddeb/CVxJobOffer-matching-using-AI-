from django.urls import path, include
from . import views

urlpatterns = [
    path('jobs/', views.jobs_view, name='jobs'),
    path('resumes/', views.resumes_view, name='resumes'),
    path('matching/', views.matching_view, name='matching'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('manage-jobs/', views.manage_jobs_view, name='manage_jobs'),
    
    
    path('create-job/', views.create_job_offer, name='create_job'),

]
