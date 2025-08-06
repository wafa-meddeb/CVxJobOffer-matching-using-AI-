from django.urls import path, include
from . import views

urlpatterns = [
    path('jobs/', views.jobs_view, name='jobs'),
    path('resumes/', views.resumes_view, name='resumes'),
    path('matching/', views.matching_view, name='matching'),
    path('analytics/', views.analytics_view, name='analytics'),
    path('manage-jobs/', views.manage_jobs_view, name='manage_jobs'),
    
    
    path('create-job/', views.create_job_offer, name='create_job'),
    path('jobs/', views.job_list, name='job_list'),
    path('jobs/<str:job_id>/', views.job_detail, name='job_detail'),
    path('jobs/<str:job_id>/edit/', views.job_edit, name='job_edit'),
    path('jobs/<str:job_id>/delete/', views.job_delete, name='job_delete'),
    path('delete-job/<str:job_id>/', views.job_delete, name='job_delete_ajax'),  # For AJAX
    path('jobs/<str:job_id>/delete-confirm/', views.job_delete_confirm, name='job_delete_confirm'),

]
