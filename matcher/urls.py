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

    path('upload-resumes/', views.upload_resumes, name='upload_resumes'),
    path('manage-resumes/', views.manage_resumes_view, name='manage_resumes'),
    path('delete-resume/<uuid:cv_id>/', views.delete_resume, name='delete_resume'),
    path('resumes/<uuid:cv_id>/', views.resume_detail, name='resume_detail'),
    path('resumes/<uuid:cv_id>/delete/', views.resume_delete_confirm, name='resume_delete_confirm'),
    path('api/resume-filter/', views.resume_filter_api, name='resume_filter_api'),
    path('api/search-suggestions/', views.search_suggestions_api, name='search_suggestions_api'),
    path('test-filters/', views.test_filters_view, name='test_filters'),

    # matching patterns
    path('api/match-resumes/', views.match_resumes_api, name='match_resumes_api'),
    path('api/jobs/', views.get_jobs_api, name='get_jobs_api'),
    path('api/resumes/', views.get_resumes_api, name='get_resumes_api'),
    path('api/match-history/', views.get_match_history_api, name='get_match_history_api'),
    path('api/analytics/', views.matching_analytics_api, name='matching_analytics_api'),
    
    
    #analytics patterns
    path('api/analytics/', views.analytics_api, name='analytics_api'),
    
    # Optional: Additional analytics endpoints
    path('api/analytics/period/<str:period>/', views.analytics_api, name='analytics_api_period'),
    # You might also want these additional endpoints for specific data:
    path('api/analytics/matches/top/', views.get_top_matches_api, name='top_matches_api'),
    path('api/analytics/activity/recent/', views.get_recent_activity_api, name='recent_activity_api'),
    path('api/analytics/companies/', views.get_top_companies_api, name='companies_api'),
    path('api/analytics/skills/', views.get_top_skills_api, name='skills_api'),
    path('api/analytics/experience-levels/', views.experience_levels_api, name='experience_levels_api'),
    
    
    
    # Embedding monitoring endpoints
    path('api/embedding/status/', views.embedding_status_view, name='embedding_status'),
    path('api/embedding/diagnostics/', views.embedding_diagnostics_view, name='embedding_diagnostics'),
    path('api/embedding/initialize/', views.embedding_initialize_view, name='embedding_initialize'),
    path('api/embedding/report/', views.embedding_status_report_view, name='embedding_report'),
    path('embedding/dashboard/', views.embedding_dashboard_view, name='embedding_dashboard'),

]
