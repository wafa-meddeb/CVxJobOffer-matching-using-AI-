# matcher/management/commands/test_filters.py

from django.core.management.base import BaseCommand
from matcher.models import CV
from matcher.utils.resume_filters import resume_filter_manager

class Command(BaseCommand):
    help = 'Test resume filters functionality'

    def handle(self, *args, **options):
        """
        Test function to verify resume filters are working
        """
        self.stdout.write(self.style.SUCCESS("=== Testing Resume Filters ===\n"))
        
        # Get all resumes
        all_resumes = CV.objects.all()
        total_count = all_resumes.count()
        self.stdout.write(f"Total resumes in database: {total_count}")
        
        if total_count == 0:
            self.stdout.write(self.style.WARNING("No resumes found. Please upload some resumes first."))
            return
        
        # Test 1: Search filter
        self.stdout.write("\n--- Test 1: Search Filter ---")
        search_queries = ['python', 'engineer', 'java', 'manager', 'developer']
        
        search_results = []
        for query in search_queries:
            try:
                filtered = resume_filter_manager.filter_by_search(all_resumes, query)
                count = filtered.count()
                self.stdout.write(f"Search '{query}': {count} results")
                if count > 0:
                    first_name = filtered.first().name
                    self.stdout.write(f"  First result: {first_name}")
                    search_results.append(query)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Search '{query}' failed: {e}"))
        
        # Test 2: Experience filter
        self.stdout.write("\n--- Test 2: Experience Level Filter ---")
        experience_levels = ['0-1', '2-5', '6-10', '10+']
        
        exp_results = []
        for level in experience_levels:
            try:
                filtered = resume_filter_manager.filter_by_experience_level(all_resumes, level)
                count = filtered.count()
                self.stdout.write(f"Experience level '{level}': {count} results")
                
                # Debug first resume if any results
                if count > 0:
                    first_resume = filtered.first()
                    calculated_years = resume_filter_manager._calculate_total_experience(first_resume)
                    self.stdout.write(f"  First result: {first_resume.name} ({calculated_years} years calculated)")
                    exp_results.append(level)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Experience filter '{level}' failed: {e}"))
        
        # Test 3: Education filter
        self.stdout.write("\n--- Test 3: Education Level Filter ---")
        education_levels = ['high_school', 'bachelor', 'master', 'phd']
        
        edu_results = []
        for level in education_levels:
            try:
                filtered = resume_filter_manager.filter_by_education_level(all_resumes, level)
                count = filtered.count()
                self.stdout.write(f"Education level '{level}': {count} results")
                if count > 0:
                    first_name = filtered.first().name
                    self.stdout.write(f"  First result: {first_name}")
                    edu_results.append(level)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Education filter '{level}' failed: {e}"))
        
        # Test 4: Sample resume data inspection
        self.stdout.write("\n--- Test 4: Sample Resume Data Inspection ---")
        sample_resumes = all_resumes[:2]  # First 2 resumes
        
        for i, resume in enumerate(sample_resumes, 1):
            self.stdout.write(f"\nResume {i}: {resume.name}")
            self.stdout.write(f"  Email: {resume.email}")
            self.stdout.write(f"  Skills type: {type(resume.skills)}")
            self.stdout.write(f"  Skills: {resume.skills}")
            self.stdout.write(f"  Experience type: {type(resume.experience)}")
            
            # Show experience preview
            if resume.experience:
                exp_str = str(resume.experience)
                if len(exp_str) > 200:
                    exp_str = exp_str[:200] + "..."
                self.stdout.write(f"  Experience: {exp_str}")
                
                # Calculate experience
                try:
                    years = resume_filter_manager._calculate_total_experience(resume)
                    self.stdout.write(f"  Calculated experience: {years} years")
                except Exception as e:
                    self.stdout.write(f"  Experience calculation error: {e}")
        
        # Test 5: Combined filters
        self.stdout.write("\n--- Test 5: Combined Filters ---")
        try:
            # Search + Experience
            combined1 = resume_filter_manager.filter_by_search(all_resumes, 'engineer')
            combined1 = resume_filter_manager.filter_by_experience_level(combined1, '2-5')
            self.stdout.write(f"'engineer' + '2-5 years': {combined1.count()} results")
            
            # Search + Education  
            combined2 = resume_filter_manager.filter_by_search(all_resumes, 'developer')
            combined2 = resume_filter_manager.filter_by_education_level(combined2, 'bachelor')
            self.stdout.write(f"'developer' + 'bachelor': {combined2.count()} results")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Combined filter error: {e}"))
        
        # Summary
        self.stdout.write(self.style.SUCCESS("\n=== Summary ==="))
        self.stdout.write(f"Total resumes: {total_count}")
        
        working_filters = []
        if search_results:
            working_filters.append(f"Search ({len(search_results)} queries working)")
        if exp_results:
            working_filters.append(f"Experience ({len(exp_results)} levels working)")
        if edu_results:
            working_filters.append(f"Education ({len(edu_results)} levels working)")
            
        if working_filters:
            self.stdout.write(self.style.SUCCESS(f"Working filters: {', '.join(working_filters)}"))
        else:
            self.stdout.write(self.style.WARNING("No filters returned results. Check your data structure."))
            
        self.stdout.write(self.style.SUCCESS("\n=== Filter Testing Complete ==="))