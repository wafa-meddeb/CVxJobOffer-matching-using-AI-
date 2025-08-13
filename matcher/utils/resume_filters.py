# utils/resume_filters.py

import json
import re
from datetime import datetime
from django.db.models import Q
from django.utils import timezone

class ResumeFilterManager:
    """
    Centralized class for managing resume filters with proper JSONField handling
    """
    
    def __init__(self):
        self.education_keywords = {
            'high_school': [
                'high school', 'secondary school', 'diploma', 'ged', 
                'secondary education', 'high school diploma', 'lycee', 'baccalaureat'
            ],
            'associate': [
                'associate', 'aa ', 'as ', 'aas', 'associate degree',
                'community college', '2-year degree', 'dut', 'bts'
            ],
            'bachelor': [
                'bachelor', 'ba ', 'bs ', 'bsc', 'b.a', 'b.s', 'bachelor\'s',
                'undergraduate', '4-year degree', 'beng', 'bcom', 'licence', 'ingenieur'
            ],
            'master': [
                'master', 'ma ', 'ms ', 'msc', 'm.a', 'm.s', 'mba', 'master\'s',
                'graduate', 'postgraduate', 'meng', 'med', 'mastère', 'dess'
            ],
            'phd': [
                'phd', 'ph.d', 'doctorate', 'doctoral', 'dphil', 
                'doctor of philosophy', 'postdoc', 'doctorat'
            ]
        }
    
    def filter_by_search(self, queryset, search_query):
        """
        Advanced search with better JSONField handling
        """
        if not search_query:
            return queryset
            
        search_terms = self._process_search_query(search_query)
        
        combined_query = Q()
        
        for term in search_terms:
            # Basic text fields
            term_query = (
                Q(name__icontains=term) |
                Q(email__icontains=term) |
                Q(phone__icontains=term) |
                Q(summary__icontains=term)
            )
            
            # JSONField searches - handle both list and string formats
            # Skills search
            term_query |= Q(skills__icontains=term)  # Works if skills is stored as string
            
            # For PostgreSQL with proper JSON queries (if available)
            try:
                # This works with PostgreSQL JSONField
                term_query |= Q(skills__contains=[term])
            except:
                pass  # Fallback to string search
            
            # Experience and Education JSON field searches
            term_query |= Q(experience__icontains=term)
            term_query |= Q(education__icontains=term)
            term_query |= Q(languages__icontains=term)
            term_query |= Q(certificates__icontains=term)
            
            combined_query &= term_query
        
        return queryset.filter(combined_query)
    
    def filter_by_skills(self, queryset, skills_list):
        """
        Filter by specific skills with JSONField handling
        """
        if not skills_list:
            return queryset
            
        skills_query = Q()
        for skill in skills_list:
            # Handle both string and list formats
            skills_query |= Q(skills__icontains=skill)
            
            # For proper JSON contains (PostgreSQL)
            try:
                skills_query |= Q(skills__contains=[skill])
            except:
                pass
        
        return queryset.filter(skills_query)
    
    def filter_by_experience_level(self, queryset, experience_range):
        """
        Filter by years of experience with improved JSONField parsing
        """
        if not experience_range:
            return queryset
            
        filtered_ids = []
        
        for resume in queryset:
            years_exp = self._calculate_total_experience(resume)
            
            if self._experience_matches_range(years_exp, experience_range):
                filtered_ids.append(resume.id)
        
        return queryset.filter(id__in=filtered_ids)
    
    def filter_by_education_level(self, queryset, education_level):
        """
        Filter by highest education level with JSONField handling
        """
        if not education_level or education_level not in self.education_keywords:
            return queryset
            
        keywords = self.education_keywords[education_level]
        education_query = Q()
        
        for keyword in keywords:
            # Search in education JSONField
            education_query |= Q(education__icontains=keyword)
            # Also search in summary for education mentions
            education_query |= Q(summary__icontains=keyword)
        
        return queryset.filter(education_query)
    
    def filter_by_location(self, queryset, location):
        """
        Filter by location/address (extend as needed)
        """
        if not location:
            return queryset
            
        return queryset.filter(
            Q(summary__icontains=location) |
            Q(experience__icontains=location)  # Location might be in experience
        )
    
    def _process_search_query(self, query):
        """
        Process search query to handle quotes, special characters, etc.
        """
        # Handle quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        remaining_query = re.sub(r'"[^"]*"', '', query)
        
        # Split remaining query into terms
        individual_terms = remaining_query.split()
        
        # Combine quoted phrases and individual terms
        all_terms = quoted_phrases + individual_terms
        
        # Clean and filter terms
        processed_terms = []
        for term in all_terms:
            term = term.strip()
            if len(term) >= 2:  # Ignore very short terms
                processed_terms.append(term)
        
        return processed_terms
    
    def _calculate_total_experience(self, resume):
        """
        Calculate total years of experience with improved JSONField handling
        """
        if not resume.experience:
            return 0
            
        try:
            experience_data = resume.experience
            
            # Handle different data formats
            if isinstance(experience_data, str):
                try:
                    experience_data = json.loads(experience_data)
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat as plain text
                    return self._extract_years_from_text(experience_data)
            
            if isinstance(experience_data, list):
                total_years = 0
                for exp in experience_data:
                    years = self._extract_experience_years(exp)
                    total_years += years
                return total_years
            
            elif isinstance(experience_data, dict):
                # Single experience item
                return self._extract_experience_years(experience_data)
            
            return 0
            
        except Exception as e:
            print(f"Error calculating experience for resume {resume.cv_id}: {e}")
            return 0
    
    def _extract_experience_years(self, experience_item):
        """
        Extract years from experience item with multiple strategies
        """
        if not isinstance(experience_item, dict):
            if isinstance(experience_item, str):
                return self._extract_years_from_text(experience_item)
            return 0
        
        # Strategy 1: Look for explicit years/duration field
        years_text = (
            experience_item.get('years', '') or 
            experience_item.get('duration', '') or
            experience_item.get('period', '') or
            experience_item.get('time', '')
        )
        
        if years_text:
            years = self._extract_years_from_text(str(years_text))
            if years > 0:
                return years
        
        # Strategy 2: Calculate from start/end dates
        start_date = (
            experience_item.get('start_date') or 
            experience_item.get('from') or
            experience_item.get('start') or
            experience_item.get('start_year')
        )
        end_date = (
            experience_item.get('end_date') or 
            experience_item.get('to') or
            experience_item.get('end') or
            experience_item.get('end_year', 'present')
        )
        
        if start_date:
            calculated_years = self._calculate_years_between_dates(start_date, end_date)
            if calculated_years > 0:
                return calculated_years
        
        # Strategy 3: Look in job title or description for years
        title = experience_item.get('title', '') or experience_item.get('position', '')
        description = experience_item.get('description', '') or experience_item.get('details', '')
        company = experience_item.get('company', '') or experience_item.get('employer', '')
        
        combined_text = f"{title} {description} {company}"
        extracted_years = self._extract_years_from_text(combined_text)
        
        if extracted_years > 0:
            return extracted_years
        
        # Strategy 4: Default assumption for entries with no clear duration
        # If we have a position but no clear duration, assume 1 year minimum
        if title or description or company:
            return 1
        
        return 0
    
    def _extract_years_from_text(self, text):
        """
        Parse years from text using comprehensive regex patterns
        """
        if not text:
            return 0
            
        text = str(text).lower()
        
        # Pattern 1: "X years" or "X year"
        years_match = re.search(r'(\d+)\s*(?:years?|ans?|années?)', text)
        if years_match:
            return int(years_match.group(1))
        
        # Pattern 2: "X-Y years" (take average)
        range_match = re.search(r'(\d+)\s*[-–—]\s*(\d+)\s*(?:years?|ans?|années?)', text)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            return (start + end) // 2
        
        # Pattern 3: "X+ years" or "over X years"
        plus_match = re.search(r'(\d+)\+\s*(?:years?|ans?|années?)', text)
        if plus_match:
            return int(plus_match.group(1))
        
        over_match = re.search(r'(?:over|more than|plus de)\s*(\d+)\s*(?:years?|ans?|années?)', text)
        if over_match:
            return int(over_match.group(1))
        
        # Pattern 4: Date ranges like "2020-2023" or "2020 to 2023"
        date_range_match = re.search(r'(\d{4})\s*[-–—to à]\s*(\d{4})', text)
        if date_range_match:
            start_year, end_year = int(date_range_match.group(1)), int(date_range_match.group(2))
            return max(0, end_year - start_year)
        
        # Pattern 5: "Since YYYY" or "Depuis YYYY"
        since_match = re.search(r'(?:since|depuis)\s*(\d{4})', text)
        if since_match:
            start_year = int(since_match.group(1))
            current_year = datetime.now().year
            return max(0, current_year - start_year)
        
        # Pattern 6: Just numbers (assume years if reasonable)
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            for num_str in numbers:
                num = int(num_str)
                if 1 <= num <= 50:  # Reasonable range for years of experience
                    return num
        
        return 0
    
    def _calculate_years_between_dates(self, start_date, end_date):
        """
        Calculate years between dates with flexible parsing
        """
        try:
            start = self._parse_date(start_date)
            if not start:
                return 0
            
            if str(end_date).lower() in ['present', 'current', 'now', 'actuel', 'maintenant', '']:
                end = datetime.now()
            else:
                end = self._parse_date(end_date)
                if not end:
                    end = datetime.now()
            
            years = (end - start).days / 365.25
            return max(0, round(years, 1))
            
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error calculating years between dates {start_date} and {end_date}: {e}")
            return 0
    
    def _parse_date(self, date_str):
        """
        Flexible date parsing with multiple formats
        """
        if not date_str:
            return None
            
        date_str = str(date_str).strip()
        
        # Common date formats
        formats = [
            '%Y-%m-%d',      # 2023-01-15
            '%Y/%m/%d',      # 2023/01/15
            '%d/%m/%Y',      # 15/01/2023
            '%m/%d/%Y',      # 01/15/2023
            '%d-%m-%Y',      # 15-01-2023
            '%m-%d-%Y',      # 01-15-2023
            '%Y-%m',         # 2023-01
            '%Y/%m',         # 2023/01
            '%m/%Y',         # 01/2023
            '%m-%Y',         # 01-2023
            '%Y',            # 2023
            '%B %Y',         # January 2023
            '%b %Y',         # Jan 2023
            '%B %d, %Y',     # January 15, 2023
            '%b %d, %Y',     # Jan 15, 2023
            '%d %B %Y',      # 15 January 2023
            '%d %b %Y',      # 15 Jan 2023
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try to extract just the year if other formats fail
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            try:
                year = int(year_match.group(0))
                return datetime(year, 1, 1)
            except ValueError:
                pass
        
        return None
    
    def _experience_matches_range(self, years, range_str):
        """
        Check if years of experience matches the range
        """
        if range_str == '0-1':
            return 0 <= years <= 1
        elif range_str == '2-5':
            return 2 <= years <= 5
        elif range_str == '6-10':
            return 6 <= years <= 10
        elif range_str == '10+':
            return years > 10
        
        return False
    
    def get_filter_statistics(self, queryset):
        """
        Get statistics for filter options
        """
        stats = {
            'total_count': queryset.count(),
            'experience_levels': {'0-1': 0, '2-5': 0, '6-10': 0, '10+': 0},
            'education_levels': {},
            'upload_periods': {},
        }
        
        # Calculate experience level distribution
        for resume in queryset:
            years = self._calculate_total_experience(resume)
            
            if 0 <= years <= 1:
                level = '0-1'
            elif 2 <= years <= 5:
                level = '2-5'
            elif 6 <= years <= 10:
                level = '6-10'
            else:
                level = '10+'
            
            stats['experience_levels'][level] += 1
        
        # Calculate education level distribution
        for education_level, keywords in self.education_keywords.items():
            count = 0
            for resume in queryset:
                if resume.education:
                    education_text = str(resume.education).lower()
                    if any(keyword in education_text for keyword in keywords):
                        count += 1
            
            if count > 0:
                stats['education_levels'][education_level] = count
        
        return stats
    
    def debug_resume_experience(self, resume):
        """
        Debug function to understand how experience is being parsed
        """
        print(f"\nDebugging resume: {resume.name} (ID: {resume.cv_id})")
        print(f"Experience data type: {type(resume.experience)}")
        print(f"Experience data: {resume.experience}")
        
        total_years = self._calculate_total_experience(resume)
        print(f"Calculated total years: {total_years}")
        
        if resume.experience:
            try:
                exp_data = resume.experience
                if isinstance(exp_data, str):
                    exp_data = json.loads(exp_data)
                
                if isinstance(exp_data, list):
                    for i, exp in enumerate(exp_data):
                        years = self._extract_experience_years(exp)
                        print(f"  Experience {i+1}: {exp} -> {years} years")
                        
            except Exception as e:
                print(f"Error parsing experience: {e}")


# Singleton instance for easy import
resume_filter_manager = ResumeFilterManager()