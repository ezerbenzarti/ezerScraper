from typing import Dict, List, Optional, Set
import re
from langdetect import detect
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FieldMapper:
    def __init__(self):
        # Define field mappings for different languages
        self.field_mappings = {
            'en': {
                'name': {'name', 'full name', 'contact name', 'person', 'individual'},
                'phone': {'phone', 'telephone', 'tel', 'mobile', 'cell', 'contact number', 'number'},
                'email': {'email', 'e-mail', 'mail', 'contact email', 'electronic mail'},
                'address': {'address', 'location', 'street', 'building', 'office', 'place'},
                'domain': {'industry', 'field', 'sector', 'business area', 'specialty', 'focus', 'domain of activity'},
                'website': {'website', 'site', 'web', 'url', 'company website', 'web address'},
                'poste': {'position', 'job', 'role', 'title', 'function', 'poste', 'occupation'}
            },
            'fr': {
                'name': {'nom', 'nom complet', 'contact', 'personne', 'individu'},
                'phone': {'téléphone', 'tel', 'portable', 'mobile', 'numéro', 'contact'},
                'email': {'email', 'courriel', 'mail', 'adresse email', 'adresse électronique'},
                'address': {'adresse', 'lieu', 'rue', 'bâtiment', 'bureau', 'emplacement'},
                'domain': {'industrie', 'secteur', 'spécialité', 'domaine d\'activité', 'secteur d\'activité', 'métier'},
                'website': {'site web', 'site', 'web', 'url', 'site de l\'entreprise', 'adresse web'},
                'poste': {'poste', 'emploi', 'fonction', 'titre', 'rôle', 'occupation'}
            }
        }

        # Define context patterns for better understanding
        self.context_patterns = {
            'en': {
                'contact_info': r'(?:contact|reach|get in touch|connect|reach out)',
                'location_info': r'(?:location|where|address|find|locate)',
                'personal_info': r'(?:who|person|individual|name)',
                'communication': r'(?:phone|call|email|message|contact)',
                'business': r'(?:company|business|organization|firm)',
                'industry': r'(?:industry|field|sector|domain|specialty|focus|business area)'
            },
            'fr': {
                'contact_info': r'(?:contact|joindre|contacter|rejoindre|connecter)',
                'location_info': r'(?:emplacement|où|adresse|trouver|localiser)',
                'personal_info': r'(?:qui|personne|individu|nom)',
                'communication': r'(?:téléphone|appeler|email|message|contact)',
                'business': r'(?:entreprise|société|organisation|firme)',
                'industry': r'(?:industrie|domaine|secteur|spécialité|domaine d\'activité|métier)'
            }
        }

    def detect_language(self, text: str) -> str:
        """Detect the language of the given text."""
        try:
            return detect(text)
        except:
            return 'en'  # Default to English if detection fails

    def get_field_mappings(self, language: str) -> Dict[str, Set[str]]:
        """Get field mappings for the specified language."""
        return self.field_mappings.get(language, self.field_mappings['en'])

    def get_context_patterns(self, language: str) -> Dict[str, str]:
        """Get context patterns for the specified language."""
        return self.context_patterns.get(language, self.context_patterns['en'])

    def map_field(self, text: str, language: Optional[str] = None) -> Optional[str]:
        """
        Map the given text to a field based on language and context.
        Returns the field name if found, None otherwise.
        """
        if language is None:
            language = self.detect_language(text)
        
        text = text.lower().strip()
        mappings = self.get_field_mappings(language)
        
        for field, synonyms in mappings.items():
            if text in synonyms:
                return field
            
            # Check for partial matches
            for synonym in synonyms:
                if synonym in text or text in synonym:
                    return field
        
        return None

    def understand_context(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Understand the context of the given text and return relevant fields.
        """
        if language is None:
            language = self.detect_language(text)
        
        text = text.lower().strip()
        patterns = self.get_context_patterns(language)
        relevant_fields = set()
        
        # Check for context patterns
        for context_type, pattern in patterns.items():
            if re.search(pattern, text):
                if context_type == 'contact_info':
                    relevant_fields.update(['phone', 'email'])
                elif context_type == 'location_info':
                    relevant_fields.update(['address'])
                elif context_type == 'personal_info':
                    relevant_fields.update(['name', 'poste'])
                elif context_type == 'communication':
                    relevant_fields.update(['phone', 'email'])
                elif context_type == 'business':
                    relevant_fields.update(['domain', 'name'])
                elif context_type == 'industry':
                    relevant_fields.update(['domain'])
        
        # Also try direct field mapping
        mapped_field = self.map_field(text, language)
        if mapped_field:
            relevant_fields.add(mapped_field)
        
        return list(relevant_fields)

    def enhance_question(self, question: str, language: Optional[str] = None) -> str:
        """
        Enhance the question with additional context based on understanding.
        """
        if language is None:
            language = self.detect_language(question)
        
        relevant_fields = self.understand_context(question, language)
        if not relevant_fields:
            return question
        
        # Add field-specific context to the question
        field_context = {
            'name': 'the full name of the person or organization',
            'phone': 'the phone number or contact number',
            'email': 'the email address or electronic mail',
            'address': 'the complete address or location',
            'domain': 'the industry or field',
            'website': 'the website or web address',
            'poste': 'the job position or role'
        }
        
        context_parts = [field_context[field] for field in relevant_fields if field in field_context]
        if context_parts:
            enhanced_question = f"{question} (looking for {', '.join(context_parts)})"
            return enhanced_question
        
        return question

# Create a global instance
field_mapper = FieldMapper() 