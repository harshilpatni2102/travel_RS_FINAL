"""
Gemini AI Integration Module
Provides AI-powered insights and explanations for travel recommendations
"""
import google.generativeai as genai
from typing import Dict, Optional

# Check if Gemini is available
try:
    import google.generativeai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiEnhancer:
    """
    AI-powered recommendation enhancer using Google Gemini
    Generates personalized travel insights and explanations
    """
    
    def __init__(self, api_key: str):
        """Initialize Gemini API"""
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def explain_recommendation(
        self,
        city: str,
        country: str,
        user_query: str,
        score_breakdown: Dict[float, float],
        destination_data: Dict
    ) -> str:
        """
        Generate AI explanation for why this destination matches the query
        
        Args:
            city: Destination city name
            country: Country name
            user_query: Original user search query
            score_breakdown: Dictionary with similarity and rating scores
            destination_data: Complete destination information
            
        Returns:
            AI-generated explanation string
        """
        try:
            # Extract relevant activity scores
            activities = {
                'Culture': destination_data.get('culture', 0),
                'Adventure': destination_data.get('adventure', 0),
                'Nature': destination_data.get('nature', 0),
                'Beaches': destination_data.get('beaches', 0),
                'Nightlife': destination_data.get('nightlife', 0),
                'Cuisine': destination_data.get('cuisine', 0),
                'Wellness': destination_data.get('wellness', 0),
                'Urban': destination_data.get('urban', 0),
                'Seclusion': destination_data.get('seclusion', 0)
            }
            
            # Find top 3 activities
            top_activities = sorted(activities.items(), key=lambda x: x[1], reverse=True)[:3]
            top_activities_str = ", ".join([f"{act}: {score}/5" for act, score in top_activities])
            
            description = destination_data.get('short_description', '')
            
            # Create enhanced prompt for Gemini
            prompt = f"""You are an expert travel guide. Write a compelling 3-4 sentence explanation for why {city}, {country} is PERFECT for someone searching: "{user_query}"

Destination Details:
- Description: {description}
- Top Strengths: {top_activities_str}
- Budget: {destination_data.get('budget_level', 'Mid-range')}

Your response MUST include:
1. WHY it matches their search (be specific)
2. 2-3 FAMOUS attractions/things to do there
3. WHAT makes it special/unique
4. ONE insider tip or best time to visit

Write in an exciting, personal tone. Make them want to book a flight NOW! Maximum 120 words."""

            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            # Fallback with basic info
            top_acts = sorted(activities.items(), key=lambda x: x[1], reverse=True)[:2]
            acts_text = " and ".join([act.lower() for act, _ in top_acts])
            return f"✨ {city} is perfect for {acts_text} enthusiasts! With a {score_breakdown.get('similarity_score', 0):.0%} match to your search, this destination offers authentic experiences and memorable adventures."
    
    def get_travel_tips(self, city: str, country: str, query_context: str) -> str:
        """Generate travel tips for a destination"""
        try:
            prompt = f"""Provide 3 quick travel tips for visiting {city}, {country} in the context of: {query_context}.
            
Format as bullet points. Be specific and practical. Under 80 words total."""
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return "Travel tips unavailable"


def get_gemini_enhancer(api_key: Optional[str]) -> Optional[GeminiEnhancer]:
    """
    Factory function to create GeminiEnhancer instance
    
    Args:
        api_key: Gemini API key
        
    Returns:
        GeminiEnhancer instance or None if unavailable
    """
    if not GEMINI_AVAILABLE:
        return None
    
    if not api_key or api_key.strip() == "":
        return None
    
    try:
        return GeminiEnhancer(api_key)
    except Exception:
        return None
