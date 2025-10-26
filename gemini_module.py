"""
Gemini AI Integration Module for Enhanced Travel Recommendations
Provides intelligent destination insights and improved recommendation quality
"""

import google.generativeai as genai
import configparser
import os
import logging
from typing import List, Dict, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini availability flag
GEMINI_AVAILABLE = True
try:
    import google.generativeai as genai
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Gemini features will be disabled.")


class GeminiEnhancer:
    """
    Integrates Google Gemini AI to enhance travel recommendations with:
    - Enriched destination descriptions
    - Intelligent recommendation explanations
    - Contextual travel insights
    - Improved accuracy through AI reasoning
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini AI with API key from config or parameter
        
        Args:
            api_key: Optional API key (if not provided, reads from config.ini)
        """
        if api_key:
            self.api_key = api_key
        else:
            # Try to read from config.ini with proper encoding
            self.api_key = None
            try:
                config = configparser.ConfigParser()
                if os.path.exists('config.ini'):
                    # Read with UTF-8 encoding to handle special characters
                    config.read('config.ini', encoding='utf-8')
                    self.api_key = config.get('GEMINI', 'api_key', fallback=None)
            except Exception as e:
                logger.warning(f"⚠️ Could not read config.ini: {e}")
                self.api_key = None
        
        if self.api_key and self.api_key != "YOUR_API_KEY_HERE":
            try:
                genai.configure(api_key=self.api_key)
                # Use the latest stable Gemini 2.0 Flash model
                self.model = genai.GenerativeModel('models/gemini-2.0-flash-exp')
                self.enabled = True
                logger.info("✅ Gemini AI initialized successfully with gemini-2.0-flash-exp")
            except Exception as e:
                self.enabled = False
                logger.warning(f"⚠️ Gemini initialization failed: {e}")
        else:
            self.enabled = False
            logger.warning("⚠️ Gemini API key not found. AI enhancements disabled.")
    
    def generate_enhanced_description(self, city: str, country: str, region: str, 
                                     activities: Dict[str, float], budget: str) -> str:
        """
        Generate an enriched, engaging description for a destination using Gemini
        
        Args:
            city: City name
            country: Country name
            region: Geographic region
            activities: Dictionary of activity ratings
            budget: Budget level (Budget/Mid-Range/Luxury)
        
        Returns:
            Enhanced description text
        """
        if not self.enabled:
            return f"{city}, {country} - A beautiful destination in {region}."
        
        try:
            # Identify top activities
            top_activities = sorted(activities.items(), key=lambda x: x[1], reverse=True)[:3]
            activity_text = ", ".join([f"{act} ({score:.1f}/5)" for act, score in top_activities])
            
            prompt = f"""Create a compelling 2-3 sentence travel description for {city}, {country} ({region}).

Key characteristics:
- Budget level: {budget}
- Top activities: {activity_text}

Make it engaging, informative, and highlight what makes this destination unique. Focus on experiences travelers would enjoy."""

            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"❌ Gemini description generation failed: {e}")
            return f"{city}, {country} - A wonderful {budget.lower()} destination known for {', '.join([act[0] for act in top_activities[:2]])}."
    
    def explain_recommendation(self, city: str, country: str, user_query: str, 
                              score_breakdown: Dict[str, float] = None, 
                              destination_data: Dict = None) -> str:
        """
        Generate an intelligent explanation for why this destination was recommended.
        Creates a detailed paragraph (4-6 sentences) with what the city is famous for.
        
        Args:
            city: City name
            country: Country name
            user_query: User's original query
            score_breakdown: Dictionary of score components (nlp_score, content_score, etc.)
            destination_data: Full destination data dictionary
        
        Returns:
            Natural language explanation paragraph
        """
        if not self.enabled:
            return f"{city} has been recommended based on your preferences."
        
        # Extract top activities from destination data
        activity_cols = ['culture', 'adventure', 'nature', 'beaches', 
                        'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
        top_activities = []
        if destination_data:
            activities = [(act, destination_data.get(act, 0)) 
                         for act in activity_cols if destination_data.get(act, 0) >= 4.0]
            top_activities = [act for act, score in sorted(activities, key=lambda x: x[1], reverse=True)[:3]]
        
        if not top_activities:
            top_activities = ['culture', 'cuisine', 'urban']  # defaults
        
        # Get match score
        match_score = score_breakdown.get('final_score', 0) if score_breakdown else 0
        
        try:
            prompt = f"""Write a detailed, engaging travel recommendation paragraph (5-7 sentences) for {city}, {country}.

User's Travel Request: "{user_query}"

Destination Profile:
- Top Attractions/Activities: {', '.join(top_activities)}
- Match Score: {match_score:.2f}/1.00 (Excellent match!)

YOUR PARAGRAPH MUST INCLUDE (IN THIS ORDER):

1. FIRST SENTENCE: Start with "Bali is famous for..." or "{city} is renowned for..." - mention what makes this city/destination ICONIC and world-famous (landmarks, culture, natural beauty, unique characteristics)

2. Explain specifically why {city} perfectly matches the traveler's request: "{user_query}"

3. Describe the TOP experiences they'll enjoy based on these strengths: {', '.join(top_activities)} - be specific and vivid

4. Include a unique characteristic, hidden gem, or local secret that makes {city} special

5. End with a compelling, enthusiastic reason why they should visit NOW

Write in an enthusiastic, personal tone that inspires immediate travel. Use descriptive language and make it feel like a travel expert's insider recommendation."""

            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            # Verify it mentions what the city is famous for
            if not any(word in result.lower() for word in ['famous', 'renowned', 'known for', 'celebrated']):
                # Add it at the beginning if missing
                result = f"{city} is renowned for its exceptional {top_activities[0]} experiences. {result}"
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Gemini explanation failed: {e}")
            return f"{city} is famous for its {top_activities[0]} and offers excellent {', '.join(top_activities[:2])} experiences that match your interests."
    
    def refine_recommendations(self, recommendations: List[Dict], user_query: str, 
                              top_n: int = 5) -> List[Dict]:
        """
        Use Gemini to re-rank and refine recommendations for better accuracy
        
        Args:
            recommendations: List of recommendation dictionaries
            user_query: User's search query
            top_n: Number of top recommendations to refine
        
        Returns:
            Refined and enhanced recommendations
        """
        if not self.enabled or not recommendations:
            return recommendations
        
        try:
            # Take top candidates for refinement
            candidates = recommendations[:min(top_n * 2, len(recommendations))]
            
            # Build destination summaries
            dest_summaries = []
            for i, rec in enumerate(candidates[:10]):  # Limit for API efficiency
                activities = [k for k, v in sorted(
                    {k: rec.get(k, 0) for k in ['culture', 'adventure', 'nature', 'beaches', 
                                                'nightlife', 'cuisine', 'wellness', 'urban']}.items(),
                    key=lambda x: x[1], reverse=True
                )[:3]]
                dest_summaries.append(
                    f"{i+1}. {rec['city']}, {rec['country']} - {rec.get('budget_level', 'Mid-Range')}, "
                    f"Best for: {', '.join(activities)}"
                )
            
            prompt = f"""A traveler is searching for: "{user_query}"

Here are potential destinations (ranked by algorithm):
{chr(10).join(dest_summaries)}

Task: From these options, select the TOP {min(top_n, len(candidates))} destinations that BEST match the query.
Return ONLY the numbers (e.g., "1,3,5,2,7") in order of best to worst match.
Consider semantic meaning, activity alignment, and traveler intent."""

            response = self.model.generate_content(prompt)
            
            # Parse Gemini's ranking
            ranking_text = response.text.strip()
            rankings = []
            for char in ranking_text:
                if char.isdigit():
                    idx = int(char) - 1
                    if 0 <= idx < len(candidates):
                        rankings.append(idx)
            
            # Reorder based on Gemini's suggestions
            if rankings:
                reordered = [candidates[i] for i in rankings[:top_n]]
                # Add remaining destinations
                remaining = [rec for i, rec in enumerate(recommendations) 
                           if i >= len(candidates) or i not in rankings]
                return reordered + remaining
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Gemini refinement failed: {e}")
            return recommendations
    
    def generate_travel_insights(self, city: str, country: str, season: str = "any") -> Dict[str, str]:
        """
        Generate contextual travel insights and tips
        
        Args:
            city: City name
            country: Country name
            season: Season for travel (optional)
        
        Returns:
            Dictionary with insights (best_time, tips, must_see, cuisine)
        """
        if not self.enabled:
            return {
                "best_time": "Year-round",
                "tips": "Research local customs and weather before visiting.",
                "must_see": "Explore the main attractions.",
                "cuisine": "Try local specialties."
            }
        
        try:
            prompt = f"""Provide quick travel insights for {city}, {country}:
1. Best time to visit (1 sentence)
2. One practical travel tip (1 sentence)
3. Must-see attraction (1 sentence)
4. Signature dish to try (1 sentence)

Format as: BEST_TIME: ... | TIP: ... | MUST_SEE: ... | CUISINE: ..."""

            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Parse response
            insights = {}
            for line in text.split('|'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace('_', ' ')
                    insights[key] = value.strip()
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Gemini insights failed: {e}")
            return {
                "best_time": "Year-round",
                "tips": "Check travel advisories and local weather.",
                "must_see": "Explore popular landmarks.",
                "cuisine": "Sample authentic local cuisine."
            }
    
    def batch_enhance_descriptions(self, destinations: List[Dict], max_batch: int = 5) -> List[Dict]:
        """
        Enhance multiple destination descriptions in batch (with rate limiting)
        
        Args:
            destinations: List of destination dictionaries
            max_batch: Maximum number to process
        
        Returns:
            Destinations with enhanced descriptions
        """
        if not self.enabled:
            return destinations
        
        enhanced = []
        for i, dest in enumerate(destinations[:max_batch]):
            try:
                activities = {k: dest.get(k, 0) for k in 
                            ['culture', 'adventure', 'nature', 'beaches', 
                             'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']}
                
                enhanced_desc = self.generate_enhanced_description(
                    dest['city'], 
                    dest['country'], 
                    dest.get('region', ''),
                    activities,
                    dest.get('budget_level', 'Mid-Range')
                )
                
                dest['enhanced_description'] = enhanced_desc
                enhanced.append(dest)
                
                # Rate limiting (Gemini has 60 requests/min limit)
                if i < max_batch - 1:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"❌ Batch enhancement failed for {dest.get('city', 'unknown')}: {e}")
                enhanced.append(dest)
        
        # Add remaining destinations without enhancement
        enhanced.extend(destinations[max_batch:])
        return enhanced


# Singleton instance
_gemini_instance = None

def get_gemini_enhancer(api_key: Optional[str] = None) -> GeminiEnhancer:
    """
    Get or create singleton Gemini enhancer instance
    
    Args:
        api_key: Optional API key
    
    Returns:
        GeminiEnhancer instance
    """
    global _gemini_instance
    if _gemini_instance is None:
        _gemini_instance = GeminiEnhancer(api_key)
    return _gemini_instance


if __name__ == "__main__":
    # Test the module
    print("🧪 Testing Gemini Integration...")
    
    enhancer = GeminiEnhancer()
    
    if enhancer.enabled:
        # Test description generation
        desc = enhancer.generate_enhanced_description(
            "Paris", "France", "Western Europe",
            {"culture": 4.8, "cuisine": 4.9, "urban": 4.7, "nightlife": 4.2},
            "Mid-Range"
        )
        print(f"\n📝 Enhanced Description:\n{desc}")
        
        # Test recommendation explanation
        explanation = enhancer.explain_recommendation(
            "Tokyo", "Japan", "looking for amazing food and cultural experiences",
            {"total_score": 0.89, "nlp_score": 0.92, "content_score": 0.88},
            ["cuisine", "culture", "urban"]
        )
        print(f"\n💡 Recommendation Explanation:\n{explanation}")
    else:
        print("\n⚠️ Gemini not configured. Add API key to config.ini")
