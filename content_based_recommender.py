"""
Simple Content-Based Recommendation System
Focuses purely on activity matching without NLP embeddings
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import streamlit as st

from config import MIN_ACTIVITY_SCORE


class ContentBasedRecommender:
    """
    Simple content-based recommender using activity scores
    No NLP embeddings - just keyword matching and activity filtering
    """
    
    def __init__(self, df: pd.DataFrame, gemini_enhancer=None):
        self.df = df.copy()
        self.gemini_enhancer = gemini_enhancer
        
        # Activity columns
        self.activity_cols = [
            'culture', 'adventure', 'nature', 'beaches',
            'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion'
        ]
    
    def _extract_location(self, query: str) -> Optional[str]:
        """Extract country from query"""
        query_lower = query.lower()
        countries = self.df['country'].dropna().unique()
        
        for country in countries:
            if len(country) > 3 and country.lower() in query_lower:
                return country
        return None
    
    def _detect_activities(self, query: str) -> List[str]:
        """Detect activities from keywords"""
        query_lower = query.lower()
        
        activity_keywords = {
            'beaches': ['beach', 'beaches', 'sand', 'coast', 'seaside', 'ocean', 'sea', 'swimming', 'tropical', 'island'],
            'nature': ['mountain', 'mountains', 'nature', 'natural', 'wildlife', 'forest', 'jungle', 'trek', 'hiking', 'hills', 'peaks', 'scenic', 'landscape'],
            'culture': ['culture', 'cultural', 'heritage', 'historic', 'history', 'museum', 'temple', 'palace', 'monuments', 'traditional', 'ancient'],
            'adventure': ['adventure', 'trekking', 'hiking', 'climbing', 'rafting', 'sports', 'activities', 'extreme', 'thrilling'],
            'nightlife': ['nightlife', 'party', 'clubs', 'bars', 'entertainment', 'night', 'dancing'],
            'cuisine': ['food', 'cuisine', 'culinary', 'restaurant', 'eat', 'dining', 'dishes', 'street food', 'local food'],
            'wellness': ['wellness', 'spa', 'relax', 'relaxation', 'peaceful', 'calm', 'yoga', 'meditation', 'retreat'],
            'urban': ['city', 'urban', 'metropolitan', 'modern', 'shopping', 'cosmopolitan', 'downtown'],
            'seclusion': ['secluded', 'remote', 'isolated', 'quiet', 'peaceful', 'escape', 'private', 'off the beaten']
        }
        
        detected = []
        for activity, keywords in activity_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected.append(activity)
        
        return detected
    
    def _score_destinations(self, df: pd.DataFrame, activities: List[str], query: str) -> pd.DataFrame:
        """Score destinations based on activity match - ONLY show real matches"""
        scores = []
        
        for idx, row in df.iterrows():
            score = 0.0
            
            # Score based on detected activities
            if activities:
                matched_activities = 0
                total_activity_score = 0
                
                for activity in activities:
                    if activity in row and pd.notna(row[activity]):
                        activity_value = float(row[activity])
                        # MUST be >= 3.0 to count as a match
                        if activity_value >= 3.0:
                            matched_activities += 1
                            total_activity_score += activity_value / 5.0  # Normalize to 0-1
                
                # Only score if ALL activities matched
                if matched_activities == len(activities):
                    # Perfect match - all activities present
                    score = total_activity_score / len(activities)
                elif matched_activities >= len(activities) // 2 and len(activities) > 1:
                    # For multiple activities, allow matching at least half
                    score = (total_activity_score / len(activities)) * 0.8
                else:
                    # Not enough matches - score = 0 (will be filtered out)
                    score = 0.0
            else:
                # No activities detected
                score = 0.0
            
            # Small bonus for description match (max 0.1)
            if score > 0:
                description = str(row.get('short_description', '')).lower()
                query_words = query.lower().split()
                matching_words = sum(1 for word in query_words if len(word) > 3 and word in description)
                description_bonus = min(0.1, matching_words * 0.02)
                score += description_bonus
            
            scores.append(min(1.0, score))  # Cap at 1.0
        
        df['similarity_score'] = scores
        return df
    
    def recommend(self, query: str, top_n: int = 5, use_ai: bool = True) -> pd.DataFrame:
        """
        Main recommendation function
        
        Args:
            query: User's natural language query
            top_n: Number of recommendations
            use_ai: Whether to generate AI insights
            
        Returns:
            DataFrame with recommendations
        """
        if not query or not query.strip():
            st.warning("Please enter a search query")
            return pd.DataFrame()
        
        # Filter by location
        country = self._extract_location(query)
        filtered_df = self.df.copy()
        
        if country:
            filtered_df = filtered_df[filtered_df['country'].str.lower() == country.lower()].copy()
            if len(filtered_df) > 0:
                st.success(f"🌍 Showing destinations in **{country}**")
            else:
                st.error(f"❌ No destinations found in {country}")
                return pd.DataFrame()
        
        # Detect activities
        activities = self._detect_activities(query)
        if activities:
            st.info(f"🎯 Searching for: **{', '.join([a.title() for a in activities])}**")
        
        # Score destinations
        filtered_df = self._score_destinations(filtered_df, activities, query)
        
        # Filter based on whether activities were detected
        if activities:
            # Activity-based search - keep all matches (activity score already filters quality)
            filtered_df = filtered_df[filtered_df['similarity_score'] >= 0.1]  # Very low threshold - just remove zeros
        else:
            # General search - use higher threshold
            filtered_df = filtered_df[filtered_df['similarity_score'] >= 0.25]
        
        # Sort by score
        filtered_df = filtered_df.sort_values('similarity_score', ascending=False)
        
        # Get results
        total_matches = len(filtered_df)
        results = filtered_df.head(top_n)
        
        if len(results) == 0:
            st.warning("❌ No destinations found matching your criteria. Try different keywords or location.")
            return pd.DataFrame()
        
        # Show how many results found
        if total_matches > top_n:
            st.info(f"✅ Found {total_matches} matches! Showing top {len(results)} destinations.")
        elif total_matches == len(results):
            st.success(f"✅ Found all {len(results)} matching destination(s)!")
        else:
            st.info(f"✅ Found {len(results)} match(es)")
        
        # Generate AI insights if enabled
        if use_ai and self.gemini_enhancer:
            try:
                st.info("🤖 Generating AI insights...")
                ai_insights = []
                for idx, row in results.iterrows():
                    insight = self.gemini_enhancer.explain_recommendation(
                        city=row.get('city', 'Unknown'),
                        country=row.get('country', ''),
                        user_query=query,
                        score_breakdown={'similarity_score': row.get('similarity_score', 0)},
                        destination_data=row.to_dict()
                    )
                    ai_insights.append(insight)
                results['ai_insight'] = ai_insights
            except Exception as e:
                st.warning(f"⚠️ AI insights unavailable")
        
        return results


def create_content_based_recommender(df: pd.DataFrame, gemini_enhancer=None) -> ContentBasedRecommender:
    """Factory function"""
    return ContentBasedRecommender(df, gemini_enhancer)
