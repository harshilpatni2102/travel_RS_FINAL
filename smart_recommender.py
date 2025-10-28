"""
Smart Recommendation Engine with AI Validation
Combines NLP semantic similarity with content-based filtering and AI verification
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional
import streamlit as st

from config import MIN_ACTIVITY_SCORE, SIMILARITY_THRESHOLD
from embedding_manager import get_embedding_manager


class SmartRecommender:
    """
    Hybrid recommendation system combining:
    1. NLP-based semantic similarity (using BERT embeddings)
    2. Content-based filtering (activity preferences)
    3. AI validation (Gemini API for accuracy)
    """
    
    def __init__(self, df: pd.DataFrame, gemini_enhancer=None):
        self.df = df.copy()
        self.embedding_manager = get_embedding_manager()
        self.gemini_enhancer = gemini_enhancer
        
        # Activity columns for content-based filtering
        self.activity_cols = [
            'culture', 'adventure', 'nature', 'beaches',
            'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion'
        ]
    
    def _extract_location_from_query(self, query: str) -> Optional[str]:
        """Extract country name from query"""
        query_lower = query.lower()
        
        # Get unique countries from dataset
        countries = self.df['country'].dropna().unique()
        
        for country in countries:
            if len(country) > 3 and country.lower() in query_lower:
                return country
        
        return None
    
    def _detect_activities(self, query: str) -> List[str]:
        """Detect which activities user is interested in"""
        query_lower = query.lower()
        
        activity_keywords = {
            'beaches': ['beach', 'beaches', 'sand', 'coast', 'seaside', 'ocean', 'sea', 'swimming'],
            'nature': ['mountain', 'mountains', 'nature', 'natural', 'wildlife', 'forest', 'jungle', 'trek', 'hiking', 'hills', 'peaks'],
            'culture': ['culture', 'cultural', 'heritage', 'historic', 'history', 'museum', 'temple', 'palace', 'monuments'],
            'adventure': ['adventure', 'trekking', 'hiking', 'climbing', 'rafting', 'sports', 'activities'],
            'nightlife': ['nightlife', 'party', 'clubs', 'bars', 'entertainment', 'night'],
            'cuisine': ['food', 'cuisine', 'culinary', 'restaurant', 'eat', 'dining', 'dishes'],
            'wellness': ['wellness', 'spa', 'relax', 'relaxation', 'peaceful', 'calm', 'yoga'],
            'urban': ['city', 'urban', 'metropolitan', 'modern', 'shopping'],
            'seclusion': ['secluded', 'remote', 'isolated', 'quiet', 'peaceful', 'escape', 'private']
        }
        
        detected = []
        for activity, keywords in activity_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected.append(activity)
        
        return detected
    
    def _filter_by_location(self, df: pd.DataFrame, country: str) -> pd.DataFrame:
        """Filter destinations by country"""
        filtered = df[df['country'].str.lower() == country.lower()].copy()
        
        if len(filtered) > 0:
            st.success(f"🌍 Showing destinations in **{country}**")
        else:
            st.error(f"❌ No destinations found in {country}")
        
        return filtered
    
    def _filter_by_activities(self, df: pd.DataFrame, activities: List[str]) -> pd.DataFrame:
        """Filter destinations that match requested activities"""
        if not activities:
            return df
        
        st.info(f"🎯 Filtering for: **{', '.join([a.title() for a in activities])}**")
        
        # Filter destinations with high scores in requested activities
        valid_indices = []
        
        for idx, row in df.iterrows():
            match_count = 0
            for activity in activities:
                if activity in row and pd.notna(row[activity]):
                    activity_value = float(row[activity])
                    # Inclusive threshold: >= 3.0 (60% - shows more beach destinations)
                    if activity_value >= 3.0:
                        match_count += 1
            
            # Single activity: must match
            # Multiple activities: must match at least half
            required_matches = 1 if len(activities) == 1 else max(1, len(activities) // 2)
            
            if match_count >= required_matches:
                valid_indices.append(idx)
        
        filtered = df.loc[valid_indices].copy() if valid_indices else pd.DataFrame()
        
        if len(valid_indices) > 0:
            st.success(f"✅ Found {len(valid_indices)} destinations with {'/'.join(activities)} ratings (3.0+)")
        else:
            st.warning(f"❌ No destinations found with good {'/'.join(activities)} scores. Try different criteria.")
        
        return filtered
    
    def _compute_semantic_similarity(self, query: str, df: pd.DataFrame) -> pd.Series:
        """Compute semantic similarity using pre-computed BERT embeddings"""
        # Get query embedding
        query_embedding = self.embedding_manager.encode_query(query).reshape(1, -1)
        
        # Get all destination embeddings
        dest_embeddings = self.embedding_manager.get_all_embeddings(df)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, dest_embeddings)[0]
        
        return pd.Series(similarities, index=df.index)
    
    def _generate_ai_insights(self, results: pd.DataFrame, query: str) -> pd.DataFrame:
        """Generate AI explanations for recommendations"""
        if not self.gemini_enhancer or len(results) == 0:
            return results
        
        try:
            st.info("🤖 Generating AI insights...")
            
            ai_insights = []
            for idx, row in results.iterrows():
                insight = self.gemini_enhancer.explain_recommendation(
                    city=row.get('city', 'Unknown'),
                    country=row.get('country', ''),
                    user_query=query,
                    score_breakdown={
                        'similarity_score': row.get('similarity_score', 0),
                        'overall_rating': row.get('overall_rating', 0)
                    },
                    destination_data=row.to_dict()
                )
                ai_insights.append(insight)
            
            results['ai_insight'] = ai_insights
            
        except Exception as e:
            st.warning(f"⚠️ AI insights unavailable: {str(e)}")
        
        return results
    
    def recommend(self, query: str, top_n: int = 5, use_ai: bool = True) -> pd.DataFrame:
        """
        Main recommendation function
        
        Args:
            query: User's natural language query
            top_n: Number of recommendations to return
            use_ai: Whether to generate AI insights
            
        Returns:
            DataFrame with recommended destinations
        """
        if not query or not query.strip():
            st.warning("Please enter a search query")
            return pd.DataFrame()
        
        # Step 1: Extract location filter
        country = self._extract_location_from_query(query)
        filtered_df = self.df.copy()
        
        if country:
            filtered_df = self._filter_by_location(filtered_df, country)
            if len(filtered_df) == 0:
                return pd.DataFrame()
        
        # Step 2: Detect and filter by activities
        activities = self._detect_activities(query)
        if activities:
            filtered_df = self._filter_by_activities(filtered_df, activities)
            if len(filtered_df) == 0:
                return pd.DataFrame()
        
        # Step 3: Compute semantic similarity
        similarities = self._compute_semantic_similarity(query, filtered_df)
        
        # Add similarity scores to dataframe
        filtered_df['similarity_score'] = similarities
        
        # Step 4: Filter and sort
        # If activities detected, don't filter by semantic similarity (activity match is enough!)
        if activities:
            # Activity-based search - sort by activity relevance, not semantic similarity
            filtered_df = filtered_df.sort_values('similarity_score', ascending=False)
        else:
            # General search - use semantic similarity threshold
            filtered_df = filtered_df[filtered_df['similarity_score'] >= 0.25]
            filtered_df = filtered_df.sort_values('similarity_score', ascending=False)
        
        # Step 5: Get results - show all high-quality matches up to top_n
        total_matches = len(filtered_df)
        results = filtered_df.head(top_n)
        
        if len(results) == 0:
            st.warning("❌ No destinations found matching your criteria. Try different keywords or location.")
            return pd.DataFrame()
        
        # Show result count feedback
        if total_matches > top_n:
            st.info(f"✅ Found {total_matches} matches! Showing top {len(results)} destinations. Increase 'Results' to see more.")
        elif total_matches == len(results):
            st.success(f"✅ Found all {len(results)} matching destination(s) for your query!")
        else:
            st.info(f"✅ Found {len(results)} high-quality match(es) (showing only highly relevant destinations)")
        
        # Step 6: Generate AI insights
        if use_ai and self.gemini_enhancer:
            results = self._generate_ai_insights(results, query)
        
        return results
    
    def get_popular_destinations(self, top_n: int = 10, continent: str = None) -> pd.DataFrame:
        """Get most popular destinations"""
        df = self.df.copy()
        
        if continent and continent != "All":
            df = df[df['region'].str.contains(continent, case=False, na=False)]
        
        return df.sort_values('overall_rating', ascending=False).head(top_n)


def create_recommender(df: pd.DataFrame, gemini_enhancer=None) -> SmartRecommender:
    """Factory function to create recommender"""
    return SmartRecommender(df, gemini_enhancer)
