"""
Recommendation Engine for Travel Destination System
===================================================
This module implements various recommendation strategies:
- Content-Based Filtering using NLP semantic similarity
- Hybrid approaches combining multiple signals
- Popularity-based fallback recommendations

Academic Alignment:
- Demonstrates content-based recommendation algorithms
- Implements feature weighting and score normalization
- Showcases hybrid recommendation strategies
- Provides explainability through score decomposition
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


class TravelRecommender:
    """
    Main recommendation engine for travel destinations.
    Combines NLP similarity with content-based filtering.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the recommender with destination data.
        
        Args:
            df (pd.DataFrame): Dataset containing destination information
        """
        self.df = df.copy()
        self.activity_columns = [
            'culture', 'adventure', 'nature', 'beaches', 
            'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion'
        ]
        self._preprocess_data()
    
    def _preprocess_data(self):
        """
        Preprocess the dataset for recommendation.
        Handles missing values and normalizes features.
        """
        # Fill missing values
        for col in self.activity_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Ensure numeric types
        for col in self.activity_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Create an overall rating if not present
        if 'overall_rating' not in self.df.columns:
            self.df['overall_rating'] = self.df[self.activity_columns].mean(axis=1)
        
        # Normalize budget level to numeric
        if 'budget_level' in self.df.columns:
            budget_map = {'budget': 1, 'mid-range': 2, 'luxury': 3}
            self.df['budget_numeric'] = self.df['budget_level'].map(
                lambda x: budget_map.get(str(x).lower(), 2)
            )
    
    def apply_filters(self, 
                     continent: Optional[str] = None,
                     budget: Optional[str] = None,
                     min_rating: float = 0.0) -> pd.DataFrame:
        """
        Apply user-specified filters to the dataset.
        
        Args:
            continent (str): Filter by continent/region
            budget (str): Filter by budget level
            min_rating (float): Minimum overall rating threshold
            
        Returns:
            pd.DataFrame: Filtered dataset
        """
        filtered_df = self.df.copy()
        
        # Apply continent filter
        if continent and continent != "All":
            if 'region' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['region'].str.contains(continent, case=False, na=False)
                ]
        
        # Apply budget filter
        if budget and budget != "All":
            if 'budget_level' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['budget_level'].str.lower() == budget.lower()
                ]
        
        # Apply rating filter
        if 'overall_rating' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['overall_rating'] >= min_rating]
        
        return filtered_df
    
    def get_content_based_score(self, row: pd.Series, 
                               user_preferences: Dict[str, float]) -> float:
        """
        Calculate content-based score based on activity preferences.
        
        Args:
            row (pd.Series): Destination data row
            user_preferences (Dict[str, float]): User's activity preferences (0-5 scale)
            
        Returns:
            float: Content-based similarity score
        """
        if not user_preferences:
            return 0.0
        
        score = 0.0
        weight_sum = 0.0
        
        for activity, user_rating in user_preferences.items():
            if activity in row and pd.notna(row[activity]):
                # Calculate similarity (inverse of absolute difference)
                dest_rating = float(row[activity])
                similarity = 1 - abs(dest_rating - user_rating) / 5.0
                score += similarity * user_rating  # Weight by user's importance
                weight_sum += user_rating
        
        # Normalize by total weight
        if weight_sum > 0:
            score = score / weight_sum
        
        return score
    
    def compute_popularity_score(self, row: pd.Series) -> float:
        """
        Calculate popularity score based on overall ratings.
        
        Args:
            row (pd.Series): Destination data row
            
        Returns:
            float: Popularity score (0-1 range)
        """
        if 'overall_rating' in row and pd.notna(row['overall_rating']):
            # Normalize to 0-1 range (assuming ratings are 0-5)
            return float(row['overall_rating']) / 5.0
        return 0.0
    
    def get_hybrid_score(self,
                        nlp_score: float,
                        content_score: float,
                        popularity_score: float,
                        alpha: float = 0.5,
                        beta: float = 0.3,
                        gamma: float = 0.2) -> float:
        """
        Combine multiple recommendation signals into a hybrid score.
        
        Args:
            nlp_score (float): Semantic similarity score from NLP
            content_score (float): Content-based filtering score
            popularity_score (float): Popularity/rating score
            alpha (float): Weight for NLP score (default: 0.5)
            beta (float): Weight for content score (default: 0.3)
            gamma (float): Weight for popularity score (default: 0.2)
            
        Returns:
            float: Hybrid recommendation score
        """
        # Ensure weights sum to 1
        total = alpha + beta + gamma
        alpha, beta, gamma = alpha/total, beta/total, gamma/total
        
        hybrid_score = (alpha * nlp_score + 
                       beta * content_score + 
                       gamma * popularity_score)
        
        return hybrid_score
    
    def recommend(self,
                 nlp_similarity_scores: pd.Series,
                 user_query: str = "",
                 user_preferences: Dict[str, float] = None,
                 continent: Optional[str] = None,
                 budget: Optional[str] = None,
                 min_rating: float = 0.0,
                 top_n: int = 5,
                 alpha: float = 0.5,
                 beta: float = 0.3,
                 gamma: float = 0.2) -> pd.DataFrame:
        """
        Generate top-N recommendations using hybrid approach.
        
        Args:
            nlp_similarity_scores (pd.Series): NLP similarity scores for all destinations
            user_query (str): User's natural language query
            user_preferences (Dict[str, float]): User's activity preferences
            continent (str): Continent filter
            budget (str): Budget filter
            min_rating (float): Minimum rating filter
            top_n (int): Number of recommendations to return
            alpha (float): Weight for NLP similarity
            beta (float): Weight for content-based similarity
            gamma (float): Weight for popularity
            
        Returns:
            pd.DataFrame: Top-N recommended destinations with scores
        """
        # Apply filters
        filtered_df = self.apply_filters(continent, budget, min_rating)
        
        # Handle empty results
        if len(filtered_df) == 0:
            st.warning("No destinations match your filters. Showing popular destinations instead.")
            filtered_df = self.df.copy()
        
        # Align NLP scores with filtered data
        filtered_indices = filtered_df.index
        filtered_nlp_scores = nlp_similarity_scores.loc[filtered_indices]
        
        # Calculate content-based scores
        if user_preferences:
            content_scores = filtered_df.apply(
                lambda row: self.get_content_based_score(row, user_preferences),
                axis=1
            )
        else:
            content_scores = pd.Series(0.0, index=filtered_indices)
        
        # Calculate popularity scores
        popularity_scores = filtered_df.apply(self.compute_popularity_score, axis=1)
        
        # Normalize all scores to 0-1 range
        def normalize_series(s):
            if s.max() > s.min():
                return (s - s.min()) / (s.max() - s.min())
            return s
        
        filtered_nlp_scores = normalize_series(filtered_nlp_scores)
        content_scores = normalize_series(content_scores)
        popularity_scores = normalize_series(popularity_scores)
        
        # Compute hybrid scores
        hybrid_scores = filtered_df.index.map(
            lambda idx: self.get_hybrid_score(
                filtered_nlp_scores[idx],
                content_scores[idx],
                popularity_scores[idx],
                alpha, beta, gamma
            )
        )
        
        # Add scores to dataframe
        results = filtered_df.copy()
        results['nlp_score'] = filtered_nlp_scores
        results['content_score'] = content_scores
        results['popularity_score'] = popularity_scores
        results['final_score'] = hybrid_scores
        
        # Sort by final score and return top N
        results = results.sort_values('final_score', ascending=False)
        
        return results.head(top_n)
    
    def get_popular_destinations(self, 
                                top_n: int = 10,
                                continent: Optional[str] = None,
                                budget: Optional[str] = None) -> pd.DataFrame:
        """
        Get most popular destinations based on ratings.
        Useful as fallback when no query is provided.
        
        Args:
            top_n (int): Number of destinations to return
            continent (str): Optional continent filter
            budget (str): Optional budget filter
            
        Returns:
            pd.DataFrame: Top-N popular destinations
        """
        filtered_df = self.apply_filters(continent, budget, 0.0)
        
        if len(filtered_df) == 0:
            filtered_df = self.df.copy()
        
        # Sort by overall rating
        popular = filtered_df.sort_values('overall_rating', ascending=False)
        
        return popular.head(top_n)
    
    def get_destinations_by_activity(self, 
                                    activity: str,
                                    top_n: int = 10) -> pd.DataFrame:
        """
        Get top destinations for a specific activity.
        
        Args:
            activity (str): Activity name (e.g., 'beaches', 'culture')
            top_n (int): Number of destinations to return
            
        Returns:
            pd.DataFrame: Top-N destinations for that activity
        """
        if activity not in self.activity_columns:
            return pd.DataFrame()
        
        sorted_df = self.df.sort_values(activity, ascending=False)
        return sorted_df.head(top_n)
    
    def search_destination(self, city_name: str) -> Optional[pd.Series]:
        """
        Search for a specific destination by city name.
        
        Args:
            city_name (str): City name to search for
            
        Returns:
            pd.Series: Destination data if found, None otherwise
        """
        if 'city' not in self.df.columns:
            return None
        
        # Case-insensitive search
        matches = self.df[
            self.df['city'].str.contains(city_name, case=False, na=False)
        ]
        
        if len(matches) > 0:
            return matches.iloc[0]
        
        return None
    
    def get_similar_destinations(self, 
                                city_name: str,
                                top_n: int = 5) -> pd.DataFrame:
        """
        Find destinations similar to a given city based on activities.
        
        Args:
            city_name (str): Reference city name
            top_n (int): Number of similar destinations to return
            
        Returns:
            pd.DataFrame: Similar destinations
        """
        # Find the reference city
        reference = self.search_destination(city_name)
        
        if reference is None:
            return pd.DataFrame()
        
        # Extract activity profile
        activity_profile = {}
        for col in self.activity_columns:
            if col in reference and pd.notna(reference[col]):
                activity_profile[col] = float(reference[col])
        
        # Calculate similarity with all other destinations
        similarities = []
        for idx, row in self.df.iterrows():
            if row.get('city') == reference.get('city'):
                continue  # Skip the reference city itself
            
            sim_score = self.get_content_based_score(row, activity_profile)
            similarities.append((idx, sim_score))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_indices = [idx for idx, _ in similarities[:top_n]]
        similar_df = self.df.loc[top_indices].copy()
        similar_df['similarity_score'] = [score for _, score in similarities[:top_n]]
        
        return similar_df


def create_recommender(df: pd.DataFrame) -> TravelRecommender:
    """
    Factory function to create a recommender instance.
    
    Args:
        df (pd.DataFrame): Destination dataset
        
    Returns:
        TravelRecommender: Initialized recommender system
    """
    return TravelRecommender(df)


if __name__ == "__main__":
    # Test the recommender module
    print("Testing Recommender Module...")
    
    # Create sample data
    sample_data = {
        'city': ['Paris', 'Bali', 'Tokyo', 'New York', 'Barcelona'],
        'country': ['France', 'Indonesia', 'Japan', 'USA', 'Spain'],
        'region': ['Europe', 'Asia', 'Asia', 'North America', 'Europe'],
        'culture': [5.0, 4.0, 4.5, 4.0, 4.5],
        'beaches': [2.0, 5.0, 2.0, 2.5, 4.0],
        'cuisine': [5.0, 4.5, 5.0, 4.5, 4.5],
        'nightlife': [4.5, 4.0, 4.5, 5.0, 4.5],
        'budget_level': ['luxury', 'mid-range', 'mid-range', 'luxury', 'mid-range']
    }
    
    df = pd.DataFrame(sample_data)
    recommender = create_recommender(df)
    
    # Test filtering
    filtered = recommender.apply_filters(continent='Europe')
    print(f"\nEuropean destinations: {len(filtered)}")
    
    # Test popular destinations
    popular = recommender.get_popular_destinations(top_n=3)
    print(f"\nTop 3 popular destinations: {popular['city'].tolist()}")
    
    print("\n✓ Recommender Module test completed successfully!")
