"""
NLP Module for Travel Destination Recommendation System
========================================================
This module handles all Natural Language Processing tasks including:
- Text preprocessing and cleaning
- Tokenization and lemmatization
- Semantic embeddings using BERT-based models
- Cosine similarity computation between user queries and destination descriptions

Academic Alignment:
- Demonstrates NLP preprocessing pipeline
- Implements transformer-based semantic embeddings
- Showcases similarity computation for text matching
"""

import re
import string
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


class NLPProcessor:
    """
    Handles all NLP operations for the recommendation system.
    Uses sentence-transformers for semantic embeddings.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the NLP processor with a pretrained sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence-transformers model to use.
                             Default: 'all-MiniLM-L6-v2' (fast and efficient)
        """
        self.model_name = model_name
        self.model = self._load_model()
        
        # Common stopwords (extended list)
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'want', 'like', 'need', 'looking'
        }
        
        # Synonym mappings for travel-related terms
        self.synonym_map = {
            'beach': ['coast', 'seaside', 'shore', 'coastal', 'sand'],
            'city': ['urban', 'metropolitan', 'downtown', 'metropolis'],
            'mountain': ['mountains', 'peaks', 'ranges', 'alpine', 'highlands'],
            'nature': ['natural', 'wilderness', 'outdoors', 'wildlife', 'scenic', 'countryside'],
            'adventure': ['adventurous', 'exciting', 'thrilling', 'hiking', 'trekking', 'climbing', 'rafting'],
            'culture': ['cultural', 'heritage', 'historic', 'historical', 'museums', 'art'],
            'food': ['cuisine', 'culinary', 'dining', 'restaurants', 'gastronomy'],
            'nightlife': ['party', 'entertainment', 'night', 'clubs', 'bars'],
            'relaxing': ['wellness', 'spa', 'peaceful', 'calm', 'quiet', 'serene', 'relaxation'],
            'luxury': ['upscale', 'premium', 'expensive', 'high-end', 'five-star'],
            'budget': ['cheap', 'affordable', 'economical', 'low-cost', 'inexpensive'],
            'tropical': ['warm', 'hot', 'sunny', 'humid'],
            'cold': ['cool', 'winter', 'snow', 'skiing', 'chilly'],
            'winter': ['cold', 'cool', 'pleasant', 'mild', 'december', 'january', 'february'],
            'summer': ['hot', 'warm', 'sunny', 'june', 'july', 'august'],
            'spring': ['mild', 'pleasant', 'march', 'april', 'may'],
            'autumn': ['fall', 'cool', 'september', 'october', 'november'],
        }
    
    @st.cache_resource
    def _load_model(_self):
        """
        Load the sentence transformer model with caching.
        Uses Streamlit's cache to avoid reloading on every interaction.
        
        Returns:
            SentenceTransformer: Loaded model ready for encoding
        """
        try:
            model = SentenceTransformer(_self.model_name)
            return model
        except Exception as e:
            st.error(f"Error loading NLP model: {e}")
            # Fallback to a smaller model
            return SentenceTransformer('paraphrase-MiniLM-L3-v2')
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits (but keep spaces)
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove common stopwords from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without stopwords
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def expand_synonyms(self, text: str) -> str:
        """
        Expand text with travel-related synonyms to improve matching.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with expanded synonyms
        """
        words = text.split()
        expanded_words = words.copy()
        
        for word in words:
            if word in self.synonym_map:
                expanded_words.extend(self.synonym_map[word])
        
        return ' '.join(expanded_words)
    
    def preprocess(self, text: str, remove_stops: bool = True) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): Raw input text
            remove_stops (bool): Whether to remove stopwords
            
        Returns:
            str: Fully preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove stopwords if requested
        if remove_stops:
            text = self.remove_stopwords(text)
        
        # Expand with synonyms for better matching
        text = self.expand_synonyms(text)
        
        return text
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour only
    def encode_texts(_self, texts: List[str], _cache_version: str = "v2.0") -> np.ndarray:
        """
        Encode texts into semantic embeddings using the transformer model.
        
        Args:
            texts (List[str]): List of texts to encode
            
        Returns:
            np.ndarray: Matrix of embeddings (n_texts x embedding_dim)
        """
        try:
            embeddings = _self.model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            st.error(f"Error encoding texts: {e}")
            return np.zeros((len(texts), 384))  # Return zero embeddings as fallback
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            doc_embeddings (np.ndarray): Document embedding matrix
            
        Returns:
            np.ndarray: Similarity scores for each document
        """
        # Reshape query if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        return similarities
    
    def create_destination_text(self, row: pd.Series) -> str:
        """
        Create a rich text representation of a destination for embedding.
        Combines multiple features into one comprehensive text.
        
        Args:
            row (pd.Series): DataFrame row with destination data
            
        Returns:
            str: Combined text representation
        """
        text_parts = []
        
        # Add city and country
        if pd.notna(row.get('city')):
            text_parts.append(str(row['city']))
        if pd.notna(row.get('country')):
            text_parts.append(str(row['country']))
        if pd.notna(row.get('region')):
            text_parts.append(str(row['region']))
        
        # Add description
        if pd.notna(row.get('short_description')):
            text_parts.append(str(row['short_description']))
        
        # Add seasonal information based on temperature data
        if 'avg_temp_monthly' in row and pd.notna(row['avg_temp_monthly']):
            try:
                import json
                temp_data = json.loads(row['avg_temp_monthly'].replace("'", '"'))
                
                # Analyze winter months (Dec, Jan, Feb)
                winter_temps = [temp_data.get(str(m), {}).get('avg', 0) for m in [12, 1, 2]]
                avg_winter = sum(winter_temps) / len(winter_temps) if winter_temps else 0
                
                # Analyze summer months (Jun, Jul, Aug)
                summer_temps = [temp_data.get(str(m), {}).get('avg', 0) for m in [6, 7, 8]]
                avg_summer = sum(summer_temps) / len(summer_temps) if summer_temps else 0
                
                # Add winter suitability
                if 15 <= avg_winter <= 28:
                    text_parts.extend(['perfect winter destination', 'ideal winter weather', 'pleasant winter', 'winter travel'] * 5)
                elif 10 <= avg_winter < 15:
                    text_parts.extend(['cool winter', 'mild winter', 'winter destination'] * 3)
                elif avg_winter < 10:
                    text_parts.extend(['cold winter', 'winter sports', 'snow'] * 2)
                
                # Add summer suitability
                if 20 <= avg_summer <= 30:
                    text_parts.extend(['perfect summer', 'ideal summer', 'summer destination'] * 3)
                elif avg_summer > 30:
                    text_parts.extend(['hot summer', 'tropical summer'] * 2)
                
            except:
                pass
        
        # Add activity features (high ratings indicate strong features)
        activity_cols = ['culture', 'adventure', 'nature', 'beaches', 
                        'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
        
        for col in activity_cols:
            if col in row and pd.notna(row[col]):
                # Add activity name multiple times based on rating (for emphasis)
                rating = float(row[col])
                if rating >= 4.0:
                    text_parts.extend([col] * 3)
                elif rating >= 3.0:
                    text_parts.extend([col] * 2)
                elif rating >= 2.0:
                    text_parts.append(col)
        
        # Add budget level
        if pd.notna(row.get('budget_level')):
            budget = str(row['budget_level']).lower()
            text_parts.append(budget)
            if budget == 'budget':
                text_parts.extend(['affordable', 'cheap', 'economical'])
            elif budget == 'luxury':
                text_parts.extend(['upscale', 'premium', 'expensive'])
        
        return ' '.join(text_parts)
    
    def get_nlp_similarity_scores(self, df: pd.DataFrame, 
                                  user_query: str) -> pd.Series:
        """
        Main function to compute NLP-based similarity scores between
        user query and all destinations with temperature-aware filtering.
        
        Args:
            df (pd.DataFrame): Dataset with destination information
            user_query (str): User's natural language query
            
        Returns:
            pd.Series: Similarity scores for each destination
        """
        # Preprocess user query
        processed_query = self.preprocess(user_query)
        
        # Handle empty query
        if not processed_query.strip():
            return pd.Series(np.zeros(len(df)), index=df.index)
        
        # Detect if query is about winter/specific season
        query_lower = user_query.lower()
        is_winter_query = any(term in query_lower for term in ['winter', 'december', 'january', 'february', 'cold season'])
        is_summer_query = any(term in query_lower for term in ['summer', 'june', 'july', 'august', 'hot'])
        
        # Create destination texts with enhanced seasonal info
        destination_texts = df.apply(self.create_destination_text, axis=1).tolist()
        
        # Encode query
        query_embedding = self.encode_texts([processed_query], _cache_version="v2.0")
        
        # Encode destinations
        doc_embeddings = self.encode_texts(destination_texts, _cache_version="v2.0")
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, doc_embeddings)
        
        # Apply temperature-based filtering for winter queries
        if is_winter_query and 'avg_temp_monthly' in df.columns:
            import json
            for idx, row in df.iterrows():
                if pd.notna(row.get('avg_temp_monthly')):
                    try:
                        temp_data = json.loads(row['avg_temp_monthly'].replace("'", '"'))
                        # Winter months: Dec (12), Jan (1), Feb (2)
                        winter_temps = [temp_data.get(str(m), {}).get('avg', 0) for m in [12, 1, 2]]
                        avg_winter = sum(winter_temps) / len(winter_temps) if winter_temps else 0
                        
                        # Boost score for ideal winter destinations (15-28°C)
                        if 15 <= avg_winter <= 28:
                            similarities[idx] *= 1.5  # 50% boost
                        # Slightly boost mild winter (10-15°C or 28-32°C)
                        elif (10 <= avg_winter < 15) or (28 < avg_winter <= 32):
                            similarities[idx] *= 1.2  # 20% boost
                        # Penalize too hot destinations (>35°C)
                        elif avg_winter > 35:
                            similarities[idx] *= 0.3  # 70% penalty
                        # Penalize very cold (<5°C) unless query mentions "cold" or "snow"
                        elif avg_winter < 5 and not any(term in query_lower for term in ['cold', 'snow', 'skiing', 'winter sports']):
                            similarities[idx] *= 0.4  # 60% penalty
                    except:
                        pass
        
        # Apply temperature-based filtering for summer queries
        elif is_summer_query and 'avg_temp_monthly' in df.columns:
            import json
            for idx, row in df.iterrows():
                if pd.notna(row.get('avg_temp_monthly')):
                    try:
                        temp_data = json.loads(row['avg_temp_monthly'].replace("'", '"'))
                        # Summer months: Jun (6), Jul (7), Aug (8)
                        summer_temps = [temp_data.get(str(m), {}).get('avg', 0) for m in [6, 7, 8]]
                        avg_summer = sum(summer_temps) / len(summer_temps) if summer_temps else 0
                        
                        # Boost score for good summer destinations (20-30°C)
                        if 20 <= avg_summer <= 30:
                            similarities[idx] *= 1.5
                        # Slightly boost hot summer (30-35°C)
                        elif 30 < avg_summer <= 35:
                            similarities[idx] *= 1.2
                        # Penalize too cold (<15°C)
                        elif avg_summer < 15:
                            similarities[idx] *= 0.3
                    except:
                        pass
        
        # Normalize back to 0-1 range
        max_sim = similarities.max()
        if max_sim > 0:
            similarities = similarities / max_sim
        
        return pd.Series(similarities, index=df.index)


# Factory function for easy module usage
def create_nlp_processor(model_name: str = 'all-MiniLM-L6-v2') -> NLPProcessor:
    """
    Factory function to create an NLP processor instance.
    
    Args:
        model_name (str): Name of the sentence-transformers model
        
    Returns:
        NLPProcessor: Initialized NLP processor
    """
    return NLPProcessor(model_name=model_name)


if __name__ == "__main__":
    # Test the NLP module
    print("Testing NLP Module...")
    
    processor = create_nlp_processor()
    
    # Test preprocessing
    test_text = "I'm looking for BEAUTIFUL beaches and amazing food in Asia!"
    cleaned = processor.preprocess(test_text)
    print(f"\nOriginal: {test_text}")
    print(f"Cleaned: {cleaned}")
    
    # Test encoding
    test_texts = ["beach vacation", "mountain adventure", "city nightlife"]
    embeddings = processor.encode_texts(test_texts)
    print(f"\nEmbedding shape: {embeddings.shape}")
    
    print("\n✓ NLP Module test completed successfully!")
