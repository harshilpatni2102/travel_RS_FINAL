"""
Embedding Manager - Handles storage and retrieval of pre-computed BERT embeddings
This module demonstrates NLP concepts: vectorization, semantic embeddings, and efficient storage
"""
import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from typing import List, Optional
import streamlit as st

from config import DB_PATH, EMBEDDING_MODEL, DATASET_PATH


class EmbeddingManager:
    """Manages BERT embeddings with SQLite storage for efficient retrieval"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.model = None
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with embeddings table"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS destination_embeddings (
                city_id TEXT PRIMARY KEY,
                city TEXT NOT NULL,
                country TEXT NOT NULL,
                embedding BLOB NOT NULL,
                text_representation TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_model(self):
        """Lazy load the BERT model"""
        if self.model is None:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        return self.model
    
    def embeddings_exist(self) -> bool:
        """Check if embeddings already exist in database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM destination_embeddings")
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    
    def create_destination_text(self, row: pd.Series) -> str:
        """
        Create rich text representation for embedding
        Combines location, description, and activity information
        """
        parts = []
        
        # Location information
        if pd.notna(row.get('city')):
            parts.append(str(row['city']))
        if pd.notna(row.get('country')):
            parts.append(str(row['country']))
        if pd.notna(row.get('region')):
            parts.append(str(row['region']))
        
        # Description
        if pd.notna(row.get('short_description')):
            parts.append(str(row['short_description']))
        
        # Activity features (with emphasis on high ratings)
        activity_cols = ['culture', 'adventure', 'nature', 'beaches', 
                        'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
        
        for col in activity_cols:
            if col in row and pd.notna(row[col]):
                rating = float(row[col])
                if rating >= 4.5:
                    parts.extend([col] * 5)  # High emphasis
                elif rating >= 4.0:
                    parts.extend([col] * 3)  # Medium emphasis
                elif rating >= 3.0:
                    parts.extend([col] * 2)  # Low emphasis
        
        # Budget
        if pd.notna(row.get('budget_level')):
            budget = str(row['budget_level']).lower()
            parts.append(budget)
            if budget == 'budget':
                parts.extend(['affordable', 'cheap'])
            elif budget == 'luxury':
                parts.extend(['upscale', 'premium'])
        
        return ' '.join(parts)
    
    def compute_and_store_embeddings(self, df: pd.DataFrame, force_recompute: bool = False):
        """
        Compute BERT embeddings for all destinations and store in database
        
        Args:
            df: DataFrame with destination data
            force_recompute: If True, recompute even if embeddings exist
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Check if embeddings already exist
        cursor.execute("SELECT COUNT(*) FROM destination_embeddings")
        count = cursor.fetchone()[0]
        
        if count > 0 and not force_recompute:
            conn.close()
            return
        
        # Clear existing embeddings if recomputing
        if force_recompute:
            cursor.execute("DELETE FROM destination_embeddings")
            conn.commit()
        
        # Load model
        model = self._load_model()
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each destination
        for idx, (_, row) in enumerate(df.iterrows()):
            # Update progress
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Computing embeddings: {idx + 1}/{len(df)} - {row.get('city', 'Unknown')}")
            
            # Create text representation
            text = self.create_destination_text(row)
            
            # Compute embedding
            embedding = model.encode([text])[0]
            
            # Serialize embedding
            embedding_blob = pickle.dumps(embedding)
            
            # Store in database
            cursor.execute('''
                INSERT OR REPLACE INTO destination_embeddings 
                (city_id, city, country, embedding, text_representation)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                row.get('id', str(idx)),
                row.get('city', 'Unknown'),
                row.get('country', 'Unknown'),
                embedding_blob,
                text
            ))
        
        conn.commit()
        conn.close()
        
        progress_bar.empty()
        status_text.empty()
    
    def get_embedding(self, city_id: str) -> Optional[np.ndarray]:
        """Retrieve embedding for a specific destination"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT embedding FROM destination_embeddings WHERE city_id = ?",
            (city_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return pickle.loads(result[0])
        return None
    
    def get_all_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Retrieve all embeddings in the same order as dataframe
        
        Args:
            df: DataFrame with destination data (must have 'id' or index)
            
        Returns:
            numpy array of shape (n_destinations, embedding_dim)
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        embeddings = []
        for idx, row in df.iterrows():
            city_id = row.get('id', str(idx))
            cursor.execute(
                "SELECT embedding FROM destination_embeddings WHERE city_id = ?",
                (city_id,)
            )
            result = cursor.fetchone()
            if result:
                embeddings.append(pickle.loads(result[0]))
            else:
                # Return zero vector if not found
                embeddings.append(np.zeros(384))
        
        conn.close()
        return np.array(embeddings)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode user query into embedding vector
        
        Args:
            query: User's search query
            
        Returns:
            Embedding vector
        """
        model = self._load_model()
        return model.encode([query])[0]


# Singleton instance
_embedding_manager = None

def get_embedding_manager() -> EmbeddingManager:
    """Get or create embedding manager instance"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager
