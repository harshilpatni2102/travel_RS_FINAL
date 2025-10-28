"""
Configuration module for Travel Recommendation System
Loads API keys and settings
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "database"
DB_DIR.mkdir(exist_ok=True)

# Database
DB_PATH = DB_DIR / "embeddings.db"

# Gemini API Configuration
def load_gemini_api_key():
    """Load Gemini API key from .env file"""
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv('GEMINI_API_KEY')

GEMINI_API_KEY = load_gemini_api_key()

# Dataset
DATASET_PATH = DATA_DIR / "Worldwide-Travel-Cities-Dataset-Ratings-and-Climate.csv"

# NLP Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and efficient BERT model
EMBEDDING_DIM = 384  # Dimension of embeddings

# Recommendation Settings
DEFAULT_TOP_N = 5
MIN_ACTIVITY_SCORE = 3.5  # Minimum score for activity filtering
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity for recommendations
