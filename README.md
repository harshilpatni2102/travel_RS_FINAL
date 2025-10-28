# 🌍 Travel Destination Recommendation System

**An AI-Powered Travel Recommender combining NLP and Recommendation Systems**

This is an academic project demonstrating advanced concepts in Natural Language Processing and Recommendation Systems, featuring BERT-based semantic search, content-based filtering, and AI validation.

## 🎯 Project Overview

This system helps users discover perfect travel destinations using natural language queries. It combines:

- **NLP-based Semantic Search**: BERT embeddings for understanding query meaning
- **Content-Based Filtering**: Activity and location-based filtering
- **AI Validation**: Gemini AI for generating personalized insights
- **Efficient Storage**: Pre-computed embeddings stored in SQLite database

## ✨ Features

### 🧠 NLP Capabilities
- **BERT Embeddings**: Uses `all-MiniLM-L6-v2` transformer model
- **Semantic Understanding**: Understands natural language queries
- **Pre-computed Storage**: Embeddings stored in database for fast retrieval
- **Cosine Similarity**: Accurate semantic matching

### 🎯 Smart Recommendations
- **Location Filtering**: Automatically detects country mentions ("beaches in India")
- **Activity Detection**: Identifies activities (beaches, mountains, culture, etc.)
- **Strict Filtering**: Only shows destinations with high activity scores (≥3.5/5)
- **Ranked Results**: Sorted by semantic similarity

### 🤖 AI Integration
- **Gemini API**: Generates personalized travel insights
- **Contextual Explanations**: AI explains why each destination matches
- **Accurate Validation**: Ensures recommendations are relevant

### 🎨 Beautiful UI
- **Modern Design**: Clean, professional interface
- **Responsive Layout**: Works on all screen sizes
- **Google Fonts**: Inter font family for readability
- **Gradient Backgrounds**: Eye-catching visuals
- **Interactive Cards**: Expandable destination details

## 📁 Project Structure

```
rs_travel/
├── main_app.py              # Main Streamlit application
├── config.py                # Configuration and settings
├── embedding_manager.py     # BERT embedding storage & retrieval
├── smart_recommender.py     # Recommendation engine with AI
├── requirements.txt         # Python dependencies
├── data/                    # Dataset folder
│   └── Worldwide-Travel-Cities-Dataset-Ratings-and-Climate.csv
└── database/               # SQLite database for embeddings
    └── embeddings.db       # Automatically created
```

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/travel_RS_FINAL.git
cd travel_RS_FINAL
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Get your free API key from: https://makersuite.google.com/app/apikey

### 4. Run Application

```bash
streamlit run main_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Dataset

**Source**: Worldwide Travel Cities Dataset (Ratings and Climate)

**Features**:
- **Cities**: 560+ destinations worldwide
- **Attributes**: Culture, Adventure, Nature, Beaches, Nightlife, Cuisine, Wellness, Urban, Seclusion
- **Metadata**: Country, Region, Budget Level, Ratings, Climate Data

## 🎓 Academic Concepts Demonstrated

### Natural Language Processing
1. **Text Preprocessing**: Tokenization, cleaning
2. **Vectorization**: BERT-based embeddings (384-dimensional)
3. **Semantic Similarity**: Cosine similarity computation
4. **Transfer Learning**: Pre-trained transformer models

### Recommendation Systems
1. **Content-Based Filtering**: Activity preference matching
2. **Hybrid Approach**: Combining NLP + content features
3. **Cold Start Solution**: Works without user history
4. **Explainability**: AI-generated insights

### Software Engineering
1. **Database Design**: SQLite for embedding storage
2. **Caching**: Streamlit caching for performance
3. **Modular Architecture**: Separation of concerns
4. **Error Handling**: Robust fallback mechanisms

## 💡 Example Queries

Try these natural language searches:

- `"Beautiful beaches in India"`
- `"Mountain trekking in Nepal"`
- `"Cultural heritage sites in Italy"`
- `"Adventure sports in New Zealand"`
- `"Peaceful wellness retreats in Thailand"`
- `"Luxury beach resorts in Maldives"`
- `"Budget-friendly cities in Europe"`

## 🔧 How It Works

### 1. Query Processing
```
User Query → Location Detection → Activity Detection
```

### 2. Filtering Pipeline
```
Full Dataset → Location Filter → Activity Filter → Filtered Set
```

### 3. Semantic Matching
```
Query → BERT Embedding → Cosine Similarity → Ranked Results
```

### 4. AI Enhancement
```
Top Results → Gemini API → Personalized Insights
```

## 📈 Technical Specifications

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit 1.31.0 |
| NLP Model | all-MiniLM-L6-v2 (BERT) |
| Embeddings | 384-dimensional vectors |
| Database | SQLite 3 |
| AI API | Google Gemini 2.0 Flash |
| Similarity | Cosine Similarity |
| Language | Python 3.11+ |

## 🎯 Key Innovations

1. **Pre-computed Embeddings**: Faster than real-time embedding
2. **Strict Activity Filtering**: Ensures relevant results (≥3.5/5 threshold)
3. **AI Validation**: Gemini verifies and explains recommendations
4. **Smart Location Detection**: Auto-detects country names in queries
5. **Modern UI/UX**: Professional, gradient-based design

## 📝 Academic Alignment

### NLP Course Requirements ✅
- [x] Text preprocessing pipeline
- [x] Tokenization and vectorization
- [x] BERT/Transformer embeddings
- [x] Semantic similarity computation
- [x] Real-world application

### Recommendation Systems Requirements ✅
- [x] Content-based filtering
- [x] Hybrid approach
- [x] Feature engineering
- [x] Ranking algorithm
- [x] Explainability

## 🐛 Troubleshooting

**Issue**: Embeddings not computing
- **Solution**: Delete `database/embeddings.db` and restart app

**Issue**: AI insights not showing
- **Solution**: Check `.env` file has valid GEMINI_API_KEY

**Issue**: Slow first run
- **Solution**: Normal! BERT model downloads on first run (~90MB)

## 📚 References

1. Sentence-Transformers: https://www.sbert.net/
2. Streamlit Documentation: https://docs.streamlit.io/
3. Google Gemini API: https://ai.google.dev/
4. BERT Paper: Devlin et al., 2018

## 👨‍💻 Author

**Academic Project**  
Natural Language Processing + Recommendation Systems  
2025

---

**Note**: This is an academic project demonstrating NLP and Recommendation System concepts. The dataset and recommendations are for educational purposes only.
