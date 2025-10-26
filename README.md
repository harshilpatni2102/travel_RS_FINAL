# ✈️ Travel Destination Recommendation System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-ready travel destination recommendation system that combines **Natural Language Processing (NLP)** with **Content-Based Filtering** to provide intelligent, personalized travel suggestions.

**Academic Project** for:
- 📚 Natural Language Processing (NLP)
- 🎯 Recommendation Systems

---

## 🌟 Features

### 🤖 Natural Language Processing
- **Semantic Understanding**: Uses transformer-based BERT embeddings (`sentence-transformers`) to understand natural language queries
- **Text Preprocessing**: Advanced cleaning, stopword removal, and synonym expansion
- **Contextual Matching**: Computes cosine similarity between user preferences and destination profiles
- **Multi-Feature Integration**: Combines city descriptions, activities, and metadata for rich semantic representation

### 🎯 Recommendation Engine
- **Content-Based Filtering**: Matches user preferences with destination activity profiles
- **Hybrid Approach**: Combines multiple recommendation signals (NLP similarity, content matching, popularity)
- **Customizable Weights**: Adjustable parameters for different recommendation strategies
- **Explainability**: Score breakdown showing how recommendations are computed
- **Smart Filtering**: Region, budget, and rating filters with fallback mechanisms

### 📊 Interactive Visualizations
- **Geographic Mapping**: Interactive world map with destination markers (Folium)
- **Statistical Charts**: Bar charts, pie charts, histograms using Plotly
- **Activity Heatmaps**: Visual comparison of destination activity profiles
- **Score Breakdowns**: Transparent view of recommendation computation

### 💡 User Experience
- **Modern UI**: Clean, responsive interface built with Streamlit
- **Real-time Updates**: Instant recommendations as you type
- **Multiple Views**: Tabs for recommendations, popular destinations, analytics, and search
- **Export Functionality**: Download recommendations as CSV
- **Performance Optimized**: Caching for fast response times

---

## 📁 Project Structure

```
rs_travel/
├── app.py                          # Main Streamlit application
├── nlp_module.py                   # NLP preprocessing & semantic analysis
├── recommender.py                  # Recommendation engine logic
├── utils.py                        # Visualization & helper functions
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── LICENSE                         # MIT License
└── data/
    └── Worldwide-Travel-Cities-Dataset-Ratings-and-Climate.csv
```

### Module Breakdown

#### `app.py` - Frontend Application
- Streamlit web interface
- User input handling
- Tab-based navigation
- Interactive visualizations
- Result display and export

#### `nlp_module.py` - NLP Pipeline
- Text preprocessing (cleaning, tokenization)
- Stopword removal
- Synonym expansion for travel terms
- BERT-based semantic embeddings
- Cosine similarity computation
- Destination text representation

#### `recommender.py` - Recommendation Logic
- Content-based filtering algorithms
- Hybrid score computation
- Filter application (region, budget, rating)
- Popularity-based ranking
- Similar destination finding
- City search functionality

#### `utils.py` - Utilities
- Interactive chart generation (Plotly)
- Geographic mapping (Folium)
- Data formatting and display
- Score visualization
- HTML card generation

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB+ RAM recommended (for BERT models)

### Installation

1. **Clone or download the project**
   ```bash
   cd path/to/rs_travel
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify data file**
   Ensure `Worldwide-Travel-Cities-Dataset-Ratings-and-Climate.csv` is in the `data/` folder

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

---

## 📖 How to Use

### 1. **Natural Language Query**
In the sidebar, describe your ideal travel destination in plain English:

**Examples:**
- *"I want beautiful beaches and great seafood in Southeast Asia"*
- *"Looking for cultural experiences and museums in Europe on a budget"*
- *"Adventure activities like hiking and nature in South America"*
- *"Luxury wellness retreats with spa and relaxation"*

### 2. **Apply Filters** (Optional)
- **Region**: Filter by continent/geographic region
- **Budget**: Choose Budget, Mid-Range, or Luxury
- **Minimum Rating**: Set quality threshold
- **Number of Results**: How many recommendations to show

### 3. **Adjust Weights** (Advanced)
Fine-tune the recommendation algorithm:
- **NLP Similarity**: How much to prioritize semantic matching
- **Content Match**: Weight for activity profile similarity
- **Popularity**: Influence of overall ratings

### 4. **Explore Results**

#### 🎯 Recommendations Tab
- Personalized suggestions with detailed cards
- Score breakdowns showing algorithm transparency
- Download results as CSV

#### 🏆 Popular Destinations Tab
- Top-rated destinations overall
- Filter by region and budget
- Activity-specific rankings

#### 📊 Visualizations Tab
- Interactive charts and graphs
- Activity popularity analysis
- Geographic world map
- Distribution statistics

#### 🔎 Search Tab
- Look up specific cities
- View complete destination profiles
- Find similar destinations

---

## 🧠 Technical Deep Dive

### NLP Architecture

1. **Text Preprocessing**
   ```python
   User Query → Lowercase → Remove Punctuation → Remove Stopwords → Synonym Expansion
   ```

2. **Embedding Generation**
   - Model: `all-MiniLM-L6-v2` (sentence-transformers)
   - Output: 384-dimensional dense vectors
   - Captures semantic meaning beyond keyword matching

3. **Destination Representation**
   - Combines: city name, country, region, description, activities, budget
   - Activity emphasis based on ratings (high-rated activities repeated)
   - Rich context for accurate matching

4. **Similarity Computation**
   ```python
   Similarity = cosine_similarity(query_embedding, destination_embeddings)
   ```

### Recommendation Algorithm

**Hybrid Score Formula:**

```
Final_Score = α × NLP_Score + β × Content_Score + γ × Popularity_Score
```

Where:
- **α (default: 0.5)** - Weight for semantic similarity
- **β (default: 0.3)** - Weight for activity matching
- **γ (default: 0.2)** - Weight for overall popularity

**Normalization:**
- All scores normalized to [0, 1] range
- Ensures fair comparison across different scales

**Ranking:**
- Destinations sorted by final score (descending)
- Top-N results returned to user

### Performance Optimizations

1. **Caching**
   - `@st.cache_data`: Dataset loading
   - `@st.cache_resource`: NLP model loading
   - Embeddings cached per session

2. **Efficient Processing**
   - Vectorized operations with NumPy/Pandas
   - Batch encoding for multiple texts
   - Lazy loading of visualizations

3. **Fallback Mechanisms**
   - Empty query → Popular destinations
   - No filter matches → Remove filters
   - Graceful error handling

---

## 📊 Dataset Information

**Source**: Worldwide Travel Cities Dataset (Ratings and Climate)

**Columns:**
- `city`, `country`, `region` - Geographic information
- `latitude`, `longitude` - Coordinates for mapping
- `short_description` - Text description of destination
- `avg_temp_monthly` - Climate data
- `ideal_durations` - Recommended stay duration
- `budget_level` - Budget, Mid-Range, or Luxury
- **Activity Ratings** (0-5 scale):
  - `culture` - Museums, heritage, history
  - `adventure` - Outdoor activities, sports
  - `nature` - Natural beauty, wildlife
  - `beaches` - Coastal and water activities
  - `nightlife` - Entertainment, clubs, bars
  - `cuisine` - Food quality and variety
  - `wellness` - Spa, yoga, relaxation
  - `urban` - City life, shopping, infrastructure
  - `seclusion` - Privacy, tranquility

---

## 🎓 Academic Alignment

### Natural Language Processing Components

✅ **Text Preprocessing**
- Tokenization, stopword removal, normalization
- Domain-specific synonym expansion

✅ **Semantic Embeddings**
- Transformer-based language models (BERT)
- Dense vector representations
- Transfer learning from pre-trained models

✅ **Similarity Metrics**
- Cosine similarity for semantic matching
- Distance-based ranking

✅ **Feature Engineering**
- Multi-modal text combination
- Weighted feature representation

### Recommendation System Components

✅ **Content-Based Filtering**
- Feature-based similarity computation
- User preference modeling
- Activity profile matching

✅ **Hybrid Approaches**
- Multiple recommendation signals
- Weighted score aggregation
- Adaptive weighting

✅ **Explainability**
- Score decomposition
- Transparent ranking factors
- User-interpretable results

✅ **Evaluation & Filtering**
- Constraint satisfaction
- Multi-criteria ranking
- Fallback strategies

---

## 📈 Future Enhancements

### Potential Improvements
- 🌐 **Multilingual Support**: Translate queries to English automatically
- 📱 **Mobile Optimization**: Responsive design for smartphones
- 🔐 **User Accounts**: Save preferences and search history
- 🌤️ **Live Data Integration**: Real-time weather and events APIs
- 🤝 **Collaborative Filtering**: User-based recommendations using `surprise` library
- 🎨 **Theme Customization**: Light/dark mode toggle
- 🗣️ **Voice Input**: Speech-to-text for queries
- 📸 **Image Integration**: Visual destination galleries
- 💬 **Chatbot Interface**: Conversational recommendation flow
- 📊 **A/B Testing**: Compare recommendation strategies

### Scalability Considerations
- Database integration (PostgreSQL, MongoDB)
- Caching layer (Redis)
- Containerization (Docker)
- Cloud deployment (AWS, Azure, GCP)
- Load balancing for multiple users

---

## 🛠️ Development

### Running Tests
```bash
# Test individual modules
python nlp_module.py
python recommender.py
python utils.py
```

### Code Style
- Follows PEP 8 guidelines
- Type hints for function signatures
- Comprehensive docstrings
- Inline comments for clarity

### Dependencies Management
```bash
# Update requirements
pip freeze > requirements.txt

# Install specific package
pip install package-name

# Upgrade all packages
pip install --upgrade -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow existing code style
- Add docstrings for new functions
- Update README for new features
- Test thoroughly before submitting

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **sentence-transformers**: Pre-trained BERT models for semantic embeddings
- **Streamlit**: Rapid web application framework
- **Plotly**: Interactive visualization library
- **Folium**: Geospatial mapping
- **scikit-learn**: Machine learning utilities
- **Pandas & NumPy**: Data processing foundations

---

## 📧 Contact

For questions, suggestions, or academic inquiries:

- **GitHub**: [Your GitHub Profile]
- **Email**: [Your Email]
- **LinkedIn**: [Your LinkedIn]

---

## 🎯 Project Goals Achieved

✅ **Complete NLP Pipeline**: Text preprocessing → Embeddings → Similarity  
✅ **Robust Recommendation Engine**: Content-based + Hybrid approach  
✅ **Interactive Visualization**: Charts, maps, and analytics  
✅ **Production-Ready Code**: Modular, documented, and optimized  
✅ **Academic Documentation**: Clear alignment with course objectives  
✅ **User-Friendly Interface**: Intuitive design for all skill levels  

---

**Built with ❤️ for academic excellence | 2025**

*This project demonstrates the integration of cutting-edge NLP techniques with traditional recommendation algorithms to solve real-world problems in the travel domain.*
