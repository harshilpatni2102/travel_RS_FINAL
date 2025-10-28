"""
TravelAI - Modern Centered UI (NO SIDEBAR)
Completely New Design with Model Selection
"""

import streamlit as st
import pandas as pd

from config import DATASET_PATH, GEMINI_API_KEY
from embedding_manager import get_embedding_manager
from smart_recommender import create_recommender
from content_based_recommender import create_content_based_recommender
from gemini_module import get_gemini_enhancer, GEMINI_AVAILABLE

# Config - NO SIDEBAR
st.set_page_config(
    page_title="TravelAI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Dark UI
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
*{font-family:'Plus Jakarta Sans',sans-serif}
[data-testid="stSidebar"]{display:none}
.main{background:linear-gradient(180deg,#0f172a,#1e293b,#334155);background-attachment:fixed}
.block-container{max-width:1400px!important;padding:3rem 2rem!important}
.hero-title{font-size:7rem;font-weight:800;text-align:center;background:linear-gradient(135deg,#60a5fa,#a78bfa,#f472b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:1rem;letter-spacing:-3px;line-height:1.1}
.hero-subtitle{text-align:center;font-size:2.2rem;color:#cbd5e1;font-weight:400;margin-bottom:4rem}
.stTextInput input{background:rgba(30,41,59,0.6)!important;border:2px solid rgba(96,165,250,0.3)!important;border-radius:20px!important;color:white!important;font-size:1.6rem!important;padding:1.8rem!important;transition:all 0.3s ease!important}
.stTextInput input:focus{border-color:#60a5fa!important;box-shadow:0 0 0 4px rgba(96,165,250,0.2)!important;background:rgba(255,255,255,0.12)!important}
.stTextInput input::placeholder{color:#94a3b8!important}
.stButton>button{background:linear-gradient(135deg,#60a5fa,#a78bfa,#f472b6)!important;color:white!important;border:none!important;border-radius:20px!important;padding:1.8rem 3.5rem!important;font-size:1.8rem!important;font-weight:700!important;transition:all 0.3s ease!important;box-shadow:0 10px 40px rgba(96,165,250,0.4)!important;width:100%!important}
.stButton>button:hover{transform:translateY(-4px)!important;box-shadow:0 20px 50px rgba(59,130,246,0.7)!important}
.destination-card{background:rgba(255,255,255,0.06);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,0.1);border-radius:24px;padding:3rem;margin-bottom:2.5rem;position:relative;overflow:hidden;transition:all 0.3s ease}
.destination-card::before{content:'';position:absolute;top:0;left:0;right:0;height:5px;background:linear-gradient(90deg,#3b82f6,#8b5cf6,#ec4899)}
.destination-card:hover{transform:translateY(-8px);border-color:rgba(96,165,250,0.5);box-shadow:0 25px 60px rgba(59,130,246,0.4)}
.destination-title{font-size:3.5rem;font-weight:800;color:white;margin-bottom:0.8rem;letter-spacing:-1px}
.destination-location{color:#94a3b8;font-size:1.6rem;margin-bottom:2rem;font-weight:500}
.destination-description{font-size:1.4rem;line-height:2;color:#cbd5e1;background:rgba(15,23,42,0.6);padding:2rem;border-radius:16px;border-left:5px solid rgba(96,165,250,0.6);margin:2rem 0;font-weight:400}
.activity-pill{display:inline-block;background:rgba(96,165,250,0.2);border:1px solid rgba(96,165,250,0.3);color:#60a5fa;padding:0.8rem 1.6rem;border-radius:14px;font-size:1.2rem;font-weight:700;margin:0.4rem;transition:all 0.2s ease}
.activity-pill:hover{background:rgba(96,165,250,0.3);transform:scale(1.05)}
[data-testid="stMetricValue"]{font-size:3.5rem!important;font-weight:900!important;color:#60a5fa!important}
[data-testid="stMetricLabel"]{color:#cbd5e1!important;font-size:1.4rem!important;font-weight:700!important}
h1,h2,h3,h4{color:white!important}
h2{font-size:4rem!important;font-weight:800!important;margin-top:4rem!important;margin-bottom:2rem!important;text-align:center!important}
h3{font-size:2.5rem!important;font-weight:700!important;color:#e2e8f0!important}
.streamlit-expanderHeader{background:rgba(255,255,255,0.05)!important;border:1px solid rgba(255,255,255,0.1)!important;border-radius:14px!important;color:white!important;font-weight:700!important;padding:1.5rem 2rem!important;font-size:1.4rem!important}
.stAlert{background:rgba(96,165,250,0.1)!important;border:1px solid rgba(96,165,250,0.3)!important;border-radius:14px!important;color:#cbd5e1!important;font-size:1.4rem!important;padding:1.5rem!important}
#MainMenu,footer,header{visibility:hidden}
::-webkit-scrollbar{width:14px}
::-webkit-scrollbar-track{background:rgba(0,0,0,0.2)}
::-webkit-scrollbar-thumb{background:linear-gradient(135deg,#3b82f6,#8b5cf6);border-radius:10px}
.stNumberInput input{background:rgba(255,255,255,0.08)!important;border:2px solid rgba(255,255,255,0.1)!important;border-radius:14px!important;color:white!important;font-size:1.4rem!important;padding:1.2rem!important}
.stCheckbox label{color:#cbd5e1!important;font-weight:700!important;font-size:1.4rem!important}
.stRadio > div{display:flex;justify-content:center;gap:2rem}
.stRadio label{background:linear-gradient(135deg,rgba(96,165,250,0.15),rgba(167,139,250,0.15))!important;border:2px solid rgba(96,165,250,0.3)!important;border-radius:16px!important;padding:1.2rem 2.5rem!important;font-size:1.4rem!important;font-weight:700!important;color:white!important;transition:all 0.3s ease!important;cursor:pointer!important}
.stRadio label:hover{background:linear-gradient(135deg,rgba(96,165,250,0.3),rgba(167,139,250,0.3))!important;transform:translateY(-3px)!important;box-shadow:0 10px 30px rgba(96,165,250,0.4)!important}
.stRadio label[data-checked="true"]{background:linear-gradient(135deg,#60a5fa,#a78bfa)!important;border-color:#60a5fa!important;box-shadow:0 8px 30px rgba(96,165,250,0.5)!important}
p{font-size:1.3rem!important;line-height:1.8!important}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(DATASET_PATH)


@st.cache_resource
def initialize_embeddings(df):
    manager = get_embedding_manager()
    if not manager.embeddings_exist():
        with st.spinner("🧠 Building AI embeddings..."):
            manager.compute_and_store_embeddings(df)
    return manager


def display_destination(row, index):
    city = row.get('city', 'Unknown')
    country = row.get('country', '')
    similarity = row.get('similarity_score', 0)
    description = row.get('short_description', '')
    
    # Calculate overall rating from activity scores
    activity_cols = ['culture', 'adventure', 'nature', 'beaches', 'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
    ratings = [row.get(col, 0) for col in activity_cols if col in row and pd.notna(row.get(col))]
    rating = sum(ratings) / len(ratings) if ratings else 0
    
    budget = row.get('budget_level', 'Mid-range')
    
    activities = {
        '🎭 Culture': row.get('culture', 0),
        '⛰️ Adventure': row.get('adventure', 0),
        '🌿 Nature': row.get('nature', 0),
        '🏖️ Beaches': row.get('beaches', 0),
        '🎉 Nightlife': row.get('nightlife', 0),
        '🍜 Cuisine': row.get('cuisine', 0),
        '🧘 Wellness': row.get('wellness', 0),
        '🏙️ Urban': row.get('urban', 0),
        '🏞️ Seclusion': row.get('seclusion', 0)
    }
    
    top_activities = {k: v for k, v in activities.items() if v >= 3.5}
    
    st.markdown('<div class="destination-card">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        st.markdown(f'<p class="destination-title">#{index+1} {city}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="destination-location">📍 {country}</p>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Match", f"{similarity:.0%}")
    
    with col3:
        st.metric("Rating", f"{rating:.1f}/5")
    
    with col4:
        budget_icon = {"Budget": "💰", "Mid-range": "💎", "Luxury": "👑"}.get(budget, "💵")
        st.metric("Budget", f"{budget_icon}")
    
    # Display description
    if description:
        st.markdown(f'<div class="destination-description">📖 {description}</div>', unsafe_allow_html=True)
    
    if top_activities:
        st.markdown("**🎯 Perfect For:**")
        pills = ""
        for act, score in sorted(top_activities.items(), key=lambda x: x[1], reverse=True)[:6]:
            pills += f'<span class="activity-pill">{act} {score:.1f}</span>'
        st.markdown(pills, unsafe_allow_html=True)
    
    if 'ai_insight' in row and row['ai_insight']:
        with st.expander("🤖 Why This Destination?"):
            st.markdown(f'<div style="background:rgba(96,165,250,0.05);padding:2rem;border-radius:14px;border-left:5px solid #3b82f6;color:#e2e8f0;font-size:1.3rem;line-height:2">{row["ai_insight"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    # Emoji Logo
    st.markdown("<div style='text-align: center; font-size: 5rem; margin: 3rem 0 2rem 0;'>🌍✈️🗺️</div>", unsafe_allow_html=True)
    
    # Main Title - "Travel Recommendation System"
    st.markdown("""
        <h1 style='text-align: center; font-size: 5rem; font-weight: 800; 
                   color: white; margin: 0; letter-spacing: -1px; line-height: 1.2;'>
            Travel Recommendation System
        </h1>
    """, unsafe_allow_html=True)
    
    # Brand Name "TravelAI"
    st.markdown("""
        <h2 style='text-align: center; font-size: 3.5rem; font-weight: 900; 
                   background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   margin: 0.5rem 0 2rem 0; letter-spacing: -2px;'>
            Powered by TravelAI
        </h2>
    """, unsafe_allow_html=True)
    
    # Description
    st.markdown("""
        <p style='text-align: center; font-size: 1.5rem; color: #cbd5e1; margin: 0 auto 3rem auto; 
                  max-width: 900px; line-height: 2; font-weight: 500;'>
            Discover your perfect destination using <strong style='color: #60a5fa;'>BERT NLP Embeddings</strong> and 
            <strong style='color: #a78bfa;'>Content-Based Filtering</strong> from <strong style='color: #f472b6;'>560+ cities worldwide</strong>
        </p>
    """, unsafe_allow_html=True)
    
    # Tech Stack Pills - Properly centered
    st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        st.markdown("""
            <div style='text-align: center; background: rgba(59, 130, 246, 0.2); color: #60a5fa; 
                        padding: 1.2rem 0.8rem; border-radius: 16px; border: 2px solid rgba(96, 165, 250, 0.4);
                        font-size: 1.2rem; font-weight: 700;'>
                🧠 BERT NLP
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='text-align: center; background: rgba(139, 92, 246, 0.2); color: #a78bfa; 
                        padding: 1.2rem 0.8rem; border-radius: 16px; border: 2px solid rgba(167, 139, 250, 0.4);
                        font-size: 1.2rem; font-weight: 700;'>
                🎯 Content-Based
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='text-align: center; background: rgba(236, 72, 153, 0.2); color: #f472b6; 
                        padding: 1.2rem 0.8rem; border-radius: 16px; border: 2px solid rgba(244, 114, 182, 0.4);
                        font-size: 1.2rem; font-weight: 700;'>
                🤖 Gemini AI
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='text-align: center; background: rgba(34, 197, 94, 0.2); color: #4ade80; 
                        padding: 1.2rem 0.8rem; border-radius: 16px; border: 2px solid rgba(74, 222, 128, 0.4);
                        font-size: 1.2rem; font-weight: 700;'>
                📊 560+ Cities
            </div>
        """, unsafe_allow_html=True)
    
    try:
        df = load_data()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return
    
    gemini_enhancer = None
    if GEMINI_AVAILABLE and GEMINI_API_KEY:
        try:
            gemini_enhancer = get_gemini_enhancer(GEMINI_API_KEY)
        except:
            pass
    
    # Model Selection Section
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h2 style="font-size: 3.5rem; color: white; margin-bottom: 1rem;">🤖 Choose Your Recommendation Engine</h2>
        <p style="font-size: 1.4rem; color: #94a3b8; line-height: 1.8; max-width: 900px; margin: 0 auto;">
            Select the AI model that best fits your search style. Each model uses different algorithms to find your perfect destination.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2)); 
                    padding: 2.5rem; border-radius: 20px; border: 2px solid rgba(96, 165, 250, 0.3); height: 100%;">
            <div style="font-size: 3.5rem; text-align: center; margin-bottom: 1.5rem;">🎯</div>
            <h3 style="text-align: center; color: #60a5fa; margin-bottom: 1.5rem; font-size: 2.2rem;">Content-Based Filtering</h3>
            <p style="font-size: 1.3rem; color: #cbd5e1; line-height: 2; margin-bottom: 1.5rem;">
                <strong>How it works:</strong> Matches destinations based on <strong>keyword detection</strong> and <strong>activity scores</strong> (culture, adventure, nature, etc.).
            </p>
            <p style="font-size: 1.2rem; color: #94a3b8; line-height: 1.8;">
                ⚡ <strong>Fast & Lightweight</strong><br>
                🎨 Direct activity matching<br>
                📊 Simple keyword filtering
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(236, 72, 153, 0.2)); 
                    padding: 2.5rem; border-radius: 20px; border: 2px solid rgba(167, 139, 250, 0.3); height: 100%;">
            <div style="font-size: 3.5rem; text-align: center; margin-bottom: 1.5rem;">🧠</div>
            <h3 style="text-align: center; color: #a78bfa; margin-bottom: 1.5rem; font-size: 2.2rem;">Hybrid NLP (BERT)</h3>
            <p style="font-size: 1.3rem; color: #cbd5e1; line-height: 2; margin-bottom: 1.5rem;">
                <strong>How it works:</strong> Uses <strong>BERT embeddings</strong> for semantic similarity + activity filtering. Understands <strong>context & meaning</strong>.
            </p>
            <p style="font-size: 1.2rem; color: #94a3b8; line-height: 1.8;">
                🚀 <strong>Advanced AI Power</strong><br>
                🔬 BERT semantic vectors (384D)<br>
                💡 Context-aware understanding
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    model_choice = st.radio(
        "Select Model:",
        ["🎯 Content-Based (Fast)", "🧠 Hybrid NLP (Advanced)"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Show matching explanation
    if "Content-Based" in model_choice:
        st.info("📊 **Matching on:** Keywords (beach, mountain, culture) → Activity Scores (3.5+/5) → Description similarity")
    else:
        st.info("🧠 **Matching on:** BERT Semantic Embeddings (cosine similarity) → Activity Filtering (3.5+/5) → NLP Context Understanding")
    
    # Initialize embeddings only if Hybrid NLP is selected
    if "Hybrid NLP" in model_choice:
        try:
            with st.spinner("🧠 Initializing AI embeddings..."):
                embedding_manager = initialize_embeddings(df)
            recommender = create_recommender(df, gemini_enhancer)
            st.success("✅ Hybrid NLP Model Ready!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return
    else:
        recommender = create_content_based_recommender(df, gemini_enhancer)
        st.success("✅ Content-Based Model Ready!")
    
    st.markdown("---")
    
    # Search Section
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="font-size: 3.5rem; color: white; margin-bottom: 1rem;">🔍 What's Your Dream Destination?</h2>
        <p style="font-size: 1.4rem; color: #94a3b8; line-height: 1.8;">
            Describe what you're looking for and let AI find the perfect match from <strong>560+ destinations worldwide</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search",
            placeholder="e.g., 'Beautiful beaches in India', 'Mountain trekking in Nepal', 'Romantic getaway with great food'",
            label_visibility="collapsed"
        )
    
    with col2:
        # For Hybrid NLP, default to "Show All" mode (higher limit)
        if "Hybrid NLP" in model_choice:
            result_options = ["5", "10", "15", "20", "All Matches"]
            selected_results = st.selectbox("Results", result_options, index=4, label_visibility="collapsed")
            top_n = 100 if selected_results == "All Matches" else int(selected_results)
        else:
            top_n = st.number_input("Results", min_value=1, max_value=15, value=5, label_visibility="collapsed")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # AI insights only available for Hybrid NLP
        if "Hybrid NLP" in model_choice:
            use_ai = st.checkbox("✨ AI Insights", value=True if gemini_enhancer else False, help="Generate detailed explanations with famous attractions using Gemini AI (Hybrid NLP only)")
        else:
            use_ai = False
            st.markdown('<p style="color: #94a3b8; font-size: 1.1rem;">🎯 Dataset Matching Only</p>', unsafe_allow_html=True)
    with col2:
        show_stats = st.checkbox("📊 Statistics", value=True, help="Show search statistics and metrics")
    with col3:
        st.markdown(f'<p style="font-size: 1.2rem; color: #94a3b8; margin-top: 0.5rem;">Model: <strong style="color: #60a5fa;">{"Content-Based" if "Content-Based" in model_choice else "Hybrid NLP"}</strong></p>', unsafe_allow_html=True)
    
    search_btn = st.button("🚀 FIND MY DESTINATION", use_container_width=True)
    
    # Quick searches with description
    st.markdown("""
    <div style="margin: 2rem 0 1rem 0;">
        <h3 style="font-size: 2rem; color: white; margin-bottom: 0.5rem;">💡 Popular Searches</h3>
        <p style="font-size: 1.2rem; color: #94a3b8;">Try these quick searches to explore different types of destinations</p>
    </div>
    """, unsafe_allow_html=True)
    quick_cols = st.columns(5)
    quick_searches = [
        ("🏖️ Beaches", "Beautiful beaches"),
        ("⛰️ Mountains", "Mountain trekking"),
        ("🏛️ Culture", "Cultural heritage"),
        ("🎢 Adventure", "Adventure sports"),
        ("🧘 Wellness", "Wellness retreat")
    ]
    
    for idx, (label, search_term) in enumerate(quick_searches):
        with quick_cols[idx]:
            if st.button(label, use_container_width=True):
                query = search_term
                search_btn = True
    
    # Results
    if search_btn and query:
        with st.spinner("🔍 Searching..."):
            # Content-Based: No AI (just dataset)
            # Hybrid NLP: Use AI insights
            use_ai_for_search = use_ai and gemini_enhancer is not None and "Hybrid NLP" in model_choice
            results = recommender.recommend(query, top_n=top_n, use_ai=use_ai_for_search)
        
        if len(results) > 0:
            if show_stats:
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0 1.5rem 0;">
                    <h3 style="font-size: 2.5rem; color: white; margin-bottom: 0.5rem;">📊 Search Results Overview</h3>
                    <p style="font-size: 1.3rem; color: #94a3b8;">Quick statistics about your search results</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate overall ratings
                activity_cols = ['culture', 'adventure', 'nature', 'beaches', 'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
                ratings = []
                for _, row in results.iterrows():
                    row_ratings = [row.get(col, 0) for col in activity_cols if col in row and pd.notna(row.get(col))]
                    if row_ratings:
                        ratings.append(sum(row_ratings) / len(row_ratings))
                avg_rating = sum(ratings) / len(ratings) if ratings else 0
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("🎯 Found", len(results))
                with col2:
                    st.metric("📊 Avg Match", f"{results['similarity_score'].mean():.0%}")
                with col3:
                    st.metric("⭐ Avg Rating", f"{avg_rating:.1f}")
                with col4:
                    st.metric("🌍 Countries", results['country'].nunique())
                with col5:
                    top_budget = results['budget_level'].mode()[0] if len(results) > 0 else "N/A"
                    st.metric("💰 Budget", top_budget)
            
            st.markdown(f'<h2>✨ Top {len(results)} Destinations</h2><p style="text-align:center;color:#94a3b8;font-size:1.4rem;margin-top:-1rem;margin-bottom:1rem">Matching: <strong style="color:#60a5fa">"{query}"</strong></p>', unsafe_allow_html=True)
            
            # Show what criteria was used
            if "Content-Based" in model_choice:
                st.markdown("""
                <div style="background: rgba(59, 130, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin-bottom: 2rem;">
                    <p style="font-size: 1.3rem; color: #cbd5e1; line-height: 1.8; margin: 0;">
                        <strong style="color: #60a5fa;">🎯 Content-Based Matching:</strong> Results ranked by keyword matches in query + activity scores (culture, adventure, nature, etc.) + description relevance
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #8b5cf6; margin-bottom: 2rem;">
                    <p style="font-size: 1.3rem; color: #cbd5e1; line-height: 1.8; margin: 0;">
                        <strong style="color: #a78bfa;">🧠 Hybrid NLP Matching:</strong> BERT semantic similarity (384D embeddings) + activity filtering (min 3.5/5) + cosine distance ranking
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            for idx, row in results.iterrows():
                display_destination(row, list(results.index).index(idx))
            
        else:
            st.warning("🤔 No destinations found!")
    
    elif not query:
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center;padding:3rem 0">
            <div style="font-size:6rem;margin-bottom:1rem">🌍✈️🗺️</div>
            <h2 style="color:white;margin-bottom:1rem;font-size:3.5rem">How TravelAI Works</h2>
            <p style="color:#94a3b8;font-size:1.4rem;max-width:800px;margin:0 auto 2rem auto;line-height:2">
                Powered by <strong style="color:#60a5fa">BERT NLP</strong> and <strong style="color:#a78bfa">Content-Based Filtering</strong>, 
                TravelAI intelligently matches you with perfect destinations from <strong>560+ cities worldwide</strong> across <strong>9 activity categories</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="text-align:center;padding:2.5rem;background:rgba(255,255,255,0.05);border-radius:20px;border:2px solid rgba(96,165,250,0.2)">
                <div style="font-size:4.5rem;margin-bottom:1.5rem">🧠</div>
                <h3 style="color:white;margin-bottom:1rem;font-size:2rem">Smart NLP</h3>
                <p style="color:#94a3b8;font-size:1.2rem;line-height:1.8">
                    BERT semantic search with 384-dimensional embeddings for context understanding
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align:center;padding:2.5rem;background:rgba(255,255,255,0.05);border-radius:20px;border:2px solid rgba(167,139,250,0.2)">
                <div style="font-size:4.5rem;margin-bottom:1.5rem">🎯</div>
                <h3 style="color:white;margin-bottom:1rem;font-size:2rem">Precise Filtering</h3>
                <p style="color:#94a3b8;font-size:1.2rem;line-height:1.8">
                    Activity-based matching with minimum 3.5/5 threshold across 9 categories
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align:center;padding:2.5rem;background:rgba(255,255,255,0.05);border-radius:20px;border:2px solid rgba(236,72,153,0.2)">
                <div style="font-size:4.5rem;margin-bottom:1.5rem">🤖</div>
                <h3 style="color:white;margin-bottom:1rem;font-size:2rem">AI Insights</h3>
                <p style="color:#94a3b8;font-size:1.2rem;line-height:1.8">
                    Gemini AI generates personalized explanations with famous attractions
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown('<h2 style="font-size:3.5rem;text-align:center;margin:3rem 0 2rem 0">📊 Platform Statistics</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌍 Destinations", f"{len(df):,}")
        with col2:
            st.metric("🗺️ Countries", df['country'].nunique())
        with col3:
            st.metric("🎯 Activities", "9")
        with col4:
            st.metric("🤖 AI", "Yes")
    
    st.markdown("---")
    st.markdown('<div style="text-align:center;padding:2rem;color:#64748b"><p style="font-size:0.95rem;margin-bottom:0.5rem"><strong style="color:#94a3b8">TravelAI</strong> - NLP + Recommendation Systems</p><p style="font-size:0.85rem">Python • Streamlit • BERT • Gemini AI</p></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
