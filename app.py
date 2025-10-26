"""
Travel Destination Recommendation System
========================================
A comprehensive recommendation system combining NLP and Content-Based Filtering
for intelligent travel destination suggestions.

Academic Project for:
- Natural Language Processing (NLP)
- Recommendation Systems

Author: Academic Submission
Date: 2025
"""

import os
import sys
import warnings
import logging
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# Suppress Plotly and Streamlit deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Suppress Streamlit logger warnings about deprecated parameters
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Import custom modules
from nlp_module import create_nlp_processor
from recommender import create_recommender
from utils import (
    plot_top_destinations_bar,
    plot_destinations_by_continent,
    plot_budget_distribution_pie,
    render_interactive_map,
    plot_score_breakdown,
    get_activity_icons,
    display_dataframe_with_style
)
from streamlit_folium import st_folium


# Page configuration
st.set_page_config(
    page_title="Travel Destination Recommender",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main app background and font */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content area background */
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem 3rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling - Light and clean */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #2d3748;
        font-size: 1.1rem;
    }
    
    [data-testid="stSidebar"] label {
        color: #2d3748 !important;
        font-weight: 600;
        font-size: 1.15rem;
    }
    
    [data-testid="stSidebar"] .stTextArea textarea {
        font-size: 1.05rem;
        background-color: white;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }
    
    [data-testid="stSidebar"] .stSelectbox select,
    [data-testid="stSidebar"] input {
        font-size: 1.1rem;
        background-color: white;
        color: #2d3748;
        border: 2px solid #e2e8f0;
    }
    
    /* Slider styling for better visibility */
    [data-testid="stSidebar"] .stSlider {
        padding: 15px 0;
    }
    
    [data-testid="stSidebar"] .stSlider label {
        color: #2d3748 !important;
        font-size: 1.15rem;
        margin-bottom: 10px;
    }
    
    [data-testid="stSidebar"] .stSlider [role="slider"] {
        background-color: #667eea !important;
    }
    
    /* Slider value display */
    [data-testid="stSidebar"] .stSlider div {
        color: #2d3748 !important;
    }
    
    /* Help text in sidebar */
    [data-testid="stSidebar"] .stTooltipIcon {
        color: #667eea !important;
    }
    
    /* Sidebar headers with better styling */
    [data-testid="stSidebar"] h2 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 15px;
        border-radius: 8px;
        margin: 10px 0 20px 0;
        font-size: 1.5rem;
        text-align: center;
    }
    
    [data-testid="stSidebar"] h3 {
        background-color: #e2e8f0;
        color: #2d3748;
        padding: 10px 12px;
        border-radius: 6px;
        margin: 20px 0 15px 0;
        font-size: 1.3rem;
        border-left: 4px solid #667eea;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 25px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white !important;
    }
    
    /* Headers */
    h1 {
        color: #1a202c;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        padding-bottom: 15px;
        border-bottom: 4px solid #667eea;
        margin-bottom: 25px;
        text-align: center;
    }
    
    h2 {
        color: #2d3748;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    h3 {
        color: #4a5568;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    /* Paragraph text */
    p {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #2d3748;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 14px 24px;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.4);
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #5568d3 0%, #653a8b 100%);
    }
    
    /* Sidebar button styling */
    [data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 16px 24px;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        margin: 20px 0;
    }
    
    [data-testid="stSidebar"] .stButton>button:hover {
        background: linear-gradient(135deg, #5568d3 0%, #653a8b 100%);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.5);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
        font-weight: 500;
        color: #4a5568;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 10px;
        font-size: 1.05rem;
        padding: 15px;
    }
    
    /* Containers */
    .stContainer {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 15px 0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2d3748;
    }
    
    /* Sidebar expander */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 12px 15px;
        border-radius: 8px;
        color: white !important;
        font-size: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        background-color: rgba(255, 255, 255, 0.15);
    }
    
    [data-testid="stSidebar"] .streamlit-expanderContent {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 0 0 8px 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: none;
    }
    
    /* Dividers */
    hr {
        margin: 30px 0;
        border: none;
        border-top: 2px solid #e2e8f0;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        font-size: 1.05rem;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 10px;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Slider */
    .stSlider {
        padding: 10px 0;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background-color: #48bb78;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .stDownloadButton>button:hover {
        background-color: #38a169;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """
    Load the travel destinations dataset with caching.
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Try multiple possible paths
    possible_paths = [
        'data/Worldwide-Travel-Cities-Dataset-Ratings-and-Climate.csv',
        'Worldwide-Travel-Cities-Dataset-Ratings-and-Climate.csv',
        'data/Worldwide Travel Cities Dataset (Ratings and Climate).csv',
        'Worldwide Travel Cities Dataset (Ratings and Climate).csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Basic data cleaning
                df = df.dropna(subset=['city'])  # Remove entries without city name
                
                # Create overall_rating column if not present
                activity_columns = ['culture', 'adventure', 'nature', 'beaches', 
                                   'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
                if 'overall_rating' not in df.columns:
                    # Calculate overall rating as mean of activity columns
                    existing_cols = [col for col in activity_columns if col in df.columns]
                    if existing_cols:
                        df['overall_rating'] = df[existing_cols].mean(axis=1)
                    else:
                        df['overall_rating'] = 3.0  # Default rating
                
                return df
            except Exception as e:
                st.error(f"Error loading data from {path}: {e}")
                continue
    
    st.error("❌ Dataset not found! Please ensure the CSV file is in the 'data/' folder.")
    st.stop()


@st.cache_resource
def initialize_nlp_processor():
    """
    Initialize the NLP processor with caching.
    
    Returns:
        NLPProcessor: Initialized NLP processor
    """
    return create_nlp_processor()


def main():
    """Main application function."""
    
    # Header with enhanced styling
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 3rem; margin-bottom: 10px;">✈️ Travel Destination Recommendation System</h1>
            <h3 style="color: #667eea; font-weight: 500; margin-bottom: 20px;">Discover Your Perfect Travel Destination</h3>
            <p style="font-size: 1.1rem; color: #4a5568; font-style: italic;">
                Powered by Natural Language Processing and Intelligent Recommendation Algorithms
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center;">
        <p style="font-size: 1.1rem; margin: 0; color: #2d3748;">
            This system combines <strong>NLP-based semantic understanding</strong> with <strong>content-based filtering</strong> 
            to provide personalized travel recommendations based on your preferences.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("✨ Loading destination data..."):
        df = load_data()
    
    # Initialize components
    nlp_processor = initialize_nlp_processor()
    recommender = create_recommender(df)
    
    # Sidebar - Filters and Inputs
    st.sidebar.markdown("<h2 style='color: white; text-align: center;'>🔍 Search & Filter</h2>", unsafe_allow_html=True)
    
    # Natural language query input
    st.sidebar.markdown("<h3 style='color: white; margin-top: 20px;'>💬 Tell Us What You Want</h3>", unsafe_allow_html=True)
    user_query = st.sidebar.text_area(
        "Describe your ideal destination:",
        placeholder="E.g., I want to relax on beautiful beaches in Asia with great food and affordable prices",
        height=120,
        help="Use natural language to describe your travel preferences. The AI will understand and find matching destinations!"
    )
    
    # Filter options
    st.sidebar.markdown("<h3 style='color: white; margin-top: 30px;'>🎯 Additional Filters</h3>", unsafe_allow_html=True)
    
    # Continent filter
    regions = ['All'] + sorted(df['region'].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox(
        "🌍 Continent/Region:",
        regions,
        help="Filter destinations by geographic region"
    )
    
    # Budget filter
    budgets = ['All'] + sorted(df['budget_level'].dropna().unique().tolist())
    selected_budget = st.sidebar.selectbox(
        "💰 Budget Level:",
        budgets,
        help="Filter by your budget preference"
    )
    
    # Rating filter
    min_rating = st.sidebar.slider(
        "⭐ Minimum Rating:",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        help="Only show destinations with rating above this threshold"
    )
    
    # Number of recommendations
    top_n = st.sidebar.slider(
        "📊 Number of Recommendations:",
        min_value=3,
        max_value=20,
        value=5,
        step=1,
        help="How many destinations to recommend"
    )
    
    # Advanced options (collapsible)
    with st.sidebar.expander("⚙️ Advanced Options"):
        st.markdown("<p style='color: #2d3748; font-weight: 600; font-size: 1.1rem;'>Recommendation Weights:</p>", unsafe_allow_html=True)
        alpha = st.slider("NLP Similarity Weight", 0.0, 1.0, 0.5, 0.1)
        beta = st.slider("Content Match Weight", 0.0, 1.0, 0.3, 0.1)
        gamma = st.slider("Popularity Weight", 0.0, 1.0, 0.2, 0.1)
        
        st.info(f"""
        **Current Configuration:**
        - NLP: {alpha:.0%}
        - Content: {beta:.0%}
        - Popularity: {gamma:.0%}
        """)
    
    # Get recommendations button
    get_recommendations = st.sidebar.button("🚀 Get Recommendations", type="primary")
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    **Dataset Info:**
    - Total Destinations: {len(df)}
    - Regions: {df['region'].nunique()}
    - Countries: {df['country'].nunique()}
    """)
    
    # Main content area with tabs
    tabs = st.tabs(["🎯 Recommendations", "🏆 Popular Destinations", "📊 Visualizations", "🔎 Search"])
    
    # TAB 1: RECOMMENDATIONS
    with tabs[0]:
        st.header("Your Personalized Recommendations")
        
        if get_recommendations or user_query:
            if user_query.strip():
                with st.spinner("🤖 Analyzing your preferences with AI..."):
                    # Compute NLP similarity
                    nlp_scores = nlp_processor.get_nlp_similarity_scores(df, user_query)
                    
                    # Get recommendations
                    recommendations = recommender.recommend(
                        nlp_similarity_scores=nlp_scores,
                        user_query=user_query,
                        continent=selected_region if selected_region != "All" else None,
                        budget=selected_budget if selected_budget != "All" else None,
                        min_rating=min_rating,
                        top_n=top_n,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma
                    )
                
                if len(recommendations) > 0:
                    st.success(f"✅ Found {len(recommendations)} destinations matching your preferences!")
                    
                    # Display recommendations as cards
                    for idx, row in recommendations.iterrows():
                        # Use Streamlit native components with enhanced styling
                        with st.container():
                            # City header with styled background
                            st.markdown(f"""
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                            padding: 15px 20px; border-radius: 10px 10px 0 0; margin-bottom: 0;">
                                    <h2 style="color: white; margin: 0; font-size: 1.8rem;">🌍 {row.get('city', 'Unknown')}, {row.get('country', 'N/A')}</h2>
                                </div>
                                <div style="background: white; padding: 20px; border-radius: 0 0 10px 10px; 
                                            box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
                            """, unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"<p style='font-size: 1.1rem;'><strong>📍 Region:</strong> {row.get('region', 'N/A')}</p>", unsafe_allow_html=True)
                            with col2:
                                rating = row.get('overall_rating', 0)
                                stars = "⭐" * int(round(rating))
                                st.markdown(f"<p style='font-size: 1.1rem;'><strong>⭐ Rating:</strong> {rating:.2f}/5.0 {stars}</p>", unsafe_allow_html=True)
                            with col3:
                                budget = row.get('budget_level', 'N/A')
                                budget_emoji = {'budget': '💰', 'mid-range': '💰💰', 'luxury': '💰💰💰'}.get(str(budget).lower(), '💰')
                                st.markdown(f"<p style='font-size: 1.1rem;'><strong>{budget_emoji} Budget:</strong> {budget}</p>", unsafe_allow_html=True)
                            
                            # Activity highlights
                            activity_cols = ['culture', 'adventure', 'nature', 'beaches', 
                                           'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
                            activity_icons = {'culture': '🏛️', 'adventure': '🏔️', 'nature': '🌲', 'beaches': '🏖️',
                                            'nightlife': '🎉', 'cuisine': '🍽️', 'wellness': '🧘', 'urban': '🏙️', 'seclusion': '🏝️'}
                            highlights = []
                            for col in activity_cols:
                                if col in row and pd.notna(row[col]) and float(row[col]) >= 4.0:
                                    icon = activity_icons.get(col, '✨')
                                    highlights.append(f"{icon} {col.title()}")
                            
                            if highlights:
                                st.markdown(f"<p style='font-size: 1.1rem;'><strong>🎨 Highlights:</strong> {' • '.join(highlights[:4])}</p>", unsafe_allow_html=True)
                            
                            if 'final_score' in row:
                                score_color = "#48bb78" if row['final_score'] > 0.7 else "#667eea" if row['final_score'] > 0.5 else "#f6ad55"
                                st.markdown(f"<p style='font-size: 1.2rem;'><strong>🎯 Match Score:</strong> <span style='color: {score_color}; font-weight: 700;'>{row['final_score']:.3f}</span></p>", unsafe_allow_html=True)
                            
                            # Description
                            description = row.get('short_description', 'No description available.')
                            st.markdown(f"""
                                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                                            padding: 15px; border-radius: 8px; margin-top: 15px;'>
                                    <p style='font-size: 1.05rem; font-style: italic; margin: 0; color: #2d3748;'>{description}</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Show score breakdown in expander
                        with st.expander(f"📈 See Score Breakdown for {row['city']}"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                fig = plot_score_breakdown(
                                    row.get('nlp_score', 0),
                                    row.get('content_score', 0),
                                    row.get('popularity_score', 0),
                                    row['city']
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"score_{idx}")
                            
                            with col2:
                                st.metric("NLP Score", f"{row.get('nlp_score', 0):.3f}")
                                st.metric("Content Score", f"{row.get('content_score', 0):.3f}")
                                st.metric("Popularity Score", f"{row.get('popularity_score', 0):.3f}")
                                st.metric("Final Score", f"{row.get('final_score', 0):.3f}", 
                                         delta=None, delta_color="off")
                    
                    # Download recommendations
                    st.markdown("---")
                    cols_to_export = ['city', 'country', 'region', 'overall_rating', 
                                     'budget_level', 'final_score']
                    export_df = recommendations[[col for col in cols_to_export if col in recommendations.columns]]
                    
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Recommendations as CSV",
                        data=csv,
                        file_name="travel_recommendations.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No destinations match your criteria. Try adjusting your filters!")
            else:
                st.info("👆 Please enter your travel preferences in the sidebar to get personalized recommendations!")
        else:
            # Show example queries
            st.markdown("""
            ### 💡 How to Use:
            
            1. **Describe your preferences** in natural language in the sidebar
            2. **Apply filters** (optional) like region, budget, and minimum rating
            3. **Click "Get Recommendations"** to see results
            
            ### ✨ Example Queries:
            
            - *"I love beaches, water sports, and nightlife in Southeast Asia"*
            - *"Looking for cultural experiences, museums, and good food in Europe"*
            - *"Adventure activities like hiking and nature in South America"*
            - *"Luxury wellness retreats with spa and relaxation"*
            - *"Budget-friendly urban destinations with great street food"*
            
            **The system uses AI to understand your natural language and find the perfect match!**
            """)
    
    # TAB 2: POPULAR DESTINATIONS
    with tabs[1]:
        st.header("Most Popular Destinations")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Filter By:")
            pop_region = st.selectbox(
                "Region:",
                regions,
                key="pop_region"
            )
            pop_budget = st.selectbox(
                "Budget:",
                budgets,
                key="pop_budget"
            )
            pop_n = st.slider(
                "Show Top:",
                5, 20, 10,
                key="pop_n"
            )
        
        with col2:
            popular_dests = recommender.get_popular_destinations(
                top_n=pop_n,
                continent=pop_region if pop_region != "All" else None,
                budget=pop_budget if pop_budget != "All" else None
            )
            
            if len(popular_dests) > 0:
                # Show as interactive chart
                fig = plot_top_destinations_bar(
                    popular_dests,
                    top_n=pop_n,
                    title=f"Top {pop_n} Popular Destinations"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show as table
                st.subheader("Detailed View")
                display_cols = ['city', 'country', 'region', 'overall_rating', 
                               'budget_level', 'culture', 'beaches', 'cuisine']
                display_dataframe_with_style(
                    popular_dests,
                    [col for col in display_cols if col in popular_dests.columns]
                )
            else:
                st.warning("No destinations found with the selected filters.")
        
        # Activity-based popular destinations
        st.markdown("---")
        st.subheader("🎨 Popular by Activity")
        
        activity_icons = get_activity_icons()
        selected_activity = st.selectbox(
            "Select Activity:",
            list(activity_icons.keys()),
            format_func=lambda x: f"{activity_icons[x]} {x.title()}"
        )
        
        activity_dests = recommender.get_destinations_by_activity(selected_activity, top_n=10)
        
        if len(activity_dests) > 0:
            fig = plot_top_destinations_bar(
                activity_dests,
                top_n=10,
                score_column=selected_activity,
                title=f"Top Destinations for {selected_activity.title()}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: VISUALIZATIONS
    with tabs[2]:
        st.header("Data Visualizations & Analytics")
        
        # Row 1: Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Destinations by Region")
            fig = plot_destinations_by_continent(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Budget Distribution")
            fig = plot_budget_distribution_pie(df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 2: Interactive map
        st.markdown("---")
        st.subheader("🗺️ Interactive World Map")
        
        st.info("🔍 Click on markers to see destination details. Colors indicate rating (Green=Excellent, Blue=Great, Orange=Good, Red=Average)")
        
        # Filter map data
        map_df = df.copy()
        if selected_region != "All":
            map_df = map_df[map_df['region'].str.contains(selected_region, case=False, na=False)]
        
        # Limit to top destinations for performance
        map_df = map_df.nlargest(100, 'overall_rating')
        
        travel_map = render_interactive_map(map_df)
        st_folium(travel_map, width=1200, height=600, returned_objects=[])
    
    # TAB 4: SEARCH
    with tabs[3]:
        st.header("🔎 Search for Specific Destination")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                "Enter city name:",
                placeholder="E.g., Paris, Tokyo, Bali",
                help="Search for a specific city to see its details"
            )
            
            if search_query:
                result = recommender.search_destination(search_query)
                
                if result is not None:
                    st.success(f"✅ Found: {result['city']}")
                    
                    # Display destination details using native Streamlit components
                    with st.container():
                        st.markdown(f"### 🌍 {result.get('city', 'Unknown')}, {result.get('country', 'N/A')}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"📍 **Region:** {result.get('region', 'N/A')}")
                        with col2:
                            st.write(f"⭐ **Rating:** {result.get('overall_rating', 0):.2f}/5.0")
                        with col3:
                            budget = result.get('budget_level', 'N/A')
                            budget_emoji = {'budget': '💰', 'mid-range': '💰💰', 'luxury': '💰💰💰'}.get(str(budget).lower(), '💰')
                            st.write(f"{budget_emoji} **Budget:** {budget}")
                        
                        # Activity highlights
                        activity_cols = ['culture', 'adventure', 'nature', 'beaches', 
                                       'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
                        highlights = []
                        for col in activity_cols:
                            if col in result and pd.notna(result[col]) and float(result[col]) >= 4.0:
                                highlights.append(col.title())
                        
                        if highlights:
                            st.write(f"🎨 **Highlights:** {', '.join(highlights)}")
                        
                        # Description
                        description = result.get('short_description', 'No description available.')
                        st.info(f"*{description}*")
                        
                        st.markdown("---")
                    
                    # Additional details
                    st.subheader("📊 Activity Ratings")
                    
                    activity_cols = ['culture', 'adventure', 'nature', 'beaches', 
                                   'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
                    activity_data = {}
                    
                    for col in activity_cols:
                        if col in result and pd.notna(result[col]):
                            activity_data[col] = float(result[col])
                    
                    if activity_data:
                        # Create bar chart
                        import plotly.graph_objects as go
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(activity_data.keys()),
                                y=list(activity_data.values()),
                                marker=dict(
                                    color=list(activity_data.values()),
                                    colorscale='Viridis',
                                    showscale=False
                                ),
                                text=[f"{v:.2f}" for v in activity_data.values()],
                                textposition='auto'
                            )
                        ])
                        
                        fig.update_layout(
                            title=f"Activity Profile: {result['city']}",
                            xaxis_title="Activity",
                            yaxis_title="Rating",
                            template='plotly_white',
                            height=400,
                            xaxis_tickangle=-45,
                            yaxis=dict(range=[0, 5])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Similar destinations
                    st.markdown("---")
                    st.subheader("Similar Destinations")
                    
                    similar_dests = recommender.get_similar_destinations(
                        result['city'],
                        top_n=5
                    )
                    
                    if len(similar_dests) > 0:
                        for idx, row in similar_dests.iterrows():
                            with st.expander(f"{row['city']}, {row['country']} - Similarity: {row.get('similarity_score', 0):.2f}"):
                                st.write(row.get('short_description', 'No description available.'))
                                st.write(f"**Rating:** {row.get('overall_rating', 0):.2f}/5.0")
                                st.write(f"**Budget:** {row.get('budget_level', 'N/A')}")
                    else:
                        st.info("No similar destinations found.")
                
                else:
                    st.error(f"❌ No destination found matching '{search_query}'. Try a different search term.")
        
        with col2:
            st.markdown("""
            ### 🔍 Search Tips
            
            - Enter full or partial city name
            - Search is case-insensitive
            - View complete destination profile
            - Find similar destinations
            - See activity breakdown
            
            ### 📋 Available Cities
            
            Browse our database of destinations from around the world!
            """)
            
            if st.button("Show All Cities"):
                cities_df = df[['city', 'country', 'region']].sort_values('city')
                st.dataframe(cities_df, width="stretch", height=400)
    
    # Footer - Clean and Professional
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                border-radius: 15px; margin-top: 40px;">
        <h3 style="color: #667eea; margin-bottom: 20px;">🌍 About This System</h3>
        <div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap; margin: 20px 0;">
            <div style="text-align: center;">
                <div style="font-size: 2.5rem;">🤖</div>
                <p style="margin: 10px 0 5px 0; font-weight: 600; color: #2d3748;">NLP Powered</p>
                <p style="margin: 0; font-size: 0.95rem; color: #4a5568;">BERT Embeddings</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2.5rem;">🎯</div>
                <p style="margin: 10px 0 5px 0; font-weight: 600; color: #2d3748;">Smart Recommendations</p>
                <p style="margin: 0; font-size: 0.95rem; color: #4a5568;">Hybrid Filtering</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2.5rem;">📊</div>
                <p style="margin: 10px 0 5px 0; font-weight: 600; color: #2d3748;">Interactive Visuals</p>
                <p style="margin: 0; font-size: 0.95rem; color: #4a5568;">Real-time Analytics</p>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2.5rem;">⚡</div>
                <p style="margin: 10px 0 5px 0; font-weight: 600; color: #2d3748;">High Performance</p>
                <p style="margin: 0; font-size: 0.95rem; color: #4a5568;">Optimized Algorithms</p>
            </div>
        </div>
        <p style="margin-top: 25px; color: #4a5568; font-size: 0.95rem;">
            <i>Powered by Python • Streamlit • Sentence-Transformers • Scikit-learn • Plotly</i>
        </p>
        <p style="margin-top: 10px; color: #667eea; font-weight: 600;">
            © 2025 Travel Destination Recommender
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
