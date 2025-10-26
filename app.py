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
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# Import custom modules
from nlp_module import create_nlp_processor
from recommender import create_recommender
from utils import (
    plot_top_destinations_bar,
    plot_destinations_by_continent,
    plot_budget_distribution_pie,
    plot_activity_popularity_histogram,
    plot_activity_heatmap,
    render_interactive_map,
    plot_score_breakdown,
    format_destination_card,
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
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 10px;
        border-bottom: 3px solid #1f77b4;
    }
    h2 {
        color: #ff7f0e;
    }
    h3 {
        color: #2ca02c;
    }
    .stButton>button {
        width: 100%;
        background-color: #0366d6;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0256c7;
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
    
    # Header
    st.title("✈️ Travel Destination Recommendation System")
    st.markdown("""
    ### Discover Your Perfect Travel Destination
    *Powered by Natural Language Processing and Intelligent Recommendation Algorithms*
    
    This system combines **NLP-based semantic understanding** with **content-based filtering** 
    to provide personalized travel recommendations based on your preferences.
    """)
    
    # Load data
    with st.spinner("Loading destination data..."):
        df = load_data()
    
    # Initialize components
    nlp_processor = initialize_nlp_processor()
    recommender = create_recommender(df)
    
    # Sidebar - Filters and Inputs
    st.sidebar.header("🔍 Search & Filter")
    
    # Natural language query input
    st.sidebar.subheader("Tell Us What You Want")
    user_query = st.sidebar.text_area(
        "Describe your ideal destination:",
        placeholder="E.g., I want to relax on beautiful beaches in Asia with great food and affordable prices",
        height=100,
        help="Use natural language to describe your travel preferences. The AI will understand and find matching destinations!"
    )
    
    # Filter options
    st.sidebar.subheader("Additional Filters")
    
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
        st.markdown("**Recommendation Weights:**")
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
                        card_html = format_destination_card(row, show_scores=True)
                        st.markdown(card_html, unsafe_allow_html=True)
                        
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
                                st.plotly_chart(fig, width="stretch")
                            
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
                st.plotly_chart(fig, width="stretch")
                
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
            st.plotly_chart(fig, width="stretch")
    
    # TAB 3: VISUALIZATIONS
    with tabs[2]:
        st.header("Data Visualizations & Analytics")
        
        # Row 1: Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Destinations by Region")
            fig = plot_destinations_by_continent(df)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.subheader("Budget Distribution")
            fig = plot_budget_distribution_pie(df)
            st.plotly_chart(fig, width="stretch")
        
        # Row 2: Activity analysis
        st.markdown("---")
        st.subheader("Activity Popularity Analysis")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig = plot_activity_popularity_histogram(df)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("""
            ### 📊 Insights
            
            This chart shows the average ratings across different activity categories 
            for all destinations in our database.
            
            **Key Takeaways:**
            - Identifies most popular activity types
            - Helps understand destination strengths
            - Guides recommendation priorities
            """)
        
        # Row 3: Heatmap
        st.markdown("---")
        st.subheader("Activity Profile Heatmap")
        
        heatmap_n = st.slider("Number of destinations to show:", 10, 30, 15, key="heatmap_n")
        fig = plot_activity_heatmap(df, top_n=heatmap_n)
        st.plotly_chart(fig, width="stretch")
        
        # Row 4: Interactive map
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
                    
                    # Display destination card
                    card_html = format_destination_card(result, show_scores=False)
                    st.markdown(card_html, unsafe_allow_html=True)
                    
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
                        
                        st.plotly_chart(fig, width="stretch")
                    
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: #f0f2f6; border-radius: 10px;">
        <h3>📚 Academic Project Information</h3>
        <p><b>Course:</b> Natural Language Processing & Recommendation Systems</p>
        <p><b>Technologies:</b> Python, Streamlit, Sentence-Transformers (BERT), Scikit-learn, Plotly</p>
        <p><b>Key Features:</b></p>
        <ul style="list-style-position: inside;">
            <li>🤖 <b>NLP:</b> Semantic text understanding using transformer-based embeddings</li>
            <li>🎯 <b>Recommendation:</b> Hybrid content-based filtering with multiple signals</li>
            <li>📊 <b>Visualization:</b> Interactive charts and geographic mapping</li>
            <li>⚡ <b>Performance:</b> Optimized with caching and efficient algorithms</li>
        </ul>
        <p style="margin-top: 15px;">
            <i>Built with ❤️ for academic excellence | © 2025</i>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
