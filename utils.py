"""
Utilities Module for Travel Destination Recommendation System
=============================================================
This module provides visualization and helper functions including:
- Interactive charts (bar, pie, histogram)
- Geographic maps with destination markers
- Data formatting and display utilities

Academic Alignment:
- Demonstrates data visualization techniques
- Implements interactive plotting for user engagement
- Provides explainability through visual analytics
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium import plugins
import streamlit as st
from streamlit_folium import folium_static


# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_top_destinations_bar(df: pd.DataFrame, 
                              top_n: int = 10,
                              score_column: str = 'overall_rating',
                              title: str = 'Top Destinations') -> go.Figure:
    """
    Create an interactive bar chart of top destinations.
    
    Args:
        df (pd.DataFrame): Dataset with destinations
        top_n (int): Number of top destinations to show
        score_column (str): Column to use for ranking
        title (str): Chart title
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Get top N destinations
    if score_column not in df.columns:
        score_column = 'overall_rating'
    
    top_dest = df.nlargest(top_n, score_column)
    
    # Create city labels (city, country)
    if 'city' in top_dest.columns and 'country' in top_dest.columns:
        labels = top_dest['city'] + ', ' + top_dest['country']
    elif 'city' in top_dest.columns:
        labels = top_dest['city']
    else:
        labels = top_dest.index
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=top_dest[score_column],
            marker=dict(
                color=top_dest[score_column],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Score")
            ),
            text=np.round(top_dest[score_column], 2),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                         f'{score_column}: %{{y:.2f}}<br>' +
                         '<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Destination",
        yaxis_title=score_column.replace('_', ' ').title(),
        template='plotly_white',
        height=500,
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig


def plot_destinations_by_continent(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing destination count by continent/region.
    
    Args:
        df (pd.DataFrame): Dataset with destinations
        
    Returns:
        go.Figure: Plotly figure object
    """
    if 'region' not in df.columns:
        return go.Figure()
    
    # Count destinations by region
    region_counts = df['region'].value_counts().reset_index()
    region_counts.columns = ['Region', 'Count']
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=region_counts['Region'],
            y=region_counts['Count'],
            marker=dict(
                color=region_counts['Count'],
                colorscale='Blues',
                showscale=False
            ),
            text=region_counts['Count'],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                         'Destinations: %{y}<br>' +
                         '<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text='Destinations by Continent/Region', x=0.5, xanchor='center'),
        xaxis_title="Region",
        yaxis_title="Number of Destinations",
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def plot_budget_distribution_pie(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing budget level distribution.
    
    Args:
        df (pd.DataFrame): Dataset with destinations
        
    Returns:
        go.Figure: Plotly figure object
    """
    if 'budget_level' not in df.columns:
        return go.Figure()
    
    # Count destinations by budget level
    budget_counts = df['budget_level'].value_counts().reset_index()
    budget_counts.columns = ['Budget Level', 'Count']
    
    # Define colors
    colors = {'Budget': '#66c2a5', 'Mid-Range': '#fc8d62', 'Luxury': '#8da0cb'}
    budget_counts['Color'] = budget_counts['Budget Level'].map(
        lambda x: colors.get(x.title(), '#e78ac3')
    )
    
    # Create pie chart
    fig = go.Figure(data=[
        go.Pie(
            labels=budget_counts['Budget Level'],
            values=budget_counts['Count'],
            marker=dict(colors=budget_counts['Color']),
            hovertemplate='<b>%{label}</b><br>' +
                         'Destinations: %{value}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text='Destinations by Budget Level', x=0.5, xanchor='center'),
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig


def plot_activity_popularity_histogram(df: pd.DataFrame) -> go.Figure:
    """
    Create a histogram showing distribution of activity ratings.
    
    Args:
        df (pd.DataFrame): Dataset with destinations
        
    Returns:
        go.Figure: Plotly figure object
    """
    activity_cols = ['culture', 'adventure', 'nature', 'beaches', 
                     'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
    
    # Filter existing columns
    existing_cols = [col for col in activity_cols if col in df.columns]
    
    if not existing_cols:
        return go.Figure()
    
    # Calculate average rating for each activity
    avg_ratings = df[existing_cols].mean().sort_values(ascending=False)
    
    # Create bar chart (histogram style)
    fig = go.Figure(data=[
        go.Bar(
            x=avg_ratings.index,
            y=avg_ratings.values,
            marker=dict(
                color=avg_ratings.values,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Avg Rating")
            ),
            text=np.round(avg_ratings.values, 2),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                         'Average Rating: %{y:.2f}<br>' +
                         '<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text='Average Ratings by Activity Type', x=0.5, xanchor='center'),
        xaxis_title="Activity",
        yaxis_title="Average Rating",
        template='plotly_white',
        height=450,
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig


def plot_activity_heatmap(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Create a heatmap showing activity profiles of top destinations.
    
    Args:
        df (pd.DataFrame): Dataset with destinations
        top_n (int): Number of destinations to include
        
    Returns:
        go.Figure: Plotly figure object
    """
    activity_cols = ['culture', 'adventure', 'nature', 'beaches', 
                     'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
    
    # Filter existing columns
    existing_cols = [col for col in activity_cols if col in df.columns]
    
    if not existing_cols:
        return go.Figure()
    
    # Get top destinations
    top_dest = df.nlargest(top_n, 'overall_rating')
    
    # Create labels
    if 'city' in top_dest.columns:
        labels = top_dest['city'].tolist()
    else:
        labels = top_dest.index.tolist()
    
    # Get activity data
    activity_data = top_dest[existing_cols].values
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=activity_data,
        x=existing_cols,
        y=labels,
        colorscale='YlOrRd',
        hovertemplate='Destination: %{y}<br>' +
                     'Activity: %{x}<br>' +
                     'Rating: %{z:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Activity Profile Heatmap', x=0.5, xanchor='center'),
        xaxis_title="Activity",
        yaxis_title="Destination",
        template='plotly_white',
        height=max(400, top_n * 30)
    )
    
    return fig


def render_interactive_map(df: pd.DataFrame, 
                           center_lat: float = 20.0,
                           center_lon: float = 0.0,
                           zoom_start: int = 2) -> folium.Map:
    """
    Create an interactive map with destination markers.
    
    Args:
        df (pd.DataFrame): Dataset with destinations (must have lat/lon)
        center_lat (float): Map center latitude
        center_lon (float): Map center longitude
        zoom_start (int): Initial zoom level
        
    Returns:
        folium.Map: Interactive map object
    """
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )
    
    # Check for coordinate columns
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return m
    
    # Add markers for each destination
    for idx, row in df.iterrows():
        if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
            # Create popup text
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4 style="margin-bottom: 5px;">{row.get('city', 'Unknown')}</h4>
                <p style="margin: 3px 0;"><b>Country:</b> {row.get('country', 'N/A')}</p>
                <p style="margin: 3px 0;"><b>Region:</b> {row.get('region', 'N/A')}</p>
                <p style="margin: 3px 0;"><b>Rating:</b> {row.get('overall_rating', 0):.2f}/5.0</p>
                <p style="margin: 3px 0;"><b>Budget:</b> {row.get('budget_level', 'N/A')}</p>
            </div>
            """
            
            # Determine marker color based on rating
            rating = row.get('overall_rating', 0)
            if rating >= 4.5:
                color = 'green'
            elif rating >= 4.0:
                color = 'blue'
            elif rating >= 3.5:
                color = 'orange'
            else:
                color = 'red'
            
            # Add marker
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=row.get('city', 'Unknown'),
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
    
    # Add marker cluster for better performance with many markers
    if len(df) > 50:
        marker_cluster = plugins.MarkerCluster()
        for idx, row in df.iterrows():
            if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
                popup_html = f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4>{row.get('city', 'Unknown')}</h4>
                    <p><b>Rating:</b> {row.get('overall_rating', 0):.2f}/5.0</p>
                </div>
                """
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=popup_html,
                    tooltip=row.get('city', 'Unknown')
                ).add_to(marker_cluster)
        marker_cluster.add_to(m)
    
    return m


def plot_score_breakdown(nlp_score: float, 
                        content_score: float,
                        popularity_score: float,
                        city_name: str) -> go.Figure:
    """
    Create a breakdown chart showing how recommendation score is computed.
    
    Args:
        nlp_score (float): NLP similarity score
        content_score (float): Content-based score
        popularity_score (float): Popularity score
        city_name (str): Name of the city
        
    Returns:
        go.Figure: Plotly figure object
    """
    scores = {
        'NLP Similarity': nlp_score,
        'Content Match': content_score,
        'Popularity': popularity_score
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            marker=dict(
                color=['#1f77b4', '#ff7f0e', '#2ca02c'],
            ),
            text=[f'{v:.3f}' for v in scores.values()],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                         'Score: %{y:.3f}<br>' +
                         '<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(text=f'Score Breakdown for {city_name}', x=0.5, xanchor='center'),
        xaxis_title="Score Component",
        yaxis_title="Score Value",
        template='plotly_white',
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def format_destination_card(row: pd.Series, 
                           show_scores: bool = False) -> str:
    """
    Format destination data as a nice HTML card.
    
    Args:
        row (pd.Series): Destination data
        show_scores (bool): Whether to show recommendation scores
        
    Returns:
        str: HTML formatted card
    """
    import html
    
    city = row.get('city', 'Unknown')
    country = row.get('country', 'N/A')
    region = row.get('region', 'N/A')
    description = row.get('short_description', 'No description available.')
    
    # Escape HTML in description to prevent rendering issues
    description = html.escape(str(description))
    
    rating = row.get('overall_rating', 0)
    budget = row.get('budget_level', 'N/A')
    
    # Activity highlights
    activity_cols = ['culture', 'adventure', 'nature', 'beaches', 
                     'nightlife', 'cuisine', 'wellness', 'urban', 'seclusion']
    highlights = []
    for col in activity_cols:
        if col in row and pd.notna(row[col]) and float(row[col]) >= 4.0:
            highlights.append(col.title())
    
    highlights_text = ', '.join(highlights[:4]) if highlights else 'Various activities'
    
    # Score section
    score_html = ""
    if show_scores and 'final_score' in row:
        score_html = f"""
        <p style="margin: 5px 0; font-size: 0.9em;">
            <b>🎯 Match Score:</b> {row['final_score']:.3f}
        </p>
        """
    
    # Budget emoji
    budget_emoji = {
        'budget': '💰',
        'mid-range': '💰💰',
        'luxury': '💰💰💰'
    }.get(str(budget).lower(), '💰')
    
    card_html = f"""
    <div style="border: 2px solid #e1e4e8; border-radius: 10px; padding: 15px; margin: 10px 0; background: white;">
        <h3 style="margin-top: 0; color: #0366d6;">{city}, {country}</h3>
        <p style="margin: 5px 0; color: #666;"><b>📍 Region:</b> {region}</p>
        <p style="margin: 5px 0;"><b>⭐ Rating:</b> {rating:.2f}/5.0</p>
        <p style="margin: 5px 0;"><b>{budget_emoji} Budget:</b> {budget}</p>
        <p style="margin: 5px 0;"><b>🎨 Highlights:</b> {highlights_text}</p>
        {score_html}
        <p style="margin: 10px 0 0 0; font-style: italic; color: #444;">{description}</p>
    </div>
    """
    
    return card_html


def get_activity_icons() -> Dict[str, str]:
    """
    Get emoji icons for different activities.
    
    Returns:
        Dict[str, str]: Mapping of activity names to emojis
    """
    return {
        'culture': '🏛️',
        'adventure': '🏔️',
        'nature': '🌲',
        'beaches': '🏖️',
        'nightlife': '🎉',
        'cuisine': '🍽️',
        'wellness': '🧘',
        'urban': '🏙️',
        'seclusion': '🏝️'
    }


def display_dataframe_with_style(df: pd.DataFrame, 
                                 columns_to_show: Optional[List[str]] = None) -> None:
    """
    Display a pandas DataFrame with nice styling in Streamlit.
    
    Args:
        df (pd.DataFrame): DataFrame to display
        columns_to_show (List[str]): Specific columns to show (None = all)
    """
    display_df = df.copy()
    
    if columns_to_show:
        available_cols = [col for col in columns_to_show if col in display_df.columns]
        display_df = display_df[available_cols]
    
    # Format numeric columns
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if display_df[col].max() <= 5:  # Likely a rating
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True
    )


if __name__ == "__main__":
    # Test the utilities module
    print("Testing Utilities Module...")
    
    # Create sample data
    sample_data = {
        'city': ['Paris', 'Bali', 'Tokyo', 'New York', 'Barcelona'],
        'country': ['France', 'Indonesia', 'Japan', 'USA', 'Spain'],
        'region': ['Europe', 'Asia', 'Asia', 'North America', 'Europe'],
        'latitude': [48.8566, -8.4095, 35.6762, 40.7128, 41.3851],
        'longitude': [2.3522, 115.1889, 139.6503, -74.0060, 2.1734],
        'culture': [5.0, 4.0, 4.5, 4.0, 4.5],
        'beaches': [2.0, 5.0, 2.0, 2.5, 4.0],
        'overall_rating': [4.5, 4.7, 4.6, 4.4, 4.5],
        'budget_level': ['Luxury', 'Mid-Range', 'Mid-Range', 'Luxury', 'Mid-Range']
    }
    
    df = pd.DataFrame(sample_data)
    
    print(f"Sample data created with {len(df)} destinations")
    print(f"Activity icons: {list(get_activity_icons().keys())}")
    
    print("\n✓ Utilities Module test completed successfully!")
