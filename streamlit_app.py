#!/usr/bin/env python3
"""
Netflix Binge Analysis - Streamlit Dashboard App

This app provides an interactive dashboard to explore the Netflix binge analysis results:
- Data filters for genre, year, country, and rating
- Dynamic visualizations based on user selections
- Binge-worthiness prediction algorithm
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Netflix Binge Analysis Dashboard",
    page_icon="🍿",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #E50914;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #221F1F;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #E50914;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
    }
    .divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-top: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.markdown("<h1 class='main-header'>Netflix Binge-Watching Analysis</h1>", unsafe_allow_html=True)

st.markdown("""
This dashboard explores patterns in Netflix content to uncover what makes shows and movies binge-worthy.
Discover how factors like duration, genre, release timing, and ratings influence binge potential.
""")

# Load the data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_netflix_data.csv')
        
        # Convert date columns to datetime
        date_cols = ['date_added', 'release_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Ensure genres column is converted from string to list
        if 'genres' in df.columns and isinstance(df['genres'].iloc[0], str):
            df['genres'] = df['genres'].apply(eval)
            
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the data cleaning script first.")
        return None

# Load analysis results
@st.cache_data
def load_analysis_results():
    try:
        with open('netflix_analysis_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Analysis results file not found. Please run the analysis script first.")
        return {}

df = load_data()
analysis_results = load_analysis_results()

# Check if data is loaded successfully
if df is None:
    st.stop()

# Sidebar filters
st.sidebar.markdown("## Filters")

# Year range filter
min_year = int(df['release_year'].min())
max_year = int(df['release_year'].max())
year_range = st.sidebar.slider(
    "Release Year",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Content type filter
content_types = ['All'] + sorted(df['type'].unique().tolist())
selected_type = st.sidebar.selectbox("Content Type", content_types)

# Genre filter
all_genres = set()
for genres in df['genres']:
    all_genres.update(genres)
selected_genre = st.sidebar.selectbox("Genre", ['All'] + sorted(list(all_genres)))

# Rating category filter
rating_categories = ['All'] + sorted(df['rating_category'].unique().tolist())
selected_rating = st.sidebar.selectbox("Rating Category", rating_categories)

# Country filter
top_countries = df['primary_country'].value_counts().head(10).index.tolist()
selected_country = st.sidebar.selectbox("Country", ['All'] + top_countries)

# Apply filters
filtered_df = df.copy()
if selected_type != 'All':
    filtered_df = filtered_df[filtered_df['type'] == selected_type]
if selected_genre != 'All':
    filtered_df = filtered_df[filtered_df['genres'].apply(lambda x: selected_genre in x)]
if selected_rating != 'All':
    filtered_df = filtered_df[filtered_df['rating_category'] == selected_rating]
if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['primary_country'] == selected_country]
filtered_df = filtered_df[(filtered_df['release_year'] >= year_range[0]) & 
                         (filtered_df['release_year'] <= year_range[1])]

# Display filter summary
st.markdown("### Filtered Dataset")
st.write(f"Showing {len(filtered_df)} out of {len(df)} titles")

# Main dashboard content
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Content Analysis", "Binge Patterns", "Prediction"])

# Tab 1: Overview
with tab1:
    st.markdown("<h2 class='sub-header'>Netflix Content Overview</h2>", unsafe_allow_html=True)
    
    # Key metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{len(filtered_df)}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Titles</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        movies_count = len(filtered_df[filtered_df['type'] == 'Movie'])
        st.markdown(f"<div class='metric-value'>{movies_count}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Movies</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        tvshows_count = len(filtered_df[filtered_df['type'] == 'TV Show'])
        st.markdown(f"<div class='metric-value'>{tvshows_count}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>TV Shows</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        avg_score = filtered_df['refined_binge_score'].mean()
        st.markdown(f"<div class='metric-value'>{avg_score:.1f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Avg Binge Score</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Distribution of content types
    st.markdown("<h3>Content Distribution</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        # Create pie chart for content type distribution
        type_counts = filtered_df['type'].value_counts()
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Content Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create pie chart for rating distribution
        rating_counts = filtered_df['rating_category'].value_counts()
        fig = px.pie(
            values=rating_counts.values,
            names=rating_counts.index,
            title="Rating Category Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top genres bar chart
    st.markdown("<h3>Popular Genres</h3>", unsafe_allow_html=True)
    genres_flat = [genre for sublist in filtered_df['genres'] for genre in sublist]
    genre_counts = pd.Series(genres_flat).value_counts().head(10)
    
    fig = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        title="Top 10 Genres",
        labels={'x': 'Genre', 'y': 'Number of Titles'},
        color=genre_counts.values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Content over time
    st.markdown("<h3>Content Trends Over Time</h3>", unsafe_allow_html=True)
    yearly_content = filtered_df.groupby(filtered_df['release_year'])['show_id'].count()
    
    fig = px.line(
        x=yearly_content.index,
        y=yearly_content.values,
        title="Content Released Per Year",
        labels={'x': 'Release Year', 'y': 'Number of Titles'},
        markers=True
    )
    fig.update_layout(xaxis_title="Release Year", yaxis_title="Number of Titles")
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Content Analysis
with tab2:
    st.markdown("<h2 class='sub-header'>Content Analysis</h2>", unsafe_allow_html=True)
    
    # Duration analysis
    st.markdown("<h3>Duration Analysis</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        # Movie duration histogram
        movies_filtered = filtered_df[filtered_df['type'] == 'Movie']
        if len(movies_filtered) > 0:
            fig = px.histogram(
                movies_filtered,
                x='duration_value',
                nbins=30,
                title="Movie Duration Distribution (minutes)",
                color_discrete_sequence=['indianred']
            )
            fig.update_layout(xaxis_title="Duration (minutes)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No movie data available for the selected filters.")
    
    with col2:
        # TV show seasons count
        tvshows_filtered = filtered_df[filtered_df['type'] == 'TV Show']
        if len(tvshows_filtered) > 0:
            season_counts = tvshows_filtered['duration_value'].value_counts().sort_index()
            fig = px.bar(
                x=season_counts.index,
                y=season_counts.values,
                title="TV Show Seasons Distribution",
                color=season_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_title="Number of Seasons", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No TV show data available for the selected filters.")
    
    # Genre analysis
    st.markdown("<h3>Genre Analysis</h3>", unsafe_allow_html=True)
    
    # Prepare genre binge score data
    if len(filtered_df) > 0:
        genre_scores = {}
        genre_counts = {}
        
        for genres in filtered_df['genres']:
            for genre in genres:
                if genre not in genre_scores:
                    genre_scores[genre] = []
                    genre_counts[genre] = 0
                genre_scores[genre].append(filtered_df.loc[filtered_df['genres'].apply(lambda x: genre in x), 'refined_binge_score'].mean())
                genre_counts[genre] += 1
        
        genre_avg_scores = {genre: np.mean(scores) for genre, scores in genre_scores.items() if genre_counts[genre] >= 5}
        
        if genre_avg_scores:
            # Sort genres by average binge score
            sorted_genres = sorted(genre_avg_scores.items(), key=lambda x: x[1], reverse=True)
            genres = [g[0] for g in sorted_genres[:15]]
            scores = [g[1] for g in sorted_genres[:15]]
            counts = [genre_counts[g] for g in genres]
            
            genre_df = pd.DataFrame({
                'Genre': genres,
                'Average_Binge_Score': scores,
                'Count': counts
            })
            
            fig = px.bar(
                genre_df,
                x='Average_Binge_Score',
                y='Genre',
                color='Count',
                title="Top Genres by Binge Score",
                color_continuous_scale='Viridis',
                orientation='h'
            )
            fig.update_layout(xaxis_title="Average Binge Score", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to analyze genre binge scores with the current filters.")
    else:
        st.info("No data available for the selected filters.")
    
    # Country analysis
    st.markdown("<h3>Country Analysis</h3>", unsafe_allow_html=True)
    
    # Get top countries by content count
    country_counts = filtered_df['primary_country'].value_counts().head(10)
    
    if len(country_counts) > 0:
        country_binge_scores = filtered_df.groupby('primary_country')['refined_binge_score'].mean()
        
        country_df = pd.DataFrame({
            'Country': country_counts.index,
            'Content_Count': country_counts.values,
            'Avg_Binge_Score': [country_binge_scores.get(country, 0) for country in country_counts.index]
        })
        
        fig = px.scatter(
            country_df,
            x='Content_Count',
            y='Avg_Binge_Score',
            size='Content_Count',
            color='Avg_Binge_Score',
            hover_name='Country',
            title="Country Content Volume vs Binge Score",
            color_continuous_scale='Viridis',
            size_max=50
        )
        fig.update_layout(xaxis_title="Number of Titles", yaxis_title="Average Binge Score")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No country data available for the selected filters.")

# Tab 3: Binge Patterns
with tab3:
    st.markdown("<h2 class='sub-header'>Binge-Watching Patterns</h2>", unsafe_allow_html=True)
    
    # Binge score distribution
    st.markdown("<h3>Binge Score Distribution</h3>", unsafe_allow_html=True)
    
    if len(filtered_df) > 0:
        fig = px.histogram(
            filtered_df,
            x='refined_binge_score',
            color='type',
            nbins=20,
            title="Distribution of Binge Scores",
            marginal="box",
            opacity=0.7
        )
        fig.update_layout(xaxis_title="Binge Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")
    
    # Content age vs binge score
    st.markdown("<h3>Content Age vs Binge Score</h3>", unsafe_allow_html=True)
    
    df_with_age = filtered_df.dropna(subset=['date_added', 'release_date']).copy()
    
    if len(df_with_age) > 0:
        # Calculate release gap in years
        df_with_age['release_gap_years'] = (df_with_age['date_added'] - df_with_age['release_date']).dt.days / 365.25
        
        # Create bins for the gap years
        bins = [0, 1, 3, 5, 10, 20, 100]
        labels = ['<1 year', '1-3 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']
        df_with_age['gap_category'] = pd.cut(df_with_age['release_gap_years'], bins=bins, labels=labels)
        
        # Calculate average binge score by gap category
        gap_binge = df_with_age.groupby('gap_category')['refined_binge_score'].mean().reset_index()
        gap_counts = df_with_age['gap_category'].value_counts().to_dict()
        gap_binge['count'] = gap_binge['gap_category'].apply(lambda x: gap_counts.get(x, 0))
        
        fig = px.bar(
            gap_binge,
            x='gap_category',
            y='refined_binge_score',
            color='count',
            title="Content Age vs Binge Score",
            text=gap_binge['count'].apply(lambda x: f"n={x}"),
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis_title="Gap Between Production and Netflix Addition", 
                         yaxis_title="Average Binge Score")
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot of release year vs binge score
        fig = px.scatter(
            filtered_df,
            x='release_year',
            y='refined_binge_score',
            color='type',
            size='duration_value',
            hover_name='title',
            title="Release Year vs Binge Score",
            opacity=0.7
        )
        fig.update_layout(xaxis_title="Release Year", yaxis_title="Binge Score")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for content age analysis with the selected filters.")
    
    # Rating impact on binge score
    st.markdown("<h3>Rating Impact on Binge Score</h3>", unsafe_allow_html=True)
    
    if len(filtered_df) > 0:
        rating_binge = filtered_df.groupby('rating_category')['refined_binge_score'].mean().reset_index()
        rating_counts = filtered_df['rating_category'].value_counts().to_dict()
        rating_binge['count'] = rating_binge['rating_category'].apply(lambda x: rating_counts.get(x, 0))
        
        fig = px.bar(
            rating_binge,
            x='rating_category',
            y='refined_binge_score',
            color='count',
            title="Rating Category vs Binge Score",
            text=rating_binge['count'].apply(lambda x: f"n={x}"),
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis_title="Rating Category", yaxis_title="Average Binge Score")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for rating analysis with the selected filters.")

# Tab 4: Prediction
with tab4:
    st.markdown("<h2 class='sub-header'>Binge-Worthiness Prediction</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Based on our analysis, we've developed a binge-worthiness prediction model. 
    Use the form below to input content characteristics and see the predicted binge score.
    """)
    
    # Create a form for prediction inputs
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Content type selection (Movie or TV Show)
            pred_type = st.selectbox("Content Type", ["Movie", "TV Show"])
            
            # Duration input based on content type
            if pred_type == "Movie":
                duration = st.slider("Movie Duration (minutes)", 60, 240, 105)
                seasons = 0
            else:
                seasons = st.slider("Number of Seasons", 1, 15, 3)
                duration = 0
            
            # Release year
            release_year = st.slider("Release Year", 1900, 2023, 2020)
            
        with col2:
            # Rating category
            rating = st.selectbox("Rating Category", ["Kids", "Family", "Adult", "Other"])
            
            # Number of genres
            genre_count = st.slider("Number of Genres", 1, 10, 3)
            
            # Description length
            desc_length = st.slider("Description Word Count", 10, 100, 30)
            
            # Genre selection
            genres = st.multiselect(
                "Select Primary Genres",
                ["Crime", "Thriller", "Sci-Fi", "Drama", "Comedy", "Action & Adventure", 
                 "Documentaries", "Horror", "International", "Romance"],
                ["Drama", "Thriller"]
            )
        
        # Submit button
        submit = st.form_submit_button("Predict Binge Score")
    
    # When form is submitted, calculate the prediction
    if submit:
        score = 0
        
        # TV Shows with more seasons are more binge-worthy
        if pred_type == "TV Show":
            score += 3 * min(seasons, 5)  # Cap at 5 seasons
        
        # Movies between 90-120 minutes score higher
        elif pred_type == "Movie":
            if 90 <= duration <= 120:
                score += 5
            else:
                score += max(0, 5 - abs(105 - duration) / 15)
        
        # Recent content gets higher scores
        years_since_release = 2023 - release_year 
        recency_score = max(0, 10 - years_since_release) if years_since_release <= 10 else 0
        score += recency_score * 0.5
        
        # More genres might appeal to wider audiences
        score += min(5, genre_count) * 0.5
        
        # Content with richer descriptions
        score += min(5, desc_length / 15)
        
        # Adult/Family content tends to score better
        if rating in ['Adult', 'Family']:
            score += 2
            
        # Genre bonuses based on analysis
        for genre in genres:
            if genre == 'Crime': score += 1
            if genre == 'Thriller': score += 1
            if genre == 'Sci-Fi': score += 1
            if genre == 'Drama': score += 0.5
            if genre == 'Comedy': score += 0.5
            
        # Cap at 10
        final_score = min(10, score)
        
        # Display prediction result with a gauge chart
        st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display the score in a big, eye-catching way
            st.markdown(f"<div style='background-color: #E50914; color: white; padding: 20px; border-radius: 10px; text-align: center;'><h1 style='font-size: 3rem; margin: 0;'>{final_score:.1f}/10</h1><p>Binge Score</p></div>", unsafe_allow_html=True)
            
            # Interpretation
            if final_score >= 8:
                st.success("This content has excellent binge potential!")
            elif final_score >= 6:
                st.info("This content has good binge potential.")
            elif final_score >= 4:
                st.warning("This content has moderate binge potential.")
            else:
                st.error("This content has low binge potential.")
        
        with col2:
            # Create a gauge chart to visualize the score
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = final_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Binge-Worthiness Score"},
                gauge = {
                    'axis': {'range': [0, 10], 'tickwidth': 1},
                    'bar': {'color': "#E50914"},
                    'steps': [
                        {'range': [0, 3], 'color': "#ffebe6"},
                        {'range': [3, 6], 'color': "#ffcccc"},
                        {'range': [6, 8], 'color': "#ff9999"},
                        {'range': [8, 10], 'color': "#ff6666"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': final_score
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Factors contributing to the score
        st.markdown("<h4>Key Factors Influencing The Score</h4>", unsafe_allow_html=True)
        
        factors_df = pd.DataFrame({
            'Factor': [],
            'Impact': []
        })
        
        if pred_type == "TV Show":
            factors_df = pd.concat([factors_df, pd.DataFrame({
                'Factor': [f"TV Show with {seasons} season(s)"],
                'Impact': [min(3 * min(seasons, 5), 15)]
            })], ignore_index=True)
        else:
            if 90 <= duration <= 120:
                impact = 5
                note = "optimal duration"
            else:
                impact = max(0, 5 - abs(105 - duration) / 15)
                note = "non-optimal duration"
            factors_df = pd.concat([factors_df, pd.DataFrame({
                'Factor': [f"Movie duration: {duration} min ({note})"],
                'Impact': [impact]
            })], ignore_index=True)
        
        # Add other factors
        other_factors = [
            (f"Content recency: {2023 - release_year} years old", recency_score * 0.5),
            (f"Genre diversity: {genre_count} genres", min(5, genre_count) * 0.5),
            (f"Description length: {desc_length} words", min(5, desc_length / 15)),
            (f"Rating category: {rating}", 2 if rating in ['Adult', 'Family'] else 0)
        ]
        
        for genre in genres:
            impact = 0
            if genre == 'Crime': impact = 1
            elif genre == 'Thriller': impact = 1
            elif genre == 'Sci-Fi': impact = 1
            elif genre == 'Drama': impact = 0.5
            elif genre == 'Comedy': impact = 0.5
            
            if impact > 0:
                other_factors.append((f"Genre bonus: {genre}", impact))
        
        factors_df = pd.concat([factors_df, pd.DataFrame({
            'Factor': [f[0] for f in other_factors],
            'Impact': [f[1] for f in other_factors]
        })], ignore_index=True)
        
        # Sort factors by impact
        factors_df = factors_df.sort_values('Impact', ascending=False)
        
        # Create a horizontal bar chart of factors
        fig = px.bar(
            factors_df,
            y='Factor',
            x='Impact',
            orientation='h',
            title="Factors Contributing to Binge Score",
            color='Impact',
            color_continuous_scale='Reds'
        )
        fig.update_layout(xaxis_title="Impact on Score", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

# Footer with additional information
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
**About this dashboard:**
This interactive dashboard was created as part of a Netflix binge-watching analysis project.
Data is based on a Netflix dataset containing movies and TV shows up to 2021.
""")

# Add app execution information
st.sidebar.markdown("### About")
st.sidebar.info("""
**Netflix Binge Analysis Dashboard**

This app explores patterns in Netflix content to understand what makes shows and movies binge-worthy.

Data last updated: 2023
""")

# Add the Netflix logo in the sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Netflix_2015_logo.svg/1920px-Netflix_2015_logo.svg.png", width=200) 