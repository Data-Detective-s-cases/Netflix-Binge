#!/usr/bin/env python3
"""
Netflix Binge Dataset - Data Cleaning and Preparation

Hi there! In this script, I'll walk you through how I:
- Load our raw Netflix dataset
- Handle those pesky missing values
- Clean up the duration format (converting to minutes for movies and season counts for shows)
- Parse and organize genre data from the 'listed_in' column
- Convert dates to proper datetime objects for better analysis
- Create some cool new features that will help us identify binge-worthy content
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
from tqdm import tqdm

print("Netflix Binge Analysis - Data Cleaning and Preparation")
print("-" * 50)

# First, I'll load the raw data
print("Loading raw dataset...")
df = pd.read_csv('data.csv')
print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

# Let's take a quick look at what we're working with
print("\nInitial data overview:")
print(f"Columns: {', '.join(df.columns.tolist())}")
print(f"Missing values: {df.isnull().sum().sum()} total")
for col in df.columns:
    missing = df[col].isnull().sum()
    if missing > 0:
        print(f"  - {col}: {missing} missing values ({missing/len(df)*100:.2f}%)")

# I'll clean up the column names to make them more consistent
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Now I'll tackle those missing values
print("\nHandling missing values...")
# For director and cast, I'll replace NaN with 'Unknown'
for col in ['director', 'cast', 'country', 'description']:
    df[col] = df[col].fillna('Unknown')

# Next, I need to extract the duration values in a usable format
print("\nCleaning duration column...")
def extract_duration(duration_str):
    if pd.isna(duration_str):
        return np.nan, "Unknown"
    
    # I'll extract digits and units (minutes or seasons)
    match_min = re.search(r'(\d+)\s*min', str(duration_str))
    match_seasons = re.search(r'(\d+)\s*Season', str(duration_str))
    
    if match_min:
        return int(match_min.group(1)), "Movie"
    elif match_seasons:
        return int(match_seasons.group(1)), "TV Show"
    else:
        return np.nan, "Unknown"

# Now I'll create separate columns for duration value and unit
df['duration_value'], df['duration_type'] = zip(*df['duration'].apply(extract_duration))

# I'll make sure the type column matches my extracted duration type
df['type'] = df['type'].fillna(df['duration_type'])

# Let's convert date_added to a proper datetime format
print("\nConverting date_added to datetime...")
def parse_date_added(date_str):
    if pd.isna(date_str):
        return pd.NaT
    
    try:
        # I'll remove any non-standard characters and parse the date
        clean_date = re.sub(r'[^\w\s,]', '', str(date_str)).strip()
        return pd.to_datetime(clean_date, format='%B %d %Y')
    except:
        return pd.NaT

df['date_added'] = df['date_added'].apply(parse_date_added)

# I'll also convert release_year to a datetime (year start)
df['release_date'] = pd.to_datetime(df['release_year'], format='%Y')

# Now I'll parse out the genres from the listed_in column
print("\nParsing genres from listed_in column...")
def extract_genres(genres_str):
    if pd.isna(genres_str):
        return []
    
    # I'll split by commas and clean up any whitespace
    genres = [genre.strip() for genre in str(genres_str).split(',')]
    return genres

df['genres'] = df['listed_in'].apply(extract_genres)
df['genre_count'] = df['genres'].apply(len)

# Let's create one-hot encoded columns for the genres
print("Creating one-hot encoded genre columns...")
all_genres = set()
for genres in df['genres']:
    all_genres.update(genres)

# I'll focus on the top 20 genres to keep things manageable
top_genres = pd.Series([g for sublist in df['genres'] for g in sublist]).value_counts().head(20).index
for genre in top_genres:
    df[f'genre_{genre.lower().replace(" ", "_")}'] = df['genres'].apply(lambda x: 1 if genre in x else 0)

# Time to create some interesting new features
print("\nCreating new features...")

# I'll calculate the content age (days between release and being added to Netflix)
df['content_age_days'] = (df['date_added'] - df['release_date']).dt.days

# I'll extract just the first country listed as the primary country
df['primary_country'] = df['country'].apply(
    lambda x: x.split(',')[0].strip() if not pd.isna(x) else "Unknown"
)

# Now I'll create a simplified rating category for easier analysis
def simplify_rating(rating):
    if pd.isna(rating):
        return "Unknown"
    rating = str(rating).upper()
    if rating in ['TV-Y', 'TV-Y7', 'TV-G', 'G', 'TV-Y7-FV']:
        return "Kids"
    elif rating in ['TV-PG', 'PG', 'PG-13']:
        return "Family"
    elif rating in ['TV-14', 'TV-MA', 'R', 'NC-17']:
        return "Adult"
    else:
        return "Other"

df['rating_category'] = df['rating'].apply(simplify_rating)

# I'll categorize content based on release decade
df['release_decade'] = (df['release_year'] // 10) * 10

# Let's count words in the description as an engagement metric
df['description_word_count'] = df['description'].apply(lambda x: len(str(x).split()) if not pd.isna(x) else 0)

# Finally, I'll create a binge score proxy (my experimental feature for prediction modeling)
# Higher scores might indicate higher binge-worthiness
def calculate_binge_score(row):
    score = 0
    
    # TV Shows with more seasons might be more binge-worthy
    if row['type'] == 'TV Show':
        score += 2 * row['duration_value'] if not pd.isna(row['duration_value']) else 0
    
    # More recent content might be more heavily promoted
    years_since_release = 2023 - row['release_year'] 
    recency_score = max(0, 10 - years_since_release) if years_since_release <= 10 else 0
    score += recency_score
    
    # More genres might appeal to wider audiences
    score += min(5, row['genre_count'])
    
    # Content with richer descriptions might be more engaging
    if row['description_word_count'] > 30:
        score += 2
        
    return score

df['binge_score'] = df.apply(calculate_binge_score, axis=1)

# All done! Let's save our clean dataset
print("\nSaving cleaned dataset...")
df.to_csv('cleaned_netflix_data.csv', index=False)

print(f"\nCleaning completed. I've saved the cleaned dataset with {df.shape[0]} rows and {df.shape[1]} columns")
print(f"New features added: {', '.join([col for col in df.columns if col not in pd.read_csv('data.csv').columns])}")
print("\nData is ready for analysis! Now you can run 02_comprehensive_analysis.py to dive deeper!") 