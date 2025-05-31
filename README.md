# Netflix Binge-Watching Analysis Project

This project analyzes a Netflix dataset to uncover patterns in binge-worthy content and predict what makes users binge-watch shows and movies.

## Project Structure

- **01_data_cleaning.py**: Preprocesses the raw Netflix dataset, handling missing values, standardizing formats, and generating new features
- **02_comprehensive_analysis.py**: Performs in-depth analysis of the cleaned data, generating visualizations and statistical insights
- **streamlit_app.py**: Interactive dashboard application to explore the analysis results
- **netflix_analysis_results.json**: JSON file containing all analysis results
- **binge_analysis_summary.txt**: Text summary of key findings
- **plots/**: Directory containing all generated visualization images
- **cleaned_netflix_data.csv**: Cleaned and feature-engineered dataset

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/netflix-binge-analysis.git
cd netflix-binge-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Cleaning

Run the data cleaning script to prepare the dataset:

```bash
python 01_data_cleaning.py
```

This creates `cleaned_netflix_data.csv` with the following new features:
- Duration value and type (minutes for movies, seasons for TV shows)
- Genre lists and counts
- Content age metrics
- Binge score potential

### Comprehensive Analysis

Run the analysis script to generate visualizations and insights:

```bash
python 02_comprehensive_analysis.py
```

This generates:
- Multiple visualization plots in the `/plots` directory
- Statistical analysis in the `netflix_analysis_results.json` file
- A summary report in `binge_analysis_summary.txt`

### Interactive Dashboard

Launch the Streamlit dashboard to explore the findings:

```bash
streamlit run streamlit_app.py
```

The dashboard includes:
- Interactive filters for genre, year, country, and rating
- Dynamic visualizations that update based on selections
- A binge-worthiness prediction algorithm
- Insights into what makes content binge-worthy

## Key Findings

The analysis reveals several factors that contribute to binge-worthiness:

1. **Duration Patterns**:
   - TV shows with 2-5 seasons have the highest binge potential
   - Movies between 90-120 minutes are optimal for bingeing

2. **Genre Impact**:
   - Crime, Thriller, and Sci-Fi genres have the highest binge scores
   - Certain genre combinations (e.g., Crime + Thriller) perform exceptionally well

3. **Release Timing**:
   - Content added to Netflix within 1-3 years of production performs best
   - Older content (10+ years) tends to have lower binge potential

4. **International Content**:
   - Content from certain countries consistently achieves higher binge scores
   - International content is growing in binge potential year over year

5. **Rating Influence**:
   - Adult and Family content has higher binge potential than Kids content
   - Content rating significantly impacts viewing patterns

## Binge Score Formula

Based on our analysis, we've developed a formula to predict binge-worthiness:

```
Binge Score = (Duration Factor) + (Recency Factor) + (Genre Diversity) 
            + (Description Richness) + (Rating Bonus) + (Genre-specific Bonuses)
```

The interactive prediction model in the Streamlit app allows you to explore how different content characteristics affect the binge score.

## Dataset

The original Netflix dataset (`data.csv`) contains information about movies and TV shows available on Netflix, including:
- Title, director, cast information
- Release year and date added to Netflix
- Rating, duration, and genres
- Country of origin and description

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 