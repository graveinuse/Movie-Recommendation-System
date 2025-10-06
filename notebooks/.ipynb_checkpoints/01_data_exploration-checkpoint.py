# CELL 1: Import Libraries and Setup
"""
MovieLens 32M - Data Exploration
Author: Satya
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Plotting settings
sns.set_style('whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

print("✅ Libraries loaded successfully!")
print(f"📅 Notebook run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# CELL 2: Load Sample Dataset (For Quick Exploration)
# ============================================================

print("="*60)
print("LOADING SAMPLE DATASET")
print("="*60)

# Load sample data (100K ratings) for quick exploration
print("\n📂 Loading sample dataset...")
ratings_sample = pd.read_csv('../data/sample/ratings_sample.csv')
movies = pd.read_csv('../data/raw/movies.csv')
tags = pd.read_csv('../data/raw/tags.csv')
links = pd.read_csv('../data/raw/links.csv')

print("\n✅ Data loaded successfully!")
print(f"\n📊 Dataset Sizes:")
print(f"   Ratings (sample): {len(ratings_sample):,} rows")
print(f"   Movies: {len(movies):,} rows")
print(f"   Tags: {len(tags):,} rows")
print(f"   Links: {len(links):,} rows")

# ============================================================
# CELL 3: Data Structure Overview
# ============================================================

print("\n" + "="*60)
print("DATA STRUCTURE")
print("="*60)

print("\n1️⃣ RATINGS DataFrame:")
print(ratings_sample.head())
print(f"\nShape: {ratings_sample.shape}")
print(f"Columns: {list(ratings_sample.columns)}")
print(f"\nData Types:")
print(ratings_sample.dtypes)
print(f"\nMemory Usage: {ratings_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n2️⃣ MOVIES DataFrame:")
print(movies.head())
print(f"\nShape: {movies.shape}")

print("\n3️⃣ TAGS DataFrame:")
print(tags.head())
print(f"\nShape: {tags.shape}")

# ============================================================
# CELL 4: Basic Statistics
# ============================================================

print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)

print("\n📊 RATING STATISTICS:")
print(ratings_sample['rating'].describe())

print("\n📈 RATING DISTRIBUTION:")
rating_dist = ratings_sample['rating'].value_counts().sort_index()
for rating, count in rating_dist.items():
    percentage = (count / len(ratings_sample)) * 100
    bar = '█' * int(percentage / 2)
    print(f"   {rating}: {bar} {count:,} ({percentage:.1f}%)")

print(f"\n🎯 KEY METRICS:")
print(f"   • Unique Users: {ratings_sample['userId'].nunique():,}")
print(f"   • Unique Movies: {ratings_sample['movieId'].nunique():,}")
print(f"   • Average Rating: {ratings_sample['rating'].mean():.2f}")
print(f"   • Median Rating: {ratings_sample['rating'].median():.2f}")
print(f"   • Most Common Rating: {ratings_sample['rating'].mode()[0]}")

# ============================================================
# CELL 5: Data Quality Checks
# ============================================================

print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)

# Check for missing values
print("\n🔍 MISSING VALUES:")
print(f"Ratings - Missing: {ratings_sample.isnull().sum().sum()}")
print(f"Movies - Missing: {movies.isnull().sum().sum()}")

# Check for duplicates
print(f"\n🔍 DUPLICATE ROWS:")
print(f"Ratings - Duplicates: {ratings_sample.duplicated().sum()}")
print(f"Movies - Duplicates: {movies.duplicated().sum()}")

# Rating range validation
print(f"\n🔍 RATING RANGE:")
print(f"Min Rating: {ratings_sample['rating'].min()}")
print(f"Max Rating: {ratings_sample['rating'].max()}")
print(f"Valid Range: 0.5 to 5.0 ✓" if ratings_sample['rating'].min() >= 0.5 and ratings_sample['rating'].max() <= 5.0 else "Invalid Range ✗")

print("\n✅ Data quality checks passed!")

# ============================================================
# CELL 6: User Activity Analysis
# ============================================================

print("\n" + "="*60)
print("USER ACTIVITY ANALYSIS")
print("="*60)

# Ratings per user
ratings_per_user = ratings_sample.groupby('userId').size()

print(f"\n👥 USER ACTIVITY:")
print(f"   • Total Users: {len(ratings_per_user):,}")
print(f"   • Average ratings per user: {ratings_per_user.mean():.1f}")
print(f"   • Median ratings per user: {ratings_per_user.median():.0f}")
print(f"   • Min ratings per user: {ratings_per_user.min()}")
print(f"   • Max ratings per user: {ratings_per_user.max()}")

print(f"\n📊 USER ACTIVITY DISTRIBUTION:")
print(ratings_per_user.describe())

# Most active users
print(f"\n🔥 TOP 10 MOST ACTIVE USERS:")
top_users = ratings_per_user.sort_values(ascending=False).head(10)
for rank, (user_id, count) in enumerate(top_users.items(), 1):
    print(f"   {rank:2d}. User {user_id}: {count:,} ratings")

# ============================================================
# CELL 7: Movie Popularity Analysis
# ============================================================

print("\n" + "="*60)
print("MOVIE POPULARITY ANALYSIS")
print("="*60)

# Ratings per movie
ratings_per_movie = ratings_sample.groupby('movieId').size()

print(f"\n🎬 MOVIE STATISTICS:")
print(f"   • Total Movies Rated: {len(ratings_per_movie):,}")
print(f"   • Average ratings per movie: {ratings_per_movie.mean():.1f}")
print(f"   • Median ratings per movie: {ratings_per_movie.median():.0f}")
print(f"   • Min ratings per movie: {ratings_per_movie.min()}")
print(f"   • Max ratings per movie: {ratings_per_movie.max()}")

# Most rated movies
movie_stats = ratings_sample.groupby('movieId').agg({
    'rating': ['count', 'mean']
}).reset_index()
movie_stats.columns = ['movieId', 'count', 'mean_rating']
movie_stats = movie_stats.merge(movies[['movieId', 'title', 'genres']], on='movieId')
movie_stats = movie_stats.sort_values('count', ascending=False)

print(f"\n⭐ TOP 10 MOST RATED MOVIES:")
for rank, row in enumerate(movie_stats.head(10).itertuples(), 1):
    print(f"   {rank:2d}. {row.title}")
    print(f"       Ratings: {row.count:,} | Avg: {row.mean_rating:.2f} | Genres: {row.genres}")

# ============================================================
# CELL 8: Genre Analysis
# ============================================================

print("\n" + "="*60)
print("GENRE ANALYSIS")
print("="*60)

# Extract all genres
all_genres = movies['genres'].str.split('|').explode()
genre_counts = all_genres.value_counts()

print(f"\n🎭 GENRE STATISTICS:")
print(f"   • Total Unique Genres: {len(genre_counts)}")
print(f"   • Movies with genres: {movies[movies['genres'] != '(no genres listed)'].shape[0]:,}")
print(f"   • Movies without genres: {movies[movies['genres'] == '(no genres listed)'].shape[0]:,}")

print(f"\n📊 TOP 10 GENRES:")
for rank, (genre, count) in enumerate(genre_counts.head(10).items(), 1):
    percentage = (count / len(movies)) * 100
    print(f"   {rank:2d}. {genre:20s}: {count:,} movies ({percentage:.1f}%)")

# ============================================================
# CELL 9: Timestamp Analysis
# ============================================================

print("\n" + "="*60)
print("TEMPORAL ANALYSIS")
print("="*60)

# Convert timestamp to datetime
ratings_sample['datetime'] = pd.to_datetime(ratings_sample['timestamp'], unit='s')
ratings_sample['year'] = ratings_sample['datetime'].dt.year
ratings_sample['month'] = ratings_sample['datetime'].dt.month
ratings_sample['hour'] = ratings_sample['datetime'].dt.hour

print(f"\n📅 TEMPORAL STATISTICS:")
print(f"   • Earliest Rating: {ratings_sample['datetime'].min()}")
print(f"   • Latest Rating: {ratings_sample['datetime'].max()}")
print(f"   • Time Span: {(ratings_sample['datetime'].max() - ratings_sample['datetime'].min()).days} days")

print(f"\n📊 RATINGS BY YEAR:")
yearly_ratings = ratings_sample['year'].value_counts().sort_index()
for year, count in yearly_ratings.items():
    print(f"   {year}: {count:,} ratings")

# ============================================================
# CELL 10: Sparsity Analysis
# ============================================================

print("\n" + "="*60)
print("MATRIX SPARSITY ANALYSIS")
print("="*60)

n_users = ratings_sample['userId'].nunique()
n_movies = ratings_sample['movieId'].nunique()
n_ratings = len(ratings_sample)

# Calculate sparsity
possible_ratings = n_users * n_movies
sparsity = 1 - (n_ratings / possible_ratings)

print(f"\n📊 USER-ITEM MATRIX:")
print(f"   • Matrix Size: {n_users:,} users × {n_movies:,} movies")
print(f"   • Possible Ratings: {possible_ratings:,}")
print(f"   • Actual Ratings: {n_ratings:,}")
print(f"   • Sparsity: {sparsity*100:.4f}%")
print(f"   • Density: {(1-sparsity)*100:.4f}%")

print(f"\n💡 INTERPRETATION:")
if sparsity > 0.99:
    print(f"   Matrix is EXTREMELY sparse ({sparsity*100:.2f}%)")
    print(f"   Most user-movie pairs have no ratings")
    print(f"   Collaborative filtering will be challenging")
else:
    print(f"   Matrix sparsity is manageable")

# ============================================================
# CELL 11: Visualization 1 - Rating Distribution
# ============================================================

# Create comprehensive visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('MovieLens 32M - Data Exploration Dashboard', fontsize=16, fontweight='bold')

# 1. Rating Distribution
ax1 = axes[0, 0]
rating_counts = ratings_sample['rating'].value_counts().sort_index()
bars = ax1.bar(rating_counts.index, rating_counts.values, color='steelblue', edgecolor='black', alpha=0.7)
ax1.set_title('Distribution of Ratings', fontsize=13, fontweight='bold')
ax1.set_xlabel('Rating', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=9)

# 2. Ratings per User Distribution
ax2 = axes[0, 1]
ratings_per_user = ratings_sample.groupby('userId').size()
ax2.hist(ratings_per_user, bins=30, color='coral', edgecolor='black', alpha=0.7)
ax2.set_title('Distribution of Ratings per User', fontsize=13, fontweight='bold')
ax2.set_xlabel('Number of Ratings', fontsize=11)
ax2.set_ylabel('Number of Users', fontsize=11)
ax2.axvline(ratings_per_user.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ratings_per_user.mean():.1f}')
ax2.axvline(ratings_per_user.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {ratings_per_user.median():.0f}')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Ratings per Movie Distribution
ax3 = axes[1, 0]
ratings_per_movie = ratings_sample.groupby('movieId').size()
ax3.hist(ratings_per_movie, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
ax3.set_title('Distribution of Ratings per Movie', fontsize=13, fontweight='bold')
ax3.set_xlabel('Number of Ratings', fontsize=11)
ax3.set_ylabel('Number of Movies', fontsize=11)
ax3.axvline(ratings_per_movie.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ratings_per_movie.mean():.1f}')
ax3.axvline(ratings_per_movie.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {ratings_per_movie.median():.0f}')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Genre Distribution (Top 15)
ax4 = axes[1, 1]
all_genres = movies['genres'].str.split('|').explode()
top_genres = all_genres.value_counts().head(15)
ax4.barh(range(len(top_genres)), top_genres.values, color='mediumpurple', edgecolor='black', alpha=0.7)
ax4.set_yticks(range(len(top_genres)))
ax4.set_yticklabels(top_genres.index, fontsize=10)
ax4.set_title('Top 15 Movie Genres', fontsize=13, fontweight='bold')
ax4.set_xlabel('Number of Movies', fontsize=11)
ax4.grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(top_genres.values):
    ax4.text(v, i, f' {v:,}', va='center', fontsize=9)

plt.tight_layout()
plt.show()

print("\n✅ Visualization 1 complete!")

# ============================================================
# CELL 12: Visualization 2 - Temporal Analysis
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Temporal Analysis', fontsize=16, fontweight='bold')

# 1. Ratings over years
ax1 = axes[0, 0]
yearly_counts = ratings_sample['year'].value_counts().sort_index()
ax1.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=6, color='steelblue')
ax1.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3, color='steelblue')
ax1.set_title('Ratings Over Time (by Year)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Number of Ratings', fontsize=11)
ax1.grid(alpha=0.3)

# 2. Ratings by hour of day
ax2 = axes[0, 1]
hourly_counts = ratings_sample['hour'].value_counts().sort_index()
ax2.bar(hourly_counts.index, hourly_counts.values, color='coral', edgecolor='black', alpha=0.7)
ax2.set_title('Ratings by Hour of Day', fontsize=13, fontweight='bold')
ax2.set_xlabel('Hour', fontsize=11)
ax2.set_ylabel('Number of Ratings', fontsize=11)
ax2.set_xticks(range(0, 24, 2))
ax2.grid(axis='y', alpha=0.3)

# 3. Average rating over time
ax3 = axes[1, 0]
yearly_avg_rating = ratings_sample.groupby('year')['rating'].mean()
ax3.plot(yearly_avg_rating.index, yearly_avg_rating.values, marker='s', linewidth=2, markersize=6, color='green')
ax3.set_title('Average Rating Over Time', fontsize=13, fontweight='bold')
ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Average Rating', fontsize=11)
ax3.set_ylim(0, 5)
ax3.grid(alpha=0.3)

# 4. Ratings by month
ax4 = axes[1, 1]
monthly_counts = ratings_sample['month'].value_counts().sort_index()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax4.bar(monthly_counts.index, monthly_counts.values, color='lightgreen', edgecolor='black', alpha=0.7)
ax4.set_title('Ratings by Month', fontsize=13, fontweight='bold')
ax4.set_xlabel('Month', fontsize=11)
ax4.set_ylabel('Number of Ratings', fontsize=11)
ax4.set_xticks(range(1, 13))
ax4.set_xticklabels(months, rotation=45)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Visualization 2 complete!")

# ============================================================
# CELL 13: Top Rated Movies Analysis
# ============================================================

# Movies with at least 50 ratings
min_ratings = 50
movie_stats = ratings_sample.groupby('movieId').agg({
    'rating': ['count', 'mean', 'std']
}).reset_index()
movie_stats.columns = ['movieId', 'count', 'mean_rating', 'std_rating']
movie_stats = movie_stats[movie_stats['count'] >= min_ratings]
movie_stats = movie_stats.merge(movies[['movieId', 'title', 'genres']], on='movieId')
movie_stats = movie_stats.sort_values('mean_rating', ascending=False)

print("\n" + "="*60)
print(f"TOP RATED MOVIES (min {min_ratings} ratings)")
print("="*60)

print(f"\n⭐ TOP 20 HIGHEST RATED MOVIES:")
for rank, row in enumerate(movie_stats.head(20).itertuples(), 1):
    print(f"\n{rank:2d}. {row.title}")
    print(f"    ⭐ Rating: {row.mean_rating:.2f}/5.0 (±{row.std_rating:.2f})")
    print(f"    📊 {row.count} ratings | 🎭 {row.genres}")

# ============================================================
# CELL 14: Summary Statistics
# ============================================================

print("\n\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)

summary = {
    'Total Ratings (Sample)': f"{len(ratings_sample):,}",
    'Unique Users': f"{ratings_sample['userId'].nunique():,}",
    'Unique Movies': f"{ratings_sample['movieId'].nunique():,}",
    'Total Movies in Database': f"{len(movies):,}",
    'Average Rating': f"{ratings_sample['rating'].mean():.2f}",
    'Most Common Rating': f"{ratings_sample['rating'].mode()[0]}",
    'Matrix Sparsity': f"{sparsity*100:.4f}%",
    'Avg Ratings per User': f"{ratings_per_user.mean():.1f}",
    'Avg Ratings per Movie': f"{ratings_per_movie.mean():.1f}",
    'Date Range': f"{ratings_sample['datetime'].min().date()} to {ratings_sample['datetime'].max().date()}"
}

for key, value in summary.items():
    print(f"   • {key:30s}: {value}")

print("\n" + "="*60)
print("✅ DATA EXPLORATION COMPLETE!")
print("="*60)

print("\n📌 KEY FINDINGS:")
print("   1. High matrix sparsity - need robust CF algorithms")
print("   2. Power law distribution - few movies get most ratings")
print("   3. Rating inflation - most ratings are 3.0 or higher")
print("   4. Temporal patterns exist - can be used for time-aware recommendations")

print("\n🎯 NEXT STEPS:")
print("   1. Implement data preprocessing pipeline")
print("   2. Create train-test split")
print("   3. Build collaborative filtering models")
print("   4. Develop evaluation framework")

print("\n💾 Save this notebook and commit to GitHub!")