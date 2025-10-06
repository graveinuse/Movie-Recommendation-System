"""
Convert MovieLens 32M dataset to optimized format
Run this after downloading the ml-32m.zip file (239 MB)
"""

import pandas as pd
import os
import time

def convert_movielens_32m():
    """Convert MovieLens 32M data files to CSV format"""
    
    print("="*60)
    print("MovieLens 32M Dataset Conversion")
    print("="*60)
    
    start_time = time.time()
    
    # Check if ml-32m folder exists
    if not os.path.exists('ml-32m'):
        print("\n❌ ERROR: ml-32m folder not found!")
        print("\nPlease follow these steps:")
        print("1. Download ml-32m.zip from: https://grouplens.org/datasets/movielens/32m/")
        print("2. Extract the zip file in this project's root directory")
        print("3. Make sure you have a folder named 'ml-32m' with CSV files inside")
        return
    
    # Create output directory
    os.makedirs('data/raw', exist_ok=True)
    
    print("\n📊 Dataset Information:")
    print("   - 32 million ratings")
    print("   - 87,585 movies")
    print("   - 200,948 users")
    print("   - 2 million tags")
    
    # ============================================
    # 1. Convert Ratings (Large file - ~32M rows)
    # ============================================
    print("\n" + "="*60)
    print("1. Converting ratings.csv (This may take 2-3 minutes...)")
    print("="*60)
    
    try:
        # Read in chunks for memory efficiency
        print("   📂 Reading ratings data...")
        ratings = pd.read_csv('ml-32m/ratings.csv')
        
        print(f"   ✓ Loaded {len(ratings):,} ratings")
        print(f"   ✓ Unique users: {ratings['userId'].nunique():,}")
        print(f"   ✓ Unique movies: {ratings['movieId'].nunique():,}")
        print(f"   ✓ Memory usage: {ratings.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Save to processed folder
        print("   💾 Saving ratings.csv...")
        ratings.to_csv('data/raw/ratings.csv', index=False)
        print("   ✓ Saved to data/raw/ratings.csv")
        
        # Show sample
        print("\n   Sample ratings:")
        print(ratings.head(3).to_string(index=False))
        
        # Basic statistics
        print(f"\n   📊 Rating Statistics:")
        print(f"      - Average rating: {ratings['rating'].mean():.2f}")
        print(f"      - Rating range: {ratings['rating'].min()} to {ratings['rating'].max()}")
        print(f"      - Most common rating: {ratings['rating'].mode()[0]}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # ============================================
    # 2. Convert Movies
    # ============================================
    print("\n" + "="*60)
    print("2. Converting movies.csv")
    print("="*60)
    
    try:
        print("   📂 Reading movies data...")
        movies = pd.read_csv('ml-32m/movies.csv')
        
        print(f"   ✓ Loaded {len(movies):,} movies")
        
        # Save
        movies.to_csv('data/raw/movies.csv', index=False)
        print("   ✓ Saved to data/raw/movies.csv")
        
        # Show sample
        print("\n   Sample movies:")
        print(movies.head(3).to_string(index=False))
        
        # Genre analysis
        print(f"\n   📊 Movie Statistics:")
        all_genres = movies['genres'].str.split('|').explode()
        top_genres = all_genres.value_counts().head(5)
        print("      Top 5 genres:")
        for genre, count in top_genres.items():
            print(f"        - {genre}: {count:,} movies")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # ============================================
    # 3. Convert Tags
    # ============================================
    print("\n" + "="*60)
    print("3. Converting tags.csv")
    print("="*60)
    
    try:
        print("   📂 Reading tags data...")
        tags = pd.read_csv('ml-32m/tags.csv')
        
        print(f"   ✓ Loaded {len(tags):,} tags")
        
        # Save
        tags.to_csv('data/raw/tags.csv', index=False)
        print("   ✓ Saved to data/raw/tags.csv")
        
        # Show sample
        print("\n   Sample tags:")
        print(tags.head(3).to_string(index=False))
        
    except Exception as e:
        print(f"   ⚠️  Warning: Could not load tags.csv")
    
    # ============================================
    # 4. Convert Links (IMDb/TMDb IDs)
    # ============================================
    print("\n" + "="*60)
    print("4. Converting links.csv")
    print("="*60)
    
    try:
        print("   📂 Reading links data...")
        links = pd.read_csv('ml-32m/links.csv')
        
        print(f"   ✓ Loaded {len(links):,} links")
        
        # Save
        links.to_csv('data/raw/links.csv', index=False)
        print("   ✓ Saved to data/raw/links.csv")
        
    except Exception as e:
        print(f"   ⚠️  Warning: Could not load links.csv")
    
    # ============================================
    # Summary
    # ============================================
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    
    print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds")
    
    print("\n📁 Files created in data/raw/:")
    print("   ✓ ratings.csv")
    print("   ✓ movies.csv")
    print("   ✓ tags.csv")
    print("   ✓ links.csv")
    
    print("\n📊 Dataset Overview:")
    print(f"   • Total ratings: {len(ratings):,}")
    print(f"   • Total movies: {len(movies):,}")
    print(f"   • Total users: {ratings['userId'].nunique():,}")
    print(f"   • Total tags: {len(tags):,}")
    print(f"   • Sparsity: {(1 - len(ratings)/(ratings['userId'].nunique() * ratings['movieId'].nunique()))*100:.4f}%")
    
    print("\n📈 Rating Distribution:")
    rating_dist = ratings['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        bar_length = int(count / len(ratings) * 50)
        bar = '█' * bar_length
        print(f"   {rating}: {bar} {count:,} ({count/len(ratings)*100:.1f}%)")
    
    print("\n🎯 Next Steps:")
    print("   1. Run: jupyter notebook")
    print("   2. Open: notebooks/01_data_exploration.ipynb")
    print("   3. Start exploring your data!")
    
    print("\n" + "="*60)

def create_sample_dataset(sample_size=100000):
    """
    Create a smaller sample dataset for development/testing
    This is useful for faster iteration during development
    """
    print("\n" + "="*60)
    print("Creating Sample Dataset for Development")
    print("="*60)
    
    try:
        ratings = pd.read_csv('data/raw/ratings.csv')
        
        # Sample users
        sample_users = ratings['userId'].unique()[:1000]  # First 1000 users
        sample_ratings = ratings[ratings['userId'].isin(sample_users)]
        
        # Take only first N ratings
        sample_ratings = sample_ratings.head(sample_size)
        
        # Save sample
        os.makedirs('data/sample', exist_ok=True)
        sample_ratings.to_csv('data/sample/ratings_sample.csv', index=False)
        
        print(f"\n✅ Created sample dataset:")
        print(f"   • {len(sample_ratings):,} ratings")
        print(f"   • {sample_ratings['userId'].nunique():,} users")
        print(f"   • {sample_ratings['movieId'].nunique():,} movies")
        print(f"   • Saved to: data/sample/ratings_sample.csv")
        print("\n💡 Use this for faster development and testing!")
        
    except Exception as e:
        print(f"❌ Error creating sample: {e}")

if __name__ == "__main__":
    # Convert full dataset
    convert_movielens_32m()
    
    # Ask if user wants to create a sample
    print("\n" + "="*60)
    response = input("\n🤔 Would you like to create a sample dataset for faster development? (y/n): ")
    if response.lower() == 'y':
        create_sample_dataset()