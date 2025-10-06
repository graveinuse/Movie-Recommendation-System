# 🚀 Setup Guide - MovieLens 32M Dataset

## Prerequisites

⚠️ **Important System Requirements:**
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: 2GB free space
- **Python**: 3.11 or higher
- **Time**: Allow 10-15 minutes for complete setup

---

## Step-by-Step Setup

### 1. Download MovieLens 32M Dataset

1. Visit: https://grouplens.org/datasets/movielens/32m/
2. Click on **"ml-32m.zip"** (239 MB)
3. Save to your Downloads folder
4. Extract the ZIP file
5. Move the **`ml-32m`** folder to your project root

Your structure should look like:
```
movie-recommendation-system/
├── ml-32m/              ← This folder
│   ├── ratings.csv
│   ├── movies.csv
│   ├── tags.csv
│   └── links.csv
├── data/
├── notebooks/
└── ...
```

---

### 2. Verify Dataset Files

In PowerShell/Terminal:

```powershell
# Check if ml-32m folder exists
dir ml-32m

# You should see these files:
# - ratings.csv  (~800 MB)
# - movies.csv   (~3 MB)
# - tags.csv     (~40 MB)
# - links.csv    (~2 MB)
```

---

### 3. Run Data Conversion

```powershell
python convert_data.py
```

**What happens:**
- Reads all CSV files from `ml-32m/`
- Converts and optimizes the data
- Saves to `data/raw/` folder
- ⏱️ Takes 2-5 minutes (be patient!)

**Expected Output:**
```
====================================================
MovieLens 32M Dataset Conversion
====================================================

1. Converting ratings.csv...
   ✓ Loaded 32,000,000 ratings
   ✓ Unique users: 200,948
   ✓ Unique movies: 87,585
   
2. Converting movies.csv...
   ✓ Loaded 87,585 movies
   
3. Converting tags.csv...
   ✓ Loaded 2,000,000 tags
   
✅ CONVERSION COMPLETE!
```

---

### 4. Create Sample Dataset (Optional but Recommended)

For faster development and testing, create a smaller sample:

When prompted after conversion, type **`y`**:
```
Would you like to create a sample dataset? (y/n): y
```

This creates a 100K sample in `data/sample/` for quick testing.

---

## 📊 Understanding the Dataset

### ratings.csv
```
userId,movieId,rating,timestamp
1,307,3.5,1256677221
1,481,3.5,1256677456
1,1091,1.5,1256677471
```
- **32 million rows**
- Rating scale: 0.5 to 5.0 (in 0.5 increments)

### movies.csv
```
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
```
- **87,585 movies**
- Multiple genres per movie (pipe-separated)

### tags.csv
```
userId,movieId,tag,timestamp
3,260,classic,1474784669
3,260,sci-fi,1474784727
```
- **2 million tags**
- User-generated movie tags

### links.csv
```
movieId,imdbId,tmdbId
1,0114709,862
2,0113497,8844
```
- Links to IMDb and TMDb databases
- Useful for fetching posters/metadata

---

## 💾 Disk Space Breakdown

After conversion, your `data/` folder will contain:

```
data/
├── raw/
│   ├── ratings.csv     (~800 MB)  ← Main dataset
│   ├── movies.csv      (~3 MB)
│   ├── tags.csv        (~40 MB)
│   └── links.csv       (~2 MB)
└── sample/             (Optional)
    └── ratings_sample.csv (~10 MB)
```

**Total: ~850 MB**

---

## 🎯 Development Workflow

### For Development/Testing:
Use the **sample dataset** (faster, less memory):
```python
# In your notebooks
ratings = pd.read_csv('data/sample/ratings_sample.csv')
```

### For Final Training:
Use the **full dataset** (better accuracy):
```python
# For production models
ratings = pd.read_csv('data/raw/ratings.csv')
```

---

## ⚡ Performance Tips

### 1. Use Chunked Reading
For very large operations:
```python
chunk_size = 1000000  # 1M rows at a time
for chunk in pd.read_csv('data/raw/ratings.csv', chunksize=chunk_size):
    # Process chunk
    pass
```

### 2. Optimize Data Types
Reduce memory usage:
```python
ratings = pd.read_csv('data/raw/ratings.csv', dtype={
    'userId': 'int32',
    'movieId': 'int32',
    'rating': 'float16'
})
```

### 3. Use Sparse Matrices
For user-item matrix:
```python
from scipy.sparse import csr_matrix
# Create sparse matrix instead of dense
```

### 4. Sample During Development
Start with 1M ratings, then scale up:
```python
ratings = pd.read_csv('data/raw/ratings.csv', nrows=1000000)
```

---

## 🐛 Troubleshooting

### Problem: "MemoryError"
**Solution**: 
- Close other applications
- Use sample dataset
- Implement chunked reading
- Upgrade RAM if possible

### Problem: "File not found"
**Solution**:
- Verify `ml-32m` folder is in project root
- Check file names match exactly (case-sensitive)
- Re-extract the ZIP file

### Problem: Conversion takes too long
**Solution**:
- This is normal for 32M rows!
- First time takes 2-5 minutes
- Be patient and don't interrupt

### Problem: "Out of disk space"
**Solution**:
- Need at least 2GB free
- Clear temporary files
- Use external drive if needed

---

## ✅ Verification Checklist

After setup, verify everything works:

```powershell
# 1. Check files exist
dir data\raw

# 2. Test pandas can read
python -c "import pandas as pd; df = pd.read_csv('data/raw/ratings.csv', nrows=5); print(df)"

# 3. Check file sizes
dir data\raw | findstr csv

# 4. Verify row counts
python -c "import pandas as pd; print('Ratings:', len(pd.read_csv('data/raw/ratings.csv')))"
```

Expected output:
```
✓ ratings.csv exists (~800 MB)
✓ movies.csv exists (~3 MB)
✓ Can read with pandas
✓ Ratings: 32,000,000 rows
```

---

## 🎓 Next Steps

Once setup is complete:

1. **Start Jupyter Notebook**
   ```powershell
   jupyter notebook
   ```

2. **Create First Notebook**
   - Go to `notebooks/` folder
   - Create: `01_data_exploration.ipynb`
   - Start exploring the data!

3. **Load and Explore**
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   
   # Load sample for quick start
   ratings = pd.read_csv('../data/sample/ratings_sample.csv')
   movies = pd.read_csv('../data/raw/movies.csv')
   
   print(f"Ratings: {len(ratings):,}")
   print(f"Movies: {len(movies):,}")
   ```

---

## 📚 Additional Resources

### Dataset Documentation
- [MovieLens Website](https://grouplens.org/datasets/movielens/)
- [Dataset README](https://files.grouplens.org/datasets/movielens/ml-32m-README.html)

### Research Papers
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context

### Tutorials
- Check notebooks in `notebooks/` folder
- See example code in `src/` modules

---

## 💡 Pro Tips

1. **Always work with sample first** - Test your code on small data
2. **Save intermediate results** - Don't recompute everything
3. **Monitor memory usage** - Use `df.memory_usage()` 
4. **Use Git LFS** - For large files (if needed)
5. **Document your process** - Keep notes in notebooks

---

## 🆘 Need Help?

If you encounter issues:

1. Check this guide first
2. Look at error messages carefully
3. Google the specific error
4. Check GitHub issues
5. Ask for help with specific error details

---

**Ready to start? Run `python convert_data.py` and let's go! 🚀**