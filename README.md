# 🎬 Movie Recommendation System

A comprehensive movie recommendation system implementing multiple state-of-the-art approaches using the MovieLens 32M dataset.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

---

## 📊 Dataset

**MovieLens 32M Dataset**
- 32 million ratings
- 87,585 movies
- 200,948 users
- 2 million tag applications
- Released: May 2024
- Source: [GroupLens](https://grouplens.org/datasets/movielens/32m/)

This stable benchmark dataset provides a robust foundation for building and evaluating recommendation algorithms.

---

## 🎯 Project Overview

This project implements and compares multiple recommendation approaches:

### 1. **Collaborative Filtering**
   - **User-Based Filtering**: Recommends based on similar users' preferences
   - **Item-Based Filtering**: Recommends similar items to what users liked
   - **Matrix Factorization (SVD)**: Decomposes user-item interaction matrix

### 2. **Content-Based Filtering**
   - Utilizes movie metadata (genres, tags)
   - TF-IDF vectorization for feature extraction
   - Cosine similarity for movie recommendations

### 3. **Hybrid Model**
   - Combines collaborative and content-based approaches
   - Optimized weighted ensemble
   - Handles cold-start problem effectively

---

## 🛠️ Technologies Used

- **Python 3.11+**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Framework**: Streamlit
- **Development**: Jupyter Notebook

---

## 📁 Project Structure

```
movie-recommendation-system/
│
├── data/
│   ├── raw/                    # Original MovieLens 32M data
│   │   ├── ratings.csv         # User ratings (32M rows)
│   │   ├── movies.csv          # Movie metadata
│   │   ├── tags.csv            # User-generated tags
│   │   └── links.csv           # IMDb/TMDb identifiers
│   └── processed/              # Cleaned and preprocessed data
│       ├── train.csv
│       ├── test.csv
│       └── movie_features.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   ├── 03_content_based_filtering.ipynb
│   └── 04_hybrid_model.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── collaborative_recommender.py
│   ├── content_based_recommender.py
│   ├── hybrid_recommender.py
│   └── evaluation.py
│
├── models/
│   └── saved_models/           # Trained model files
│
├── app/
│   └── streamlit_app.py        # Web application
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- 8GB+ RAM (recommended for handling large dataset)
- 2GB free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Visit: https://grouplens.org/datasets/movielens/32m/
   - Download `ml-32m.zip` (239 MB)
   - Extract to project root
   - Run data conversion script:
   ```bash
   python convert_data.py
   ```

---

## 💻 Usage

### 1. Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Train Models
```python
from src.collaborative_recommender import CollaborativeRecommender

# Initialize and train
recommender = CollaborativeRecommender(ratings_df)
recommender.train_svd()
recommendations = recommender.get_top_n_recommendations(user_id=1, n=10)
```

### 3. Run Web Application
```bash
streamlit run app/streamlit_app.py
```

Access the app at: `http://localhost:8501`

---

## 📈 Model Performance

### Evaluation Metrics

| Model | RMSE | MAE | Precision@10 | Recall@10 | Coverage |
|-------|------|-----|--------------|-----------|----------|
| User-Based CF | TBD | TBD | TBD | TBD | TBD |
| Item-Based CF | TBD | TBD | TBD | TBD | TBD |
| SVD | TBD | TBD | TBD | TBD | TBD |
| Content-Based | - | - | TBD | TBD | TBD |
| Hybrid Model | TBD | TBD | TBD | TBD | TBD |

*Metrics will be updated as models are trained and evaluated.*

### Key Findings
- 📊 **Best Overall Model**: TBD
- ⚡ **Fastest Prediction**: TBD
- 🎯 **Best for Cold Start**: Content-Based
- 🔄 **Most Diverse**: TBD

---

## 🎨 Features

### Current Features
- ✅ Data preprocessing and cleaning
- ✅ Exploratory data analysis with visualizations
- ✅ Multiple recommendation algorithms
- ✅ Model evaluation and comparison
- ✅ Interactive web interface

### Planned Features
- 🔄 Deep learning-based recommendations (Neural Collaborative Filtering)
- 🔄 Real-time recommendation updates
- 🔄 User authentication and personalization
- 🔄 A/B testing framework
- 🔄 Recommendation explanations
- 🔄 Multi-criteria recommendations

---

## 📊 Sample Results

### Top 10 Recommended Movies for User #1

| Rank | Movie Title | Predicted Rating | Genres |
|------|-------------|------------------|--------|
| 1 | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD |
| ... | ... | ... | ... |

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_collaborative.py
```

---

## 📖 Documentation

Detailed documentation for each module:

- [Data Preprocessing](docs/data_preprocessing.md)
- [Collaborative Filtering](docs/collaborative_filtering.md)
- [Content-Based Filtering](docs/content_based_filtering.md)
- [Hybrid Model](docs/hybrid_model.md)
- [Evaluation Metrics](docs/evaluation.md)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Project Status

### Completed ✅
- [x] Project setup and structure
- [x] Data collection and preprocessing
- [ ] Exploratory data analysis
- [ ] Collaborative filtering implementation
- [ ] Content-based filtering implementation
- [ ] Hybrid model development
- [ ] Model evaluation and comparison
- [ ] Web application development
- [ ] Deployment

### Current Sprint
Working on: Data exploration and visualization

### Next Steps
- Implement collaborative filtering algorithms
- Build evaluation framework
- Create interactive visualizations

---

## 🔬 Research & References

This project implements concepts from:
- Matrix Factorization Techniques for Recommender Systems (Koren et al., 2009)
- Item-Based Collaborative Filtering Recommendation Algorithms (Sarwar et al., 2001)
- Content-Based Recommendation Systems (Pazzani & Billsus, 2007)

### Related Papers
- [Netflix Prize](https://www.netflixprize.com/)
- [MovieLens Dataset Papers](https://grouplens.org/datasets/movielens/)


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **GroupLens Research** for providing the MovieLens dataset
- **F. Maxwell Harper and Joseph A. Konstan. 2015.** The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
- The open-source community for amazing tools and libraries

---

## ⚠️ Dataset Citation

```
F. Maxwell Harper and Joseph A. Konstan. 2015. 
The MovieLens Datasets: History and Context. 
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. 
https://doi.org/10.1145/2827872
```

---

## 📊 Performance Considerations

**Note**: Due to the large size of the 32M dataset (32 million ratings):
- Initial data loading may take 2-5 minutes
- Model training requires 8GB+ RAM
- Consider using data sampling for development/testing
- Full training recommended for production models

**Optimization Tips**:
- Use chunked data loading for memory efficiency
- Implement parallel processing where possible
- Cache computed similarities
- Consider using sparse matrices for large datasets

---


## 📈 Project Metrics

![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/movie-recommendation-system)
![GitHub issues](https://img.shields.io/github/issues/yourusername/movie-recommendation-system)
![GitHub stars](https://img.shields.io/github/stars/yourusername/movie-recommendation-system)



*Last Updated: October 2025*
*Project Status: Active Development*