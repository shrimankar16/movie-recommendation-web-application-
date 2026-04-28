# 🎬 CineMatch - Movie Recommendation Web Application

A beautiful, intelligent movie recommendation system that helps you discover films you'll love based on content similarity. Built with Streamlit and powered by machine learning algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **🎯 Content-Based Recommendations**: Get personalized movie suggestions based on plot, cast, director, genres, and keywords
- **🎨 Modern UI/UX**: Sleek, dark-themed interface with smooth animations and responsive design
- **⚡ Fast & Efficient**: Cached data loading and similarity computations for instant results
- **📊 Rich Movie Information**: View genres, overviews, and similarity scores for each recommendation
- **🔍 Smart Search**: Easy-to-use dropdown search with all available movie titles
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Demo

The application features:
- A beautiful gradient hero section with the CineMatch branding
- Intuitive movie search with autocomplete
- Top 10 movie recommendations displayed in elegant cards
- Hover effects and smooth transitions for enhanced user experience

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **Backend**: Python 3.8+
- **Machine Learning**: scikit-learn (CountVectorizer, Cosine Similarity)
- **Data Processing**: Pandas, NumPy
- **HTTP Requests**: Requests library
- **Fonts**: Google Fonts (Playfair Display, DM Sans)

## 📋 Prerequisites

Before running this application, make sure you have:

- Python 3.8 or higher installed
- pip (Python package manager)
- Internet connection (for downloading the movie dataset)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shrimankar16/movie-recommendation-web-application-.git
   cd movie-recommendation-web-application-
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

1. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - The application will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

3. **Get recommendations**
   - Select a movie from the dropdown search box
   - Click the "Get Recommendations" button
   - Browse through 10 similar movie suggestions

## 📊 How It Works

### Content-Based Filtering Algorithm

1. **Data Collection**: The app fetches a movie dataset containing titles, genres, cast, directors, keywords, and overviews

2. **Feature Engineering**: 
   - Combines multiple features (keywords, cast, genres, director, overview) into a single "soup" text
   - Normalizes text (lowercase, whitespace handling)
   - Boosts director importance by tripling their weight

3. **Vectorization**: 
   - Uses CountVectorizer to convert text into numerical vectors
   - Removes English stop words
   - Limits to 12,000 features for optimal performance

4. **Similarity Computation**: 
   - Calculates cosine similarity between all movie vectors
   - Creates a similarity matrix for fast lookups

5. **Recommendation Generation**: 
   - Finds the selected movie's similarity scores
   - Returns top 10 most similar movies (excluding the input movie itself)

## 📁 Project Structure

```
movie-recommendation-web-application/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore rules
└── .env/                 # Environment variables (if needed)
```

## 🎨 UI Features

- **Custom Color Scheme**: Dark theme with gold and pink accents
- **Typography**: Elegant Playfair Display for headings, DM Sans for body text
- **Animations**: Smooth hover effects and transitions
- **Card Layout**: 2-column responsive grid for movie recommendations
- **Gradient Effects**: Eye-catching gradients for titles and buttons

## 📦 Dependencies

```
streamlit>=1.32.0      # Web application framework
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computations
scikit-learn>=1.3.0    # Machine learning algorithms
requests>=2.31.0       # HTTP requests for dataset
```

## 🔮 Future Enhancements

- [ ] Add movie posters using TMDB API
- [ ] Include user ratings and reviews
- [ ] Implement collaborative filtering
- [ ] Add filtering by genre, year, or rating
- [ ] Save favorite movies and recommendation history
- [ ] Export recommendations as PDF or share via link
- [ ] Add movie trailers and streaming availability
- [ ] Multi-language support

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Shrimankar**
- GitHub: [@shrimankar16](https://github.com/shrimankar16)

## 🙏 Acknowledgments

- Movie dataset from [rashida048's NLP Projects](https://github.com/rashida048/Some-NLP-Projects)
- Streamlit for the amazing web framework
- scikit-learn for machine learning tools
- Google Fonts for beautiful typography

## 📧 Contact

For questions, suggestions, or feedback, please open an issue on GitHub.

---

<div align="center">
  <strong>⭐ If you like this project, please give it a star! ⭐</strong>
</div>
