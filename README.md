# 467Project

Music streaming services like Spotify rely on genre classification algorithms to recommend personalized playlists to their users to improve the user experience. Accurate genre prediction enables better playlist curation, discovery features, and recommendations. However, many current approaches struggle with genre-blending tracks, subjective boundaries, and emerging musical styles. 

In this project, we aim to develop a machine learning model that classifies songs into their respective genres using structured numerical attributes. Instead of working with raw waveform or spectrogram data, we use pre-extracted features such as danceability, energy, loudness, acousticness, instrumentalness, valence, tempo, key, and mode. These features provide a compact representation of each track’s musical character, allowing us to frame genre classification as a tabular data problem.

Accurate genre classification is crucial for improving recommendation systems in music streaming platforms. By automating genre prediction, our project seeks to enhance song recommendations, playlist curation, and genre-based music discovery to provide users with a more personalized and enjoyable listening experience.

We explored several approaches, including K-Nearest Neighbors, Multilayer Perceptrons, and XGBoost. Ultimately, XGBoost emerged as the best-performing model, effectively handling class imbalance and overlapping feature distributions.

## Dataset

For this project, we utilized an open-source dataset scraped from Spotify's API. This dataset has a list of songs, the playlist genre's they are associated with, as well as a list of numerical metrics to measure features of the song such as: ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'].

## Methods

Through the development process, we tested 4 main methods: 
- **K Neighbors Classifier**: baseline.ipynb
- **Basic MLP with one hidden layer**: genre.ipynb
- **A revised MLP with 4 hidden layers and domain-specific featurization**: training_pipeline.py
- **XGBoost**: xgboost.ipynb

## How to Run

### Jupyter Notebooks

Open the files (`.ipynb`) in a source code editor or Google Colab and execute each cell sequentially.

### Python Script

Install dependencies and run the training script:
```bash
pip install -r requirements.txt
python3 training_pipeline.py

