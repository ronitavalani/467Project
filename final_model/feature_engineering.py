import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def engineer_audio_features(df, target_col='playlist_genre'):
    """
    Creates advanced features for music classification from Spotify data
    
    Args:
        df: DataFrame with original Spotify features
        target_col: Name of the target column (genre)
        
    Returns:
        DataFrame with engineered features
    """
    # Make a copy to avoid modifying the original
    df_eng = df.copy()
    
    # 1. Basic ratio features (capturing relationships between audio attributes)
    df_eng['energy_valence_ratio'] = df_eng['energy'] / (df_eng['valence'] + 0.01)
    df_eng['danceability_acousticness_ratio'] = df_eng['danceability'] / (df_eng['acousticness'] + 0.01)
    df_eng['speechiness_instrumentalness_ratio'] = df_eng['speechiness'] / (df_eng['instrumentalness'] + 0.01)
    df_eng['liveness_acousticness_ratio'] = df_eng['liveness'] / (df_eng['acousticness'] + 0.01)
    
    # 2. Normalized features
    df_eng['loudness_normalized'] = (df_eng['loudness'] - df_eng['loudness'].min()) / (df_eng['loudness'].max() - df_eng['loudness'].min())
    df_eng['tempo_normalized'] = (df_eng['tempo'] - df_eng['tempo'].min()) / (df_eng['tempo'].max() - df_eng['tempo'].min())
    
    # 3. Interaction features
    df_eng['energy_danceability'] = df_eng['energy'] * df_eng['danceability']
    df_eng['valence_energy'] = df_eng['valence'] * df_eng['energy']
    df_eng['acousticness_instrumentalness'] = df_eng['acousticness'] * df_eng['instrumentalness']
    
    # 4. Squared features (capture non-linear relationships)
    for feature in ['danceability', 'energy', 'valence', 'tempo_normalized']:
        df_eng[f'{feature}_squared'] = df_eng[feature] ** 2
    
    # 5. Music theory informed features
    # Mode (major/minor) interaction with valence
    df_eng['mode_valence'] = df_eng['mode'] * df_eng['valence']
    
    # Key and mode interaction (musical context)
    df_eng['key_mode_interaction'] = df_eng['key'] + (df_eng['mode'] * 12)
    
    # Tempo and energy relationship (high tempo, high energy tracks vs low tempo, high energy)
    df_eng['tempo_energy_ratio'] = df_eng['tempo_normalized'] / (df_eng['energy'] + 0.01)
    
    # 6. Compute distances from genre centroids (if target is available for training)
    if target_col in df_eng.columns:
        # Select only numeric features for centroid calculation
        numeric_cols = df_eng.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != target_col and 'centroid_dist' not in col]
        
        # Calculate centroids for each genre
        genres = df_eng[target_col].unique()
        for genre in genres:
            genre_data = df_eng[df_eng[target_col] == genre][numeric_cols]
            centroid = genre_data.mean().values
            
            # Calculate Euclidean distance to this genre's centroid
            df_eng[f'centroid_dist_{genre}'] = np.sqrt(((df_eng[numeric_cols].values - centroid) ** 2).sum(axis=1))
    
    return df_eng

def select_best_features(X, y, k=25):
    """
    Select the k best features using ANOVA F-statistic
    
    Args:
        X: Features DataFrame
        y: Target array
        k: Number of features to select
        
    Returns:
        DataFrame with selected features
    """
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices]
    
    print(f"Selected {len(selected_features)} best features")
    print("Top 10 features by F-score:")
    scores = selector.scores_
    top_indices = np.argsort(scores)[::-1][:10]
    for idx in top_indices:
        print(f"{X.columns[idx]}: {scores[idx]:.4f}")
    
    return X[selected_features]

def apply_feature_engineering_pipeline(df, target_col='playlist_genre', use_pca=True, n_components=20):
    """
    Complete feature engineering pipeline including:
    - Creating engineered features
    - Handling outliers
    - Feature selection
    - Optional PCA
    
    Args:
        df: Original DataFrame
        target_col: Target column name
        use_pca: Whether to use PCA
        n_components: Number of PCA components if used
        
    Returns:
        Processed DataFrame ready for model training
    """
    print(f"Original data shape: {df.shape}")
    
    # 1. Engineer features
    df_eng = engineer_audio_features(df, target_col)
    print(f"After feature engineering: {df_eng.shape}")
    
    # 2. Handle outliers using IQR method
    numeric_cols = df_eng.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    for col in numeric_cols:
        Q1 = df_eng[col].quantile(0.25)
        Q3 = df_eng[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_eng[col] = df_eng[col].clip(lower_bound, upper_bound)
    
    # 3. Prepare features and target
    X = df_eng.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    y = df_eng[target_col]
    
    # 4. Feature selection
    X_selected = select_best_features(X, y, k=min(25, X.shape[1]))
    print(f"After feature selection: {X_selected.shape}")
    
    # 5. Optional PCA for dimensionality reduction
    if use_pca and X_selected.shape[1] > n_components:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_selected)
        
        # Convert back to DataFrame with new column names
        X_final = pd.DataFrame(
            X_pca, 
            columns=[f'pca_component_{i+1}' for i in range(n_components)]
        )
        
        # Print explained variance
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"After PCA: {X_final.shape}, Explained variance: {explained_var:.4f}")
        
        return X_final, y
    else:
        return X_selected, y