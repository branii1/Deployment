import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Netflix Type Predictor",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title("🎬 Netflix Content Type Predictor")
st.markdown("""
This app predicts whether a Netflix title is likely to be a **Movie** or **TV Show** 
based on its characteristics using a trained Random Forest model.
""")

# Sidebar for input features
st.sidebar.header("Input Features")

def load_model():
    """Load the trained model from the pickle file"""
    try:
        with open('netflix_type_rf_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def user_input_features():
    """Collect user input features"""
    
    # Numerical features
    release_year = st.sidebar.slider('Release Year', 1920, 2023, 2020)
    duration_int = st.sidebar.number_input('Duration (minutes for movies, seasons for TV shows)', 
                                         min_value=1, max_value=500, value=90)
    cast_count = st.sidebar.slider('Number of Cast Members', 1, 50, 5)
    
    # Categorical features with EXACT options from the model's training data
    # These are extracted from your actual model pickle file
    rating_options = ['66 min', '74 min', '84 min', 'G', 'NC-17', 'NR', 'Not Specified', 
                     'PG', 'PG-13', 'R', 'TV-14', 'TV-G', 'TV-MA', 'TV-PG', 'TV-Y', 
                     'TV-Y7', 'TV-Y7-FV', 'UR']
    
    country_options = [
        'United States', 'India', 'United Kingdom', 'Canada', 'France', 'Japan', 
        'South Korea', 'Spain', 'Mexico', 'Australia', 'Germany', 'Philippines',
        'Brazil', 'Italy', 'Turkey', 'Thailand', 'Indonesia', 'South Africa',
        'Egypt', 'Nigeria', 'China', 'Argentina', 'Colombia', 'Pakistan',
        'Bangladesh', 'Russia', 'Malaysia', 'Saudi Arabia', 'United Arab Emirates',
        'Poland', 'Netherlands', 'Sweden', 'Belgium', 'Denmark', 'Norway',
        'Switzerland', 'Austria', 'Israel', 'Czech Republic', 'Finland',
        'Portugal', 'Greece', 'Romania', 'Hungary', 'Ukraine', 'Ireland',
        'New Zealand', 'Singapore', 'Hong Kong', 'Taiwan', 'Vietnam',
        'Not Specified'
    ]
    
    director_options = [
        'Not Specified', 'Rajiv Chilaka', 'Marcus Raboy', 'Jay Karas', 
        'Suhas Kadav', 'Jay Chapman', 'Cathy Garcia-Molina', 'Youssef Chahine',
        'Martin Scorsese', 'Raúl Campos, Jan Suter', 'Other'
    ]
    
    rating = st.sidebar.selectbox('Rating', rating_options, index=rating_options.index('TV-MA'))
    country = st.sidebar.selectbox('Country', country_options, index=country_options.index('United States'))
    director_top = st.sidebar.selectbox('Director', director_options, index=director_options.index('Not Specified'))
    
    # Create input dataframe
    input_data = {
        'release_year': release_year,
        'duration_int': duration_int,
        'cast_count': cast_count,
        'rating': rating,
        'country': country,
        'director_top': director_top
    }
    
    features = pd.DataFrame(input_data, index=[0])
    return features

def main():
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Could not load the model. Please ensure the model file is available.")
        return
    
    # Display model information
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        **Model Type:** Random Forest Classifier
        **Features Used:**
        - Release Year
        - Duration (minutes/seasons)
        - Cast Count
        - Rating
        - Country
        - Director
        
        The model was trained on Netflix content data to classify titles as Movies or TV Shows.
        """)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Parameters")
        input_df = user_input_features()
        
        # Display input features
        st.subheader("Selected Features:")
        st.dataframe(input_df.style.format({
            'release_year': '{:.0f}',
            'duration_int': '{:.0f}',
            'cast_count': '{:.0f}'
        }))
    
    with col2:
        st.header("Prediction")
        
        # Make prediction when button is clicked
        if st.button('Predict Content Type', type='primary'):
            try:
                # Make prediction
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)
                
                # Display results
                if prediction[0] == 1:
                    predicted_type = "TV Show"
                    st.success(f"**Predicted Content Type: 🎭 {predicted_type}**")
                else:
                    predicted_type = "Movie"
                    st.success(f"**Predicted Content Type: 🎬 {predicted_type}**")
                
                # Show probability scores
                st.subheader("Prediction Probabilities:")
                prob_df = pd.DataFrame({
                    'Content Type': ['Movie', 'TV Show'],
                    'Probability': prediction_proba[0]
                })
                st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))
                
                # Visualize probabilities
                st.subheader("Probability Distribution:")
                chart_data = pd.DataFrame({
                    'Probability': prediction_proba[0]
                }, index=['Movie', 'TV Show'])
                st.bar_chart(chart_data)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        with st.expander("📊 Feature Importance"):
            try:
                # Get feature names from the preprocessing pipeline
                if hasattr(model, 'named_steps'):
                    preprocessor = model.named_steps.get('preprocessor', None)
                    if preprocessor is not None:
                        feature_names = preprocessor.get_feature_names_out()
                    else:
                        feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]
                else:
                    feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(importance_df.head(10))
                
                # Display as bar chart
                st.bar_chart(importance_df.set_index('Feature').head(10))
                
            except Exception as e:
                st.info("Feature importance visualization is not available for this model configuration.")

    # Usage tips
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        - **Duration**: For movies, enter the duration in minutes. For TV shows, enter the number of seasons.
        - **Cast Count**: The number of main cast members listed.
        - **Rating**: Content rating/classification. Note that some ratings like '66 min', '74 min', '84 min' are actually duration-based categories in the original data.
        - **Country**: Primary country of production.
        - **Director**: Main director (select 'Not Specified' if unknown).
        
        **Important Note**: The model uses the exact categories shown in the dropdowns. Using values outside these categories may cause prediction errors.
        """)

if __name__ == '__main__':
    main()
