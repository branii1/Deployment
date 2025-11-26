import streamlit as st
import pickle
import pandas as pd
import numpy as np
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

def load_model_fixed():
    """Load the trained model with compatibility fixes"""
    try:
        # First try standard loading
        with open('netflix_type_rf_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.warning(f"Standard loading failed: {e}. Trying compatibility mode...")
        try:
            # Try with custom fix for module issues
            import sys
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.ensemble import RandomForestClassifier
            
            # Add dummy classes to handle import issues
            class DummyPipeline:
                pass
            class DummyColumnTransformer:
                pass
            
            # Temporarily add dummy classes to handle import errors
            sys.modules['Pipeline'] = Pipeline
            sys.modules['ColumnTransformer'] = ColumnTransformer
            
            with open('netflix_type_rf_model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e2:
            st.error(f"Compatibility loading also failed: {e2}")
            return None

def load_model_robust():
    """More robust model loading with multiple fallbacks"""
    try:
        # Method 1: Try with encoding
        with open('netflix_type_rf_model.pkl', 'rb') as file:
            model = pickle.load(file, encoding='latin1')
        return model
    except Exception as e1:
        st.warning(f"Method 1 failed: {e1}")
        try:
            # Method 2: Try with different encoding
            with open('netflix_type_rf_model.pkl', 'rb') as file:
                model = pickle.load(file, encoding='bytes')
            return model
        except Exception as e2:
            st.warning(f"Method 2 failed: {e2}")
            try:
                # Method 3: Try with joblib instead of pickle
                import joblib
                model = joblib.load('netflix_type_rf_model.pkl')
                return model
            except Exception as e3:
                st.error(f"All loading methods failed: {e3}")
                return None

def user_input_features():
    """Collect user input features"""
    
    # Numerical features
    release_year = st.sidebar.slider('Release Year', 1920, 2023, 2020)
    duration_int = st.sidebar.number_input('Duration (minutes for movies, seasons for TV shows)', 
                                         min_value=1, max_value=500, value=90)
    cast_count = st.sidebar.slider('Number of Cast Members', 1, 50, 5)
    
    # Categorical features with options from the model's training data
    rating_options = ['66 min', '74 min', '84 min', 'G', 'NC-17', 'NR', 'Not Specified', 
                     'PG', 'PG-13', 'R', 'TV-14', 'TV-G', 'TV-MA', 'TV-PG', 'TV-Y', 
                     'TV-Y7', 'TV-Y7-FV', 'UR']
    
    country_options = ['United States', 'United Kingdom', 'India', 'Canada', 'France', 
                      'Japan', 'South Korea', 'Spain', 'Germany', 'Australia', 'Mexico',
                      'Brazil', 'Italy', 'China', 'Not Specified']
    
    director_options = ['Cathy Garcia-Molina', 'Jay Chapman', 'Jay Karas', 'Marcus Raboy', 
                       'Martin Scorsese', 'Not Specified', 'Other', 'Rajiv Chilaka', 
                       'Raúl Campos, Jan Suter', 'Suhas Kadav', 'Youssef Chahine']
    
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
    model = load_model_robust()
    
    if model is None:
        st.error("""
        Could not load the model. This is usually due to version compatibility issues.
        
        **Possible solutions:**
        1. Ensure you're using scikit-learn version 1.7.2 (the model was trained with this version)
        2. Try installing the exact versions: `pip install scikit-learn==1.7.2 joblib==1.3.2`
        3. The model file might be corrupted or incomplete
        """)
        
        # Show installation commands
        with st.expander("Installation Help"):
            st.code("""
pip install scikit-learn==1.7.2
pip install joblib==1.3.2
pip install streamlit pandas numpy
            """)
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
                
                # Assuming the model predicts 0 for Movie, 1 for TV Show
                content_types = ['Movie', 'TV Show']
                
                predicted_type = content_types[prediction[0]]
                
                # Display results
                st.success(f"**Predicted Content Type: {predicted_type}**")
                
                # Show probability scores
                st.subheader("Prediction Probabilities:")
                prob_df = pd.DataFrame({
                    'Content Type': content_types,
                    'Probability': prediction_proba[0]
                })
                st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))
                
                # Visualize probabilities
                st.subheader("Probability Distribution:")
                chart_data = pd.DataFrame({
                    'Probability': prediction_proba[0]
                }, index=content_types)
                st.bar_chart(chart_data)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("This might be due to feature name mismatches. Trying alternative approach...")
                
                # Alternative prediction approach
                try:
                    # Get the feature names the model expects
                    if hasattr(model, 'feature_names_in_'):
                        expected_features = model.feature_names_in_
                        st.write(f"Model expects features: {list(expected_features)}")
                    
                    # Try with different column ordering
                    st.info("Please ensure your input features match the model's expected features.")
                except Exception as e2:
                    st.error(f"Alternative approach also failed: {e2}")

    # Usage tips
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        - **Duration**: For movies, enter the duration in minutes. For TV shows, enter the number of seasons.
        - **Cast Count**: The number of main cast members listed.
        - **Rating**: Content rating/classification.
        - **Country**: Primary country of production.
        - **Director**: Main director (select 'Not Specified' if unknown).
        
        The model uses these features to determine whether the content is more likely to be a Movie or TV Show.
        """)

if __name__ == '__main__':
    main()
