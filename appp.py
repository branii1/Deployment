import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Configure the page
st.set_page_config(
    page_title="Netflix Type Classifier",
    page_icon="🎬",
    layout="centered"
)

# Title and description
st.title("🎬 Netflix Content Type Classifier")
st.markdown("""
This app predicts whether a Netflix title is a **Movie** or **TV Show** based on its attributes.
Please fill in the details below and click 'Predict' to see the result.
""")

@st.cache_resource
def load_model():
    """
    Load the trained model with error handling and diagnostics
    """
    try:
        # Try to load the model using joblib
        loaded_object = joblib.load('netflix_type_rf_model.pkl')
        
        # Diagnostic: Check what type of object we loaded
        st.write(f"🔍 **Debug Info**: Loaded object type: {type(loaded_object)}")
        
        # If it's a dictionary, try to extract the actual model
        if isinstance(loaded_object, dict):
            st.write(f"🔍 **Debug Info**: Dictionary keys: {list(loaded_object.keys())}")
            
            # Common keys where the model might be stored
            possible_model_keys = ['model', 'classifier', 'pipeline', 'estimator', 'clf', 'rf_model']
            
            for key in possible_model_keys:
                if key in loaded_object:
                    model = loaded_object[key]
                    st.write(f"✅ Found model with key: '{key}', type: {type(model)}")
                    if hasattr(model, 'predict'):
                        st.success("✅ Model loaded successfully from dictionary!")
                        return model
            
            # If no common key found, try to find any object with predict method
            for key, value in loaded_object.items():
                if hasattr(value, 'predict'):
                    model = value
                    st.write(f"✅ Found model with key: '{key}', type: {type(model)}")
                    st.success("✅ Model loaded successfully from dictionary!")
                    return model
            
            st.error("❌ Could not find a model object in the dictionary")
            return None
            
        # If it's already a model-like object
        elif hasattr(loaded_object, 'predict'):
            st.success("✅ Model loaded successfully!")
            return loaded_object
            
        else:
            st.error(f"❌ Loaded object is not a model (type: {type(loaded_object)})")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("""
        💡 **Troubleshooting Tips:**
        - Ensure 'netflix_type_rf_model.pkl' is in the same directory
        - Check if the file is corrupted
        - Verify all required dependencies are installed
        """)
        return None

def get_input_features():
    """
    Collect user inputs for all required features
    """
    st.header("📊 Input Features")
    
    # Numerical features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        release_year = st.number_input(
            "Release Year",
            min_value=1900,
            max_value=2025,
            value=2020,
            help="Year the content was released"
        )
    
    with col2:
        duration_int = st.number_input(
            "Duration",
            min_value=1,
            max_value=500,
            value=90,
            help="Duration in minutes for movies, seasons for TV shows"
        )
    
    with col3:
        cast_count = st.number_input(
            "Cast Count",
            min_value=0,
            max_value=200,
            value=10,
            help="Number of cast members listed"
        )
    
    # Categorical features with representative categories
    st.subheader("Categorical Features")
    
    # Rating categories (based on common Netflix ratings)
    rating_options = [
        "G", "PG", "PG-13", "R", "NC-17", "TV-Y", "TV-Y7", "TV-Y7-FV", 
        "TV-G", "TV-PG", "TV-14", "TV-MA", "NR", "UR", "Not Specified"
    ]
    
    # Country options (representative list)
    country_options = [
        "United States", "United Kingdom", "Canada", "India", "Japan", 
        "South Korea", "France", "Germany", "Spain", "Australia",
        "Brazil", "Mexico", "Italy", "China", "Not Specified"
    ]
    
    # Director options (representative list)
    director_options = [
        "Martin Scorsese", "Steven Spielberg", "Christopher Nolan",
        "Quentin Tarantino", "Alfred Hitchcock", "Not Specified",
        "Other", "Rajiv Chilaka", "Cathy Garcia-Molina", "Suhas Kadav"
    ]
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        rating = st.selectbox("Rating", rating_options)
    
    with col5:
        country = st.selectbox("Country", country_options)
    
    with col6:
        director_top = st.selectbox("Primary Director", director_options)
    
    # Create feature dictionary
    features = {
        'release_year': release_year,
        'duration_int': duration_int,
        'cast_count': cast_count,
        'rating': rating,
        'country': country,
        'director_top': director_top
    }
    
    return features

def make_prediction(model, features):
    """
    Make prediction using the loaded model
    """
    try:
        # Convert features to DataFrame with correct column order
        feature_df = pd.DataFrame([features])
        
        # Debug: Show what we're passing to the model
        st.write(f"🔍 **Debug Info**: Features being sent to model:")
        st.write(feature_df)
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        
        # Map prediction to human-readable labels
        # Assuming: 0 = Movie, 1 = TV Show
        prediction_map = {0: "Movie", 1: "TV Show"}
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feature_df)[0]
            confidence = max(proba) * 100
        else:
            confidence = None
            
        return prediction_map.get(prediction, "Unknown"), confidence
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.write(f"🔍 **Debug Info**: Model type: {type(model)}")
        st.write(f"🔍 **Debug Info**: Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        return None, None

def main():
    """
    Main application function
    """
    # Load model
    model = load_model()
    
    if model is None:
        st.error("🚫 Cannot proceed without a valid model. Please check the model file.")
        st.stop()
    
    # Show model info for debugging
    with st.expander("🔧 Model Information (Debug)"):
        st.write(f"Model type: {type(model)}")
        if hasattr(model, 'steps'):
            st.write(f"Pipeline steps: {[step[0] for step in model.steps]}")
    
    # Get user inputs
    features = get_input_features()
    
    # Prediction button
    st.header("🎯 Prediction")
    
    if st.button("Predict Content Type", type="primary"):
        with st.spinner("Analyzing content..."):
            prediction, confidence = make_prediction(model, features)
            
            if prediction:
                # Display results
                st.success(f"**Prediction: {prediction}**")
                
                if confidence:
                    st.info(f"**Confidence: {confidence:.1f}%**")
                
                # Show interpretation
                if prediction == "Movie":
                    st.balloons()
                    st.markdown("""
                    🎥 **This appears to be a Movie!**
                    - Typically single, self-contained stories
                    - Usually 90-180 minutes in duration
                    - One-time viewing experience
                    """)
                else:
                    st.markdown("""
                    📺 **This appears to be a TV Show!**
                    - Episodic content with multiple episodes/seasons
                    - Ongoing narrative or standalone episodes
                    - Designed for serialized viewing
                    """)
                
                # Show feature summary
                with st.expander("View Input Summary"):
                    st.json(features)

    # Footer
    st.markdown("---")
    st.markdown(
        "🔍 *Note: This is a machine learning model prediction. " +
        "For production use, ensure categorical options match the original training data.*"
    )

if __name__ == "__main__":
    main()
