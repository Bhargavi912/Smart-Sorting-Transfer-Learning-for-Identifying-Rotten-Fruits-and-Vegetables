import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set the page configuration
st.set_page_config(page_title="Smart Sorting AI", layout="centered")

# Define the exact class names from your training data
CLASS_NAMES = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
IMG_SIZE = (224, 224)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('smart_sorting_model.keras')

st.title("🍎 Smart Sorting: Fresh vs Rotten AI")
st.write("Upload a photo of an apple, banana, or orange to analyze its freshness!")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader for the user
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("�� Analyzing...")
    
    # Preprocess the image for the model
    img_resized = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Create a batch of 1
    
    # Predict
    predictions = model.predict(img_array)[0]
    
    # Calculate overall Fresh vs Rotten percentages
    # Indices 0, 1, 2 are Fresh. Indices 3, 4, 5 are Rotten.
    fresh_prob = np.sum(predictions[0:3]) * 100
    rotten_prob = np.sum(predictions[3:6]) * 100
    
    # Find the specific identified fruit
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    
    st.subheader("📊 Analysis Results:")
    
    # Display percentages using columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🌿 Freshness Percentage", value=f"{fresh_prob:.2f}%")
    with col2:
        st.metric(label="🍂 Rotten Percentage", value=f"{rotten_prob:.2f}%")
        
    st.progress(int(fresh_prob))
    
    # Format the detected name to look nice (e.g., "freshapples" -> "Fresh Apples")
    clean_name = predicted_class.replace('fresh', 'Fresh ').replace('rotten', 'Rotten ').title()
    st.write(f"**Detected Item:** {clean_name}")
    
    # Provide a smart recommendation based on freshness
    if fresh_prob > 70:
        st.success("✅ This produce looks fresh and ready to eat or sell!")
    elif fresh_prob > 40:
        st.warning("⚠️ This produce is starting to turn. Consume it soon!")
    else:
        st.error("❌ This produce appears rotten. Please discard or compost it.")