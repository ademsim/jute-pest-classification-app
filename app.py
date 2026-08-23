import os
# Set Keras backend to PyTorch 
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import gdown
import keras
import numpy as np
from PIL import Image

# Google Drive file ID
FILE_ID = '1bTgQRz5OmhTg6UJlndV6mSfb8mOB1h7s'
MODEL_PATH = 'Jute_Pest_TL_Model.keras'

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            
        with st.spinner("Downloading model from Google Drive, please wait..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
            
    return keras.models.load_model(MODEL_PATH)

# Load the model
with st.spinner("Preparing the model, please wait..."):
    model = load_my_model()

# Jute Pest Class Names
class_labels = sorted(['Beet Armyworm', 'Black Hairy', 'Cutworm', 'Field Cricket', 'Jute Aphid', 'Jute Hairy', 'Jute Red Mite', 'Jute Semilooper', 'Jute Stem Girdler', 'Jute Stem Weevil', 'Leaf Beetle', 'Mealybug', 'Pod Borer', 'Scopula Emissaria', 'Termite', 'Termite odontotermes (Rambur)', 'Yellow Mite'])

# User Interface
st.title("Jute Pest Classification")
st.write("Please upload an image of a jute pest to classify.")

# File uploader component
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Classify Image"):
        with st.spinner("Model is analyzing the image..."):
            try:
                # Convert image to RGB format 
                image = image.convert("RGB")
                         
                # Resize image to match model's expected input shape (224x224)
                img = image.resize((224, 224), Image.Resampling.BILINEAR)
                
                # MobileNetV2 normalization
                img_array = np.array(img, dtype=np.float32)
                                                            
                # Add batch dimension to match expected shape: (1, 224, 224, 3)
                img_array = np.expand_dims(img_array, axis=0) 
                
                # Perform prediction
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0])) * 100
                
                predicted_label = class_labels[predicted_class_idx]
                
                st.success("Analysis Complete!")
                
                # Display the result
                st.info(f"Result: **{predicted_label}** - Confidence: {confidence:.2f}%")
                
                # Expandable view for all class probabilities
                with st.expander("See all class probabilities"):
                    for idx, label in enumerate(class_labels):
                        st.write(f"{label}: {predictions[0][idx]*100:.2f}%")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
