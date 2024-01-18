import streamlit as st
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

# Load model architecture from JSON file
with open("cat-dog-app/model_architecture.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

# Load model weights from HDF5 file
loaded_model.load_weights("cat-dog-app/model_weights.h5")

# Streamlit app
st.title('MNIST Digit Recognition App')

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image for model prediction
    img = image.load_img(uploaded_file, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values to be between 0 and 1
    img_array = 1 - img_array  # Invert the image (assuming white background)

    # Make prediction using the loaded model
    prediction = loaded_model.predict(img_array)

    # Display the image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Display the prediction
    digit = np.argmax(prediction)
    st.write(f"Predicted Digit: {digit}")
