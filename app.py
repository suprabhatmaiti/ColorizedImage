import streamlit as st
import numpy as np
import cv2
import os

# Paths to load the model files (update the paths as needed)
DIR = r"C:/Users/Suprabhat/Downloads/Colorizing-black-and-white-images-using-Python-master/Colorizing-black-and-white-images-using-Python-master"
PROTOTXT = os.path.join(DIR, "model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "model/pts_in_hull.npy")
MODEL = os.path.join(DIR, "model/colorization_release_v2.caffemodel")

# Load the model
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)
    
    # Load centers for ab channel quantization used for rebalancing.
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    return net

# Colorize function
def colorize_image(image, net):
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    return colorized

# Streamlit app
st.title("Black & White Image Colorizer")
st.write("Upload a black & white image to colorize it!")

# Upload image
uploaded_file = st.file_uploader("Choose a black & white image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Load the model and colorize the image
    st.write("Colorizing the image...")
    net = load_model()
    colorized_image = colorize_image(image, net)
    
    # Display the original and colorized images
    st.image([image, colorized_image], caption=["Original", "Colorized"], channels="BGR")
