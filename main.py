import os
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

# Streamlit app configuration
st.set_page_config(
    page_title="Interactive AI Drawing & Math Solver",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    body {
        background-color: #1E1E2F; 
        color: #E1E1E6; 
    }
    .stApp {
        background-color: #1E1E2F;
    }
    .big-title {
        font-size: 48px; 
        color: #00FFDD; 
        text-align: center; 
        font-weight: bold;
    }
    .small-title {
        font-size: 24px; 
        color: #00FFDD; 
        text-align: center; 
    }
    .sidebar .sidebar-content {
        background-color: #2C2C3A;
    }
    .stRadio > div {
        background-color: #2C2C3A; 
        border: 1px solid #3E3E4E; 
        border-radius: 10px;
        color: #00FFDD;
    }
    .stCheckbox {
        color: #00FFDD;
    }
    .stButton button {
        background-color: #00FFDD; 
        color: #1E1E2F; 
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar options
with st.sidebar:
    st.title("Settings")
    run = st.checkbox("Run Webcam", value=True)
    task = st.radio("Select Task", ["Guess the drawing", "Solve the math problem"])

# Main UI layout
st.markdown('<div class="big-title">Virtual Drawing</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    FRAME_WINDOW = st.image([])
with col2:
    st.markdown('<div class="small-title">Answer</div>', unsafe_allow_html=True)
    output_text_area = st.empty()

# Configure AI model
genai.configure(api_key="type api key here")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize the webcam
cap = cv2.VideoCapture(1)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Hand Detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


def getHandsInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]
        fingers = detector.fingersUp(hand1)
        return fingers, lmList
    else:
        return None


def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, prev_pos, current_pos, color=(0, 255, 0), thickness=10)
        prev_pos = current_pos
    elif fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(canvas)
    return current_pos, canvas


def sendToAI(model, canvas, fingers, task):
    if fingers == [0, 0, 0, 0, 1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content([task, pil_image])
        return response.text


prev_pos = None
canvas = None
output_text = ""

# Main loop for webcam processing
while run:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandsInfo(img)
    if info:
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, info[0], task)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.markdown(f"**{output_text}**")
    cv2.waitKey(1)

