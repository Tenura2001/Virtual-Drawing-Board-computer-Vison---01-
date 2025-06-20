# ✋ Virtual Drawing Board with AI Integration

An interactive hand-tracking project using computer vision and AI to draw on a virtual canvas and generate intelligent responses. This system allows users to draw with hand gestures and receive AI-generated answers based on the drawing — whether it's identifying a sketch or solving a handwritten math problem.

---

## 🎯 Project Features

* ✅ Real-time hand tracking using **cvzone** and **MediaPipe**
* ✅ Virtual drawing with **index finger gesture**
* ✅ Canvas clearing with **open palm gesture**
* ✅ AI-powered recognition using **Gemini (Google Generative AI)**
* ✅ Task switch between:

  * "Guess the drawing" 🖼️
  * "Solve the math problem" ➗
* ✅ Stylish interactive UI using **Streamlit** with custom dark theme

---

## 🧠 How It Works

| Gesture              | Action                                |
| -------------------- | ------------------------------------- |
| ☝️ Index Finger Up   | Draws on canvas                       |
| ✋ All Fingers Up     | Clears the canvas                     |
| 🤙 Pinky Finger Only | Triggers AI model to interpret canvas |

1. Uses your webcam to track your hand.
2. Detects finger patterns to perform actions.
3. Sends the drawing as an image to Gemini AI.
4. Displays the AI’s interpretation on-screen.

---


## 🚀 Technologies Used

* Python
* OpenCV
* cvzone / MediaPipe
* Streamlit
* PIL
* NumPy
* Google Generative AI (Gemini)

---

## 📦 Installation Instructions

```bash
# Clone the repository
https://github.com/yourusername/virtual-drawing-board.git
cd virtual-drawing-board

# Install dependencies
pip install -r requirements.txt

# Start the Streamlit app
streamlit run app.py
```

---

## 🧪 File Structure

```bash
virtual-drawing-board/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Required Python packages
├── assets/              # Images / gifs for README display
└── ...
```

---

## ⚙️ Configuration

To use Google Gemini AI, configure your API key:

```python
genai.configure(api_key="your-api-key-here")
```

> 🔐 **Never commit your API key** to version control.

---

## 🧪 Example Tasks

| Task               | Action                          |
| ------------------ | ------------------------------- |
| Draw a cat         | Select "Guess the drawing"      |
| Write "3 + 4 \* 5" | Select "Solve the math problem" |

---

## 🙌 Acknowledgements

* [cvzone](https://github.com/cvzone) by Murtaza Hassan
* [Google Generative AI](https://ai.google.dev/)
* [Streamlit](https://streamlit.io/)

---

## 📜 License

MIT License. You’re free to use and modify this project with attribution.

---

## 💡 Future Improvements

* Add hand gesture calibration
* Save drawings as PNG
* Extend gesture vocabulary
* Support drawing multiple colors

---

## ✍️ Author

**Tenura Pinsara** 

> Inspired by creativity and powered by AI.

