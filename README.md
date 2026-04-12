# 🚗 Driver Drowsiness Detection System

## 📌 Overview

This project is an **AI-based Driver Drowsiness Detection System** that monitors a user’s eyes in real time using a webcam.

If the user keeps their eyes closed for more than **10 seconds**, the system automatically triggers an **alarm** to alert them.

To stop the alarm, the user can simply show a **👍 thumbs-up gesture**, making the system interactive and easy to use.

---

## 🎯 Problem Statement

Drowsiness while driving is one of the leading causes of road accidents.
Many existing systems are either expensive or not easily accessible.

👉 This project provides a **simple, low-cost, real-time solution** using computer vision.

---

## 🚀 Features

* 👁️ Real-time eye detection using webcam
* ⏱️ Detects drowsiness after 10 seconds of eye closure
* 🔊 Automatic alarm system
* 👍 Gesture-based alarm stop (thumbs-up)
* 🌐 Web-based interface (Streamlit)
* 💻 Easy to use for non-technical users

---

## 🧠 How It Works

1. Webcam captures live video
2. Face and eye landmarks are detected using **MediaPipe**
3. System tracks whether eyes are open or closed
4. If eyes are closed for **> 10 seconds**:

   * 🚨 Alarm starts automatically
5. User shows **thumbs-up gesture**:

   * 🔇 Alarm stops

---

## 🛠️ Technologies Used

* **Python**
* **OpenCV** – Image processing
* **MediaPipe** – Face & hand landmark detection
* **Streamlit** – Web application
* **Streamlit-WebRTC** – Real-time video streaming
* **NumPy** – Data processing

---

## 📂 Project Structure

```bash
Drowsiness_Detection/
│── app.py                # Main Streamlit app
│── drowsy_detect.py      # Local detection script
│── requirements.txt      # Dependencies
│── Dockerfile            # For deployment (Render)
│── alarm.mp3             # Alarm sound
│── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup (Local)

### 1. Clone the repository

```bash
git clone https://github.com/sakshirothe/drowsiness-detection-streamlit.git
cd drowsiness-detection-streamlit
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```
## ⚠️ Limitations

* Requires good lighting for accurate detection
* Camera permissions must be enabled
* WebRTC apps may behave differently across browsers
* Not a replacement for real automotive safety systems

---

## 🔮 Future Improvements

* Mobile app integration
* Night vision support
* More gestures for control
* AI model with higher accuracy
* Integration with car systems

---

