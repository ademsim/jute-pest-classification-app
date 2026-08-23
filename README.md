# Jute Pest Classification & Detection App 🐛🌾

An end-to-end Deep Learning web application built to classify jute crop pests using **TensorFlow/Keras**, **MobileNetV2 (Transfer Learning)**, and **Streamlit**. 

This repository contains both the custom CNN / Transfer Learning training notebooks and the interactive web interface deployed for real-time pest identification.

---

## 🚀 Features
- **Transfer Learning Architecture:** Utilizes a pre-trained `MobileNetV2` backbone with fine-tuned classification layers for high accuracy and rapid inference.
- **Interactive Web UI:** Built with **Streamlit** allowing users to upload pest images and get instant predictions.
- **Confidence Scores & Probabilities:** Displays top predictions along with a breakdown of all class probabilities.
- **Cloud Model Integration:** Automatically downloads the trained `.keras` model weights from Google Drive on startup.

---

## 🛠️ Tech Stack
- **Deep Learning:** TensorFlow, Keras, PyTorch (Backend)
- **Web Framework:** Streamlit
- **Data Processing:** NumPy, Pillow
- **Deployment & Utilities:** gdown

---

## 📂 Project Structure
```text
jute-pest-classification/
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation

---

# ⚙️ Installation & Running Locally

To set up and run the application on your local machine, open your terminal and run the following commands block by block:

```bash
git clone [https://github.com/ADEM-SIMSEK/jute-pest-classification.git](https://github.com/ADEM-SIMSEK/jute-pest-classification.git)
cd jute-pest-classification
pip install -r requirements.txt
streamlit run app.py

---

## Author: Adem Şimşek
