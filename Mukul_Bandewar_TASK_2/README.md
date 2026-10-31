# Handwritten Character Recognition

## About the Project

Handwritten Character Recognition is a Python-based AI/ML project that identifies handwritten digits and alphabets from images.

This project uses a Convolutional Neural Network (CNN) to recognize handwritten characters. Users can upload an image through a simple Streamlit web application and get the predicted character with its confidence score.

## Features

- Recognizes handwritten digits and alphabets.
- Uses CNN for character classification.
- Supports image uploading (JPG, JPEG, PNG).
- Displays predicted character and confidence score.
- Simple and user-friendly Streamlit interface.
- Uses the EMNIST dataset for model training.

## Technologies Used

- Python
- TensorFlow and Keras
- CNN (Convolutional Neural Network)
- EMNIST Dataset
- NumPy
- Pillow
- Streamlit

## Project Structure

```text
Handwritten-Character-Recognition/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── model/
    ├── character_model.keras
    └── classes.json
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Open the project folder**

```bash
cd Handwritten-Character-Recognition
```

**3. Install required libraries**

```bash
pip install -r requirements.txt
```

**4. Train the model**

```bash
python train_model.py
```

**5. Run the Streamlit application**

```bash
streamlit run app.py
```

## Dataset

This project uses the EMNIST Balanced dataset, which contains handwritten digits and letters.

Dataset: https://www.nist.gov/itl/products-and-services/emnist-dataset

## How It Works

1. Upload a handwritten character image.
2. The image is converted into grayscale.
3. The image is resized to 28 × 28 pixels.
4. The trained CNN model processes the image.
5. The application predicts the character and displays its confidence score.

## Future Improvements

- Add a drawing canvas for handwriting input.
- Improve recognition accuracy.
- Support complete word recognition.
- Add sentence recognition using sequence models.
- Support real-time handwriting recognition.

## Objective

The main objective of this project is to understand image processing and deep learning by developing a system that can automatically recognize handwritten characters.

## Author

Developed as an AI/ML project using Python and Deep Learning.

