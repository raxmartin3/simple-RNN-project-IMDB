🎬 IMDB Sentiment Analysis using Simple RNN

This project implements a Sentiment Analysis model using a Simple Recurrent Neural Network (RNN) on the IMDB movie reviews dataset. The model classifies reviews as positive or negative using TensorFlow and Keras.

📌 Project Overview

The IMDB dataset contains 50,000 movie reviews labeled as positive or negative. In this project:

Reviews are tokenized and padded

A SimpleRNN model is trained for binary classification

Training & validation performance is visualized

The trained model is saved and reused for prediction

🧠 Model Architecture

Embedding Layer – Converts words into dense vectors

SimpleRNN Layer (128 units) – Learns sequential patterns

Dense Output Layer (Sigmoid) – Predicts sentiment (0 or 1)

📂 Dataset

IMDB Movie Reviews Dataset

Loaded directly using:

from tensorflow.keras.datasets import imdb


Top 10,000 most frequent words are used

Each review is padded to 200 tokens

⚙️ Requirements

Make sure you have Python 3.7+ and install the following:

pip install numpy matplotlib tensorflow

▶️ How to Run

Clone or download the project

Open terminal in project directory

Run:

python project1.py

📊 Training & Evaluation

Optimizer: Adam

Loss Function: Binary Crossentropy

Metric: Accuracy

EarlyStopping: Used to prevent overfitting

Epochs: Up to 30

Batch Size: 32

The project plots:

Training vs Validation Accuracy

Training vs Validation Loss

💾 Model Saving & Loading

The trained model is saved as:

sample_rnn_imdb.h5


And loaded again for evaluation and prediction.

🔍 Sample Prediction

The project also predicts sentiment for a sample review:

prediction = model.predict(sample_review.reshape(1, -1))
sentiment = "positive" if prediction[0][0] > 0.5 else "negative"


Output:

Predicted Sentiment

Prediction Score

Actual Label

📁 Project Structure
├── project1.py
├── sample_rnn_imdb.h5
└── README.md

🚀 Future Improvements

Replace SimpleRNN with LSTM / GRU

Add custom user input review prediction

Build a Streamlit Web App

Improve accuracy using Bidirectional RNN

👨‍💻 Author

Ravi Kr.
Machine Learning & NLP Enthusiast