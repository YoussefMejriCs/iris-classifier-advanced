🌸 Iris Flora: Advanced Classification Web Application

A full-stack data science project demonstrating a complete machine learning pipeline, from model training to interactive web deployment. This project classifies Iris flower species based on their sepal and petal measurements using a highly accurate Random Forest model, visualized via a professional Streamlit interface.

🔗 Live Demo: iris-classifier-advanced.streamlit.app

🔗 Portfolio: youssefmejri.com

✨ Features

Aspect

Description

Key Technologies

🤖 Model

Robust Random Forest Classifier trained on the classic Iris dataset.

scikit-learn, joblib

🏗️ Architecture

Clean separation of concerns: train.py for model generation and app.py for deployment.

Python

🎛️ Interface

Interactive web application allowing users to input parameters via sliders.

Streamlit

📊 Visualization

Elite, interactive data visualization for deep analysis.

Plotly Express, Graph Objects

🧠 Analysis

Includes a 3D Scatter Plot of the dataset and a dynamic Radar Chart comparing inputs to species averages.

Data Visualization

🚀 Getting Started

Follow these steps to set up and run the project locally on your machine.

1. Prerequisites

Ensure you have Python 3.7+ installed.

2. Clone the Repository

git clone [https://github.com/YoussefMejriCs/iris-classification-advanced.git](https://github.com/YoussefMejriCs/iris-classification-advanced.git)
cd iris-classification-advanced


3. Install Dependencies

Install all required Python packages using the provided requirements file.

pip install -r requirements.txt


4. Train the Model

Execute the training script to generate the serialized model file (iris_model.pkl).

python train.py


5. Run the Streamlit App

Launch the web application. This will open the app in your default browser.

streamlit run app.py


🖼️ Application Overview

Interactive Prediction Interface

Users can adjust the four key biological features (sepal length/width, petal length/width) using sliders. The model instantly outputs the predicted species along with confidence scores.

High-Class Visualization

The application provides two advanced views for data insight:

3D Scatter Plot: Visualizes the entire dataset distribution in 3D space and highlights the user's input point relative to the natural clusters.

Radar Chart: Compares the user's input features against the average feature values of the predicted species, providing immediate dimensional analysis.

🤝 Contribution

Feel free to fork the repository, submit pull requests, or open issues.

Developed by Youssef Mejri
