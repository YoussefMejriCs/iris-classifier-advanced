import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Classification AI", layout="wide")

try:
    model = joblib.load('iris_model.pkl')
except:
    st.error("Model not found. Please run train.py first.")
    st.stop()

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['species'] = [iris.target_names[i] for i in iris.target]

st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1 {
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
        color: #FF4B4B;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Input Parameters")
    
    sepal_length = st.slider('Sepal Length (cm)', 
                             float(df_iris['sepal length (cm)'].min()), 
                             float(df_iris['sepal length (cm)'].max()), 
                             float(df_iris['sepal length (cm)'].mean()))
    
    sepal_width = st.slider('Sepal Width (cm)', 
                            float(df_iris['sepal width (cm)'].min()), 
                            float(df_iris['sepal width (cm)'].max()), 
                            float(df_iris['sepal width (cm)'].mean()))
    
    petal_length = st.slider('Petal Length (cm)', 
                             float(df_iris['petal length (cm)'].min()), 
                             float(df_iris['petal length (cm)'].max()), 
                             float(df_iris['petal length (cm)'].mean()))
    
    petal_width = st.slider('Petal Width (cm)', 
                            float(df_iris['petal width (cm)'].min()), 
                            float(df_iris['petal width (cm)'].max()), 
                            float(df_iris['petal width (cm)'].mean()))

input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=iris.feature_names)

prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)
predicted_species = iris.target_names[prediction[0]]

st.title("🌸 Iris Flora | AI Classifier")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction")
    st.metric(label="Predicted Species", value=predicted_species.capitalize())
    
    st.subheader("Confidence")
    probs_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
    st.bar_chart(probs_df.T)

with col2:
    st.subheader("Data Visualization")
    tab1, tab2 = st.tabs(["3D Cluster View", "Radar Analysis"])
    
    with tab1:
        fig_3d = px.scatter_3d(df_iris, x='sepal length (cm)', y='sepal width (cm)', z='petal length (cm)',
                             color='species', opacity=0.7, title="Global Data Distribution")
        
        fig_3d.add_trace(go.Scatter3d(
            x=[sepal_length], y=[sepal_width], z=[petal_length],
            mode='markers', marker=dict(size=15, color='white', symbol='diamond'),
            name='Your Input'
        ))
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab2:
        categories = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=[sepal_length, sepal_width, petal_length, petal_width],
            theta=categories,
            fill='toself',
            name='Input Input'
        ))
        
        avg_vals = df_iris[df_iris['species'] == predicted_species].mean(numeric_only=True)
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_vals,
            theta=categories,
            fill='toself',
            name=f'Avg {predicted_species}',
            opacity=0.5
        ))
        
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)