import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Set page layout and title
st.set_page_config(page_title="Attri-NeuLift", layout="wide")
st.title("Attri-NeuLift")

# Load model
@st.cache_resource
def load_model(filepath):
    with open(filepath, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model("synapses_checkpint.pkt")

# Define input features
features = [f"f{i}" for i in range(12)]
feature_values = {}

# Two-column layout: left for sliders, right for output
col1, col2 = st.columns([1, 2])

with col1:
    with st.container():
        st.subheader("🔧 Feature Inputs")
        st.markdown("---")

        # Create a 6x2 layout for sliders
        for i in range(0, 12, 2):
            c1, c2 = st.columns(2)

            with c1:
                feature_values[features[i]] = st.slider(
                    features[i],
                    min_value=[12.0, 10.0, 8.0, -8.4, 10.0, -9.0, -31.0, 4.83, 3.64, 13.0, 5.0, -1.4][i],
                    max_value=[27.0, 17.0, 9.5, 4.7, 21.1, 4.2, 0.3, 7.0, 4.0, 75.0, 6.5, -0.1][i],
                    value=[20.0, 14.0, 8.6, 0.0, 15.0, -2.0, -15.0, 6.0, 3.8, 40.0, 5.75, -0.8][i],
                    step=0.01
                )

            with c2:
                feature_values[features[i+1]] = st.slider(
                    features[i+1],
                    min_value=[12.0, 10.0, 8.0, -8.4, 10.0, -9.0, -31.0, 4.83, 3.64, 13.0, 5.0, -1.4][i+1],
                    max_value=[27.0, 17.0, 9.5, 4.7, 21.1, 4.2, 0.3, 7.0, 4.0, 75.0, 6.5, -0.1][i+1],
                    value=[20.0, 14.0, 8.6, 0.0, 15.0, -2.0, -15.0, 6.0, 3.8, 40.0, 5.75, -0.8][i+1],
                    step=0.01
                )

with col2:
    st.subheader("📊 Predicted Conversion Rate")

    # Prepare input
    input_features = np.array([list(feature_values.values())])

    try:
        prediction = model.predict(input_features)
        conversion_rate = prediction[0]
    except Exception as e:
        conversion_rate = None
        st.error(f"An error occurred during inference: {e}")

    if conversion_rate is not None:
        st.metric(label="Estimated Conversion Rate", value=f"{conversion_rate.item() * 100:.2f}%")

    # Visualize feature values
    st.subheader("📈 Feature Values")
    df = pd.DataFrame(list(feature_values.items()), columns=["Feature", "Value"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["Feature"], df["Value"], color='skyblue')
    ax.set_title("Feature Values")
    ax.set_ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
