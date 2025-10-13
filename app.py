import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model

@st.cache_data
def load_data():
    df = pd.read_csv("train_data.csv")
    df = df.interpolate()
    return df

FEATURES = [
    'max_risk_scaling', 'time_to_tca', 'mahalanobis_distance', 'max_risk_estimate', 'c_h_per',
    'relative_velocity_t', 'c_recommended_od_span', 'relative_speed', 'c_actual_od_span',
    'c_cd_area_over_mass', 't_j2k_sma', 't_h_per', 'c_ctdot_r', 'c_cr_area_over_mass', 't_h_apo',
    'c_sigma_t', 'c_time_lastob_end', 'c_obs_available', 'c_ctdot_n'
]


def build_model(input_shape):
    inp = Input(shape=input_shape)
    x = BatchNormalization()(inp)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.30)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.20)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.10)(x)
    out = Dense(1, activation='sigmoid', dtype='float32')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model


st.title("🚀 Event Risk Classifier and Warning System")

df = load_data()
scaler = joblib.load("scaler_fast.save")

event_col = st.selectbox("Choose the event column to search by:", df.columns)
event_id = st.text_input("Enter Event Value (ID or other unique info):")

if st.button("Check Risk"):
    row = df[df[event_col].astype(str) == str(event_id)]
    if row.empty:
        st.warning("No event found with that value.")
    else:
        X_sample = row[FEATURES].values
        X_scaled = scaler.transform(X_sample)

        # Build model and load saved weights
        model = build_model((len(FEATURES),))
        try:
            model.load_weights("mlp_fast.weights.h5")
        except Exception as e:
            st.error(f" Could not load trained model weights: {e}")
            st.stop()

        # Make prediction
        y_prob = model.predict(X_scaled)[0, 0]
        risk_label = "HIGH RISK" if y_prob >= 0.5 else "LOW RISK"

        st.markdown(f"### Prediction: **{risk_label}**")
        st.markdown(f"**Probability of High Risk:** {y_prob:.2f}")

        if y_prob >= 0.5:
            st.error("⚠️ WARNING: This event is classified as HIGH RISK. Immediate review is recommended!")
        else:
            st.success(" This event is classified as LOW RISK.")
