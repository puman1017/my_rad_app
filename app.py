import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Page configuration
st.set_page_config(page_title="GRPR-Positive PCa Prediction System", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .paper-title {
        font-size: 0.9rem;
        color: #95a5a6;
        text-align: center;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .threshold-badge {
        background-color: #e8f4fd;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Title with appropriate icon
st.markdown('<p class="main-title">🔬 GRPR-Positive Prostate Cancer Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Interpretable MLP-Based Multi-Regional Radiomics System</p>', unsafe_allow_html=True)
st.markdown('<p class="paper-title">Interpretable MLP-Based Multi-Regional Radiomics for Highly Accurate Discrimination of GRPR-Positive Prostate Cancer from Benign Accumulation</p>', unsafe_allow_html=True)
st.markdown("---")

# ============ Model Thresholds (Optimal thresholds from validation set) ============
MODEL_THRESHOLDS = {
    "Synthesis Model": 0.704,
    "Whole-gland Radiomics": 0.654,
    "Lesion-based Radiomics": 0.740
}

# ============ Feature Definitions ============
FEATURES = {
    "Synthesis Model": [
        "TL", "Morphology",
        "G_original_firstorder_Skewness",
        "G_original_glrlm_ShortRunLowGrayLevelEmphasis",
        "G_original_ngtdm_Busyness",
        "G_original_shape_Elongation",
        "G_original_shape_Flatness",
        "G_original_shape_Maximum3DDiameter",
        "G_original_shape_Sphericity",
        "L_original_glszm_LargeAreaLowGrayLevelEmphasis",
        "L_original_glszm_SmallAreaLowGrayLevelEmphasis",
        "L_original_ngtdm_Busyness",
        "L_original_shape_Maximum2DDiameterRow",
        "L_original_shape_MinorAxisLength",
        "L_original_shape_SurfaceVolumeRatio"
    ],
    "Whole-gland Radiomics": [
        "Morphology", "Location",
        "G_original_shape_Elongation",
        "G_original_shape_Sphericity",
        "G_original_shape_Maximum3DDiameter",
        "G_original_glcm_MCC",
        "G_original_ngtdm_Busyness",
        "G_original_glrlm_ShortRunLowGrayLevelEmphasis"
    ],
    "Lesion-based Radiomics": [
        "TL", "Multifocality", "Morphology",
        "L_original_firstorder_Skewness",
        "L_original_shape_Maximum2DDiameterRow",
        "L_original_shape_Maximum3DDiameter",
        "L_original_shape_MinorAxisLength",
        "L_original_shape_SurfaceVolumeRatio",
        "L_original_glcm_Idn",
        "L_original_glszm_LargeAreaLowGrayLevelEmphasis",
        "L_original_glszm_SmallAreaLowGrayLevelEmphasis",
        "L_original_ngtdm_Busyness"
    ]
}

# ============ Normalization Parameters ============
# Synthesis Model
NORM_SYNTHESIS = {
    "TL": {"mean": 24.826037735849052, "std": 36.560895952433974},
    "Morphology": {"mean": 0.6226415094339622, "std": 0.4893643873724845},
    "G_original_firstorder_Skewness": {"mean": 1.9964738460754716, "std": 1.9605711646612385},
    "G_original_glrlm_ShortRunLowGrayLevelEmphasis": {"mean": 0.03549177352830189, "std": 0.022618584603415693},
    "G_original_ngtdm_Busyness": {"mean": 1.7529059209622642, "std": 2.281183810947237},
    "G_original_shape_Elongation": {"mean": 0.7598143279622643, "std": 0.07771993870686197},
    "G_original_shape_Flatness": {"mean": 0.526505072490566, "std": 0.10781096085770778},
    "G_original_shape_Maximum3DDiameter": {"mean": 58.82346403849057, "std": 9.45170330996504},
    "G_original_shape_Sphericity": {"mean": 0.718781343981132, "std": 0.03395797314734269},
    "L_original_glszm_LargeAreaLowGrayLevelEmphasis": {"mean": 1.7896366415660379, "std": 6.012924274799602},
    "L_original_glszm_SmallAreaLowGrayLevelEmphasis": {"mean": 0.051447064603773594, "std": 0.05496500474543863},
    "L_original_ngtdm_Busyness": {"mean": 0.27346535954716983, "std": 0.42282492417971584},
    "L_original_shape_Maximum2DDiameterRow": {"mean": 20.67078746324528, "std": 10.53390397882241},
    "L_original_shape_MinorAxisLength": {"mean": 15.692108754150944, "std": 7.402117682347095},
    "L_original_shape_SurfaceVolumeRatio": {"mean": 0.8063542526415095, "std": 0.3252819705657271}
}

# Whole-gland Model
NORM_WHOLE_GLAND = {
    "Morphology": {"mean": 0.6226415094339622, "std": 0.4893643873724845},
    "Location": {"mean": 0.4339622641509434, "std": 0.5003627131416442},
    "G_original_shape_Elongation": {"mean": 0.7598143279622643, "std": 0.07771993870686197},
    "G_original_shape_Sphericity": {"mean": 0.718781343981132, "std": 0.03395797314734269},
    "G_original_shape_Maximum3DDiameter": {"mean": 58.82346403849057, "std": 9.45170330996504},
    "G_original_glcm_MCC": {"mean": 0.8330770692830188, "std": 0.05329844920745289},
    "G_original_ngtdm_Busyness": {"mean": 1.7529059209622642, "std": 2.281183810947237},
    "G_original_glrlm_ShortRunLowGrayLevelEmphasis": {"mean": 0.03549177352830189, "std": 0.022618584603415693}
}

# Lesion-based Model
NORM_LESION = {
    "TL": {"mean": 24.826037735849052, "std": 36.560895952433974},
    "Multifocality": {"mean": 0.7735849056603774, "std": 0.42251579096399844},
    "Morphology": {"mean": 0.6226415094339622, "std": 0.4893643873724845},
    "L_original_firstorder_Skewness": {"mean": 0.509864873226415, "std": 0.6914269869109086},
    "L_original_shape_Maximum2DDiameterRow": {"mean": 20.67078746324528, "std": 10.53390397882241},
    "L_original_shape_Maximum3DDiameter": {"mean": 28.03286132209434, "std": 12.72672110444616},
    "L_original_shape_MinorAxisLength": {"mean": 15.692108754150944, "std": 7.402117682347095},
    "L_original_shape_SurfaceVolumeRatio": {"mean": 0.8063542526415095, "std": 0.3252819705657271},
    "L_original_glcm_Idn": {"mean": 0.8667944546415096, "std": 0.03491017289510393},
    "L_original_glszm_LargeAreaLowGrayLevelEmphasis": {"mean": 1.7896366415660379, "std": 6.012924274799602},
    "L_original_glszm_SmallAreaLowGrayLevelEmphasis": {"mean": 0.051447064603773594, "std": 0.05496500474543863},
    "L_original_ngtdm_Busyness": {"mean": 0.27346535954716983, "std": 0.42282492417971584}
}

NORM_MAP = {
    "Synthesis Model": NORM_SYNTHESIS,
    "Whole-gland Radiomics": NORM_WHOLE_GLAND,
    "Lesion-based Radiomics": NORM_LESION
}

# ============ Prediction Function with Custom Threshold ============
def predict_with_threshold(model, X, threshold):
    """
    Predict using custom threshold
    """
    pred_proba = model.predict_proba(X)
    prob_positive = pred_proba[:, 1] if pred_proba.shape[1] == 2 else pred_proba[:, 0]
    predictions = (prob_positive >= threshold).astype(int)
    return predictions, prob_positive

# ============ Normalization Function ============
def normalize_data(df, norm_params):
    """Convert raw data to Z-score normalized values"""
    df_norm = df.copy()
    for col in df.columns:
        if col in norm_params:
            mean = norm_params[col]["mean"]
            std = norm_params[col]["std"]
            if std > 0:
                df_norm[col] = (df[col] - mean) / std
            else:
                df_norm[col] = 0
    return df_norm

# ============ Load Models ============
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Synthesis Model": "models/synthesis_model.pkl",
        "Whole-gland Radiomics": "models/whole_gland_model.pkl",
        "Lesion-based Radiomics": "models/lesion_model.pkl"
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
                st.sidebar.success(f"✅ {name} loaded")
            except Exception as e:
                st.sidebar.error(f"❌ {name} failed to load")
        else:
            st.sidebar.warning(f"⚠️ {name} file not found")
    
    return models

models = load_models()

if not models:
    st.error("❌ No models found!")
    st.info("Please ensure model files are in the 'models' folder:\n"
            "- synthesis_model.pkl\n"
            "- whole_gland_model.pkl\n"
            "- lesion_model.pkl")
    st.stop()

# ============ Sidebar ============
st.sidebar.header("⚙️ Model Selection")
selected_model = st.sidebar.selectbox("Select Prediction Model", list(models.keys()))

feature_list = FEATURES[selected_model]
norm_params = NORM_MAP[selected_model]
model_threshold = MODEL_THRESHOLDS[selected_model]

# Display threshold information
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Classification Threshold")
st.sidebar.markdown(f'<span class="threshold-badge">Optimal threshold: {model_threshold:.3f}</span>', unsafe_allow_html=True)
st.sidebar.caption("Based on validation set optimal Youden index")

st.sidebar.info(f"📊 Feature Count: {len(feature_list)}")
st.sidebar.info("💡 Enter raw clinical data; system will auto-normalize")

# Display feature list
with st.sidebar.expander("View Feature List"):
    for f in feature_list:
        st.write(f"- {f}")

# ============ Main Interface ============
st.header(f"📊 {selected_model}")

# Input method selection
input_method = st.radio("Select Input Method", ["Manual Input", "Upload CSV File"], horizontal=True)

if input_method == "Manual Input":
    st.subheader("Enter Raw Clinical Data")
    st.caption("System will automatically perform Z-score normalization")
    
    # Create input form
    cols = st.columns(3)
    input_values = {}
    
    for i, feature in enumerate(feature_list):
        with cols[i % 3]:
            # Format display name
            display_name = feature.replace('_', ' ').replace('G_original_', 'G: ').replace('L_original_', 'L: ')
            input_values[feature] = st.number_input(
                display_name,
                value=0.0,
                format="%.6f",
                key=f"input_{feature}"
            )
    
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        # Build raw data
        raw_df = pd.DataFrame([input_values])
        
        with st.spinner("Normalizing..."):
            # Normalize
            norm_df = normalize_data(raw_df, norm_params)
        
        with st.spinner("Predicting..."):
            try:
                model = models[selected_model]
                predictions, prob_positive = predict_with_threshold(model, norm_df, model_threshold)
                
                prob = prob_positive[0]
                pred = predictions[0]
                
                st.markdown("---")
                st.header("📈 Prediction Result")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    result_label = "GRPR-Positive" if pred == 1 else "Benign"
                    st.metric("Predicted Class", result_label)
                with col2:
                    st.metric("GRPR-Positive Probability", f"{prob:.2%}")
                with col3:
                    st.metric("Benign Probability", f"{1-prob:.2%}")
                
                # Show threshold info
                st.caption(f"Classification threshold: {model_threshold:.3f}")
                
                # Risk level based on probability
                if prob >= 0.7:
                    st.error("⚠️ **High Risk**: Further clinical evaluation recommended")
                elif prob >= 0.4:
                    st.warning("⚠️ **Moderate Risk**: Close follow-up recommended")
                else:
                    st.success("✅ **Low Risk**: Favorable outcome")
                
                # Show normalized values (for debugging)
                with st.expander("View Normalized Values"):
                    st.dataframe(norm_df)
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")

else:  # Upload CSV
    uploaded_file = st.file_uploader("Upload CSV File (raw feature values)", type=['csv'])
    st.caption("CSV must contain all required feature columns as listed above")
    
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(raw_df)} samples")
            
            with st.expander("View Raw Data Preview"):
                st.dataframe(raw_df.head())
            
            # Check features
            missing = set(feature_list) - set(raw_df.columns)
            if missing:
                st.warning(f"⚠️ Missing features: {missing}")
            else:
                if st.button("🔍 Batch Predict", type="primary", use_container_width=True):
                    with st.spinner("Normalizing..."):
                        # Normalize
                        norm_df = normalize_data(raw_df[feature_list], norm_params)
                    
                    with st.spinner("Predicting..."):
                        try:
                            model = models[selected_model]
                            predictions, prob_positive = predict_with_threshold(model, norm_df, model_threshold)
                            
                            # Add results
                            result_df = raw_df.copy()
                            result_df['Prediction'] = ['GRPR-Positive' if p == 1 else 'Benign' for p in predictions]
                            result_df['GRPR-Positive_Probability'] = prob_positive
                            
                            st.markdown("---")
                            st.header("📊 Batch Prediction Results")
                            st.dataframe(result_df)
                            
                            # Statistics
                            st.markdown("### 📈 Summary Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Samples", len(result_df))
                            with col2:
                                positive_count = (result_df['Prediction'] == 'GRPR-Positive').sum()
                                st.metric("Predicted GRPR-Positive", positive_count)
                            with col3:
                                st.metric("Positive Rate", f"{positive_count/len(result_df):.1%}")
                            with col4:
                                st.metric("Threshold", f"{model_threshold:.3f}")
                            
                            # Download button
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"{selected_model}_results.csv",
                                mime="text/csv"
                            )
                            
                        except Exception as e:
                            st.error(f"Prediction failed: {e}")
                            
        except Exception as e:
            st.error(f"Failed to read file: {e}")

# ============ Footer ============
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Interpretable MLP-Based Multi-Regional Radiomics System | Optimal thresholds: Synthesis=0.704, Whole-gland=0.654, Lesion-based=0.740 | For research use only</p>",
    unsafe_allow_html=True
)