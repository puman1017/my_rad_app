import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="前列腺癌影像组学预测", layout="wide")
st.title("🧠 前列腺癌影像组学预测系统")

# 特征定义
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
            models[name] = joblib.load(path)
            st.sidebar.success(f"✅ {name}")
        else:
            st.sidebar.error(f"❌ {name} 文件不存在")
    return models

models = load_models()
if not models:
    st.error("模型文件缺失！请检查 models 文件夹")
    st.stop()

selected_model = st.sidebar.selectbox("选择模型", list(models.keys()))
feature_list = FEATURES[selected_model]
st.sidebar.info(f"特征数: {len(feature_list)}")

input_method = st.radio("输入方式", ["手动输入", "上传CSV"], horizontal=True)

if input_method == "手动输入":
    cols = st.columns(3)
    input_values = {}
    for i, f in enumerate(feature_list):
        with cols[i % 3]:
            input_values[f] = st.number_input(f.replace('_', ' '), value=0.0, format="%.6f")
    
    if st.button("预测", type="primary"):
        df = pd.DataFrame([input_values])
        model = models[selected_model]
        proba = model.predict_proba(df)[0][1]
        pred = "恶性" if proba > 0.5 else "良性"
        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.metric("预测结果", pred)
        col2.metric("恶性概率", f"{proba:.2%}")
        if proba >= 0.7:
            st.error("高风险")
        elif proba >= 0.4:
            st.warning("中风险")
        else:
            st.success("低风险")

else:
    uploaded = st.file_uploader("上传CSV", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        if set(feature_list).issubset(set(df.columns)):
            model = models[selected_model]
            proba = model.predict_proba(df[feature_list])[:, 1]
            df['恶性概率'] = proba
            df['预测'] = ['恶性' if p > 0.5 else '良性' for p in proba]
            st.dataframe(df)
            st.download_button("下载结果", df.to_csv(index=False), "results.csv")
        else:
            st.error("CSV缺少必要的特征列")