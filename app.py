# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os # Jaruri hai agar aapko path set karna ho

# --- 1. CONFIGURATION AND SETUP ---
st.set_page_config(
    page_title="AI Churn Predictor | Portfolio App",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Dark Mode Visibility aur Design Fixes) ---
st.markdown("""
<style>
    /* ... (Existing code) ... */

    /* 3. Input Field Styling (Drop-down Menu Fix) */
    
    /* Dropdown menu ke items aur background ko dark mode friendly karein */
    .stSelectbox > div[data-testid="stDataframeTooltipAndCorner"] {
        background-color: #374151; /* Input box background */
        color: #f3f4f6; /* Input box text color */
    }

    /* **YAHAN PAR NAYA CODE SHAMIL KAREIN** */
    /* Selectbox popup menu (jahan options dikhte hain) */
    div[data-testid="stVirtualList"] div {
        background-color: #374151 !important; /* Popup menu ka background dark grey */
        color: #f3f4f6 !important; /* Popup menu ka text halka safed */
    }

    /* Jab item select ho ya hover ho, uska color theek karein */
    div[data-testid="stVirtualList"] div:hover {
        background-color: #2563eb !important; /* Hover color blue */
        color: white !important;
    }
    /* ... (Baaki code wahi rahega) ... */
</style>
""", unsafe_allow_html=True)

# --- 2. ASSETS LOADING ---
MODEL_PATH = 'churn_deployment_assets.joblib'

@st.cache_resource
def load_assets():
    """Loads the model, features, and threshold."""
    try:
        # joblib.load is now safe because we ensured the right sklearn version
        assets = joblib.load(MODEL_PATH)
        return assets
    except FileNotFoundError:
        return None

assets = load_assets()

if assets is None:
    st.error("❌ ERROR: Deployment assets file not found. Please place 'churn_deployment_assets.joblib' in the root folder.")
    st.stop() 

model = assets['model']
REQUIRED_FEATURES = assets['input_features'] 
FINAL_THRESHOLD = assets['threshold']


# --- 3. SIDEBAR (Enhanced Professionalism) ---

with st.sidebar:
    # 1. Image Error Fix: Image ko hata dein ya naya working link daalein.
    # st.image("https://i.imgur.com/gK6kS7z.png", use_column_width=True) 
    # HUM FILHAAL IMAGE KO HATA RAHE HAIN
    st.markdown("<h2 style='text-align:center;'>📡 Churn Analyzer</h2>", unsafe_allow_html=True) 
    
    st.info(
        """
        This application utilizes an **Optimized Random Forest Classifier** to predict customer churn risk in the telecom sector. 
        It is built for portfolio demonstration using Streamlit.
        """
    )
    # app.py file mein st.sidebar block ke andar ka naya code:

with st.sidebar:
    # ... (About This Model section wahi rahega) ...

    st.markdown("---")
    st.markdown("### 📈 Key Model Insights")
    
    # Dashboard-style Metrics ko do columns mein dikhayein
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric(label="Decision Threshold", 
                  value=f"{FINAL_THRESHOLD:.2f}",
                  delta="0.55 (Optimized)",
                  delta_color="normal")
        
    with col_b:
        # F1-Score dikha rahe hain kyunki yeh Precision aur Recall ko balance karta hai
        st.metric(label="Overall F1-Score", 
                  value="75.5%",
                  delta="Balanced Metric",
                  delta_color="off")
    
    st.markdown("""
        <div style="font-size: 0.9em; padding-top: 10px; color: #aaa;">
        *Note: Metrics are based on the model's weighted average performance on the test set.*
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("Developed by **Moizz**")
    # ... baaki code ...


# --- 4. MAIN APP UI ---
st.title("Telecom Customer Churn Predictor 📡")
st.markdown("A Machine Learning powered tool to forecast customer attrition and enhance retention strategies.")
st.markdown("---")

# Input Form Structure
with st.container(border=True):
    st.subheader("Enter Customer Profile Details:")
    
    col1, col2, col3 = st.columns(3)
    
    # Categorical Inputs
    with col1:
        Contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        OnlineSecurity = st.selectbox("Online Security", ['No', 'Yes', 'No internet service']) 
        
    with col2:
        TechSupport = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service']) 
        PaymentMethod = st.selectbox("Payment Method", 
                                   ['Electronic check', 'Mailed check', 
                                    'Bank transfer (automatic)', 'Credit card (automatic)'])
        
    with col3:
        TenureMonths = st.slider("Tenure (Months)", 1, 72, 12, help="Number of months customer has stayed.")
        
    st.markdown("---")
    
    # Numerical Inputs
    col_num1, col_num2 = st.columns(2)
    MonthlyCharges = col_num1.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=50.0, step=0.01)
    TotalCharges = col_num2.number_input("Total Charges ($)", min_value=0.0, value=600.0, step=0.01)

    # Submit Button
    submit_button = st.button(label="Analyze Churn Risk", use_container_width=True, type="primary")


# --- 5. PREDICTION LOGIC AND RESULTS ---
if submit_button:
    # Input Data Preparation (ensure order matches REQUIRED_FEATURES)
    input_data = pd.DataFrame([{
        'Contract': Contract,
        'Online Security': OnlineSecurity,
        'Tech Support': TechSupport,
        'Payment Method': PaymentMethod,
        'Tenure Months': TenureMonths,
        'Monthly Charges': MonthlyCharges,
        'Total Charges': TotalCharges
    }])

    with st.spinner('Performing advanced churn analysis...'):
        churn_prob = model.predict_proba(input_data)[:, 1][0]
        retention_prob = 1 - churn_prob
        
        st.markdown("## Prediction Results 📊")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        # Metric Display
        col_res1.metric(
            label="Likelihood of Churn", 
            value=f"{churn_prob:.2%}", 
            delta=f"Threshold: {FINAL_THRESHOLD*100:.0f}%",
            delta_color="inverse"
        )
        col_res2.metric(
            label="Customer Retention Potential", 
            value=f"{retention_prob:.2%}", 
            delta_color="normal",
            delta=f"Stability Score"
        )

        # Final Decision and Animated Feedback
        if churn_prob >= FINAL_THRESHOLD:
            col_res3.error("🚨 HIGH RISK: IMMEDIATE ACTION REQUIRED")
            st.warning("Customer Retention Team: This profile exhibits strong indicators of leaving soon. Intervention is strongly advised.")
            st.balloons() # Animation for high risk
        else:
            col_res3.success("✅ LOW RISK: CUSTOMER IS STABLE")
            st.info("This customer is stable. Continue monitoring the account.")
            st.snow() # Animation for stability