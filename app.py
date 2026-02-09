import streamlit as st
import numpy as np
import joblib
import json
import os
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EU Car Predict 2026", 
    page_icon="🚗", 
    layout="centered"
)

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    """Loads models and mapping file into memory."""
    try:
        tech_model = joblib.load('tech_model.pkl')
        brand_model = joblib.load('brand_model.pkl')
        mappings = None
        if os.path.exists('mappings.json'):
            with open('mappings.json', 'r', encoding='utf-8') as f:
                mappings = json.load(f)
        return tech_model, brand_model, mappings
    except Exception as e:
        return None, None, str(e)

def estimate_kw_from_engine(engine_size, fuel_type):
    """Fallback logic if power is not provided."""
    if 'diesel' in fuel_type.lower():
        if engine_size <= 1.5: return 75
        elif engine_size <= 2.0: return 110
        return 140
    else:
        if engine_size <= 1.2: return 60
        elif engine_size <= 1.6: return 92
        elif engine_size <= 2.0: return 132
        return 184

# Run resource loading
m_tech, m_brand, mappings = load_resources()

# --- INTERFACE ---
st.title("🚗 EU Car Predict")
st.markdown("Professional vehicle valuation system for the European market (2026).")

if m_tech is None or m_brand is None or mappings is None:
    st.error(f"Critical Error: Missing resource files! (Details: {mappings})")
else:
    # --- SIDEBAR (Input Data) ---
    with st.sidebar:
        st.header("📋 Vehicle Specifications")
        
        # Brand Selection
        brand_list = sorted(list(mappings['brands'].keys()))
        selected_brand = st.selectbox("Brand", options=[b.title() for b in brand_list]).lower()

        # Model Selection
        available_models = sorted([m for m in mappings['models'].keys() if selected_brand in m])
        display_models = [m.replace(selected_brand, "").strip().title() for m in available_models]
        
        full_model_name = None
        if available_models:
            model_idx = st.selectbox("Model", options=range(len(display_models)), format_func=lambda x: display_models[x])
            full_model_name = available_models[model_idx]
        
        st.divider()
        year = st.number_input("Year of Manufacture", min_value=1990, max_value=2026, value=2020)
        engine = st.number_input("Engine Capacity (L)", min_value=0.0, max_value=10.0, value=1.6, step=0.1)
        fuel = st.selectbox("Fuel Type", ["diesel", "petrol", "hybrid", "electric", "cng", "lpg"])
        trans = st.radio("Transmission", ["manual", "automatic"], horizontal=True)
        
        st.divider()
        km_input_val = st.number_input("Mileage (km)", min_value=0, max_value=1000000, value=80000)
        cond_input_val = st.slider("Condition Factor", 0.1, 1.0, 0.85)
        
        st.divider()
        st.subheader("Power Input")
        kw_input = st.number_input("Power in kW", min_value=0, value=0)
        hp_input = st.number_input("Power in Horsepower (HP)", min_value=0, value=0)
        
        # Power Logic
        if kw_input > 0:
            kw = kw_input
            hp = int(kw * 1.36)
        elif hp_input > 0:
            hp = hp_input
            kw = int(hp / 1.36)
        else:
            kw = estimate_kw_from_engine(engine, fuel)
            hp = int(kw * 1.36)

    # --- MAIN PANEL ---
    st.subheader("💰 Valuation Result")
    
    if st.button("CALCULATE ESTIMATED PRICE", use_container_width=True):
        if full_model_name:
            try:
                # Encoding lookup
                b_enc = mappings['brands'].get(selected_brand)
                m_enc = mappings['models'].get(full_model_name)
                
                # Feature preparation
                is_auto = 1 if trans == 'automatic' else 0
                is_manual = 1 if trans == 'manual' else 0
                f_dict = {f: (1 if fuel == f else 0) for f in ['cng', 'diesel', 'electric', 'hybrid', 'lpg', 'petrol']}

                # Order: [year, kw, hp, km, engine, auto, manual, cng, diesel, electric, hybrid, lpg, petrol]
                tech_vector = np.array([[year, kw, hp, km_input_val, engine, is_auto, is_manual, 
                                         f_dict['cng'], f_dict['diesel'], f_dict['electric'], 
                                         f_dict['hybrid'], f_dict['lpg'], f_dict['petrol']]])
                brand_vector = np.array([[b_enc, m_enc]])

                # Two-stage prediction
                log_base = m_tech.predict(tech_vector)[0]
                log_brand = m_brand.predict(brand_vector)[0]
                final_price = np.expm1(log_base + log_brand) * cond_input_val

                # RESULTS DISPLAY
                st.metric("Estimated Market Price", f"{final_price:,.2f} €")
                
                st.info(f"Please note that the system operates with a ± 2,000 € margin of error, which should be taken into consideration.")
                
                # Detailed breakdown
                with st.expander("See technical details"):
                    st.write(f"**Vehicle:** {selected_brand.upper()} {full_model_name.upper()}")
                    st.write(f"**Calculated Power:** {kw} kW / {hp} HP")
                    st.write(f"**Condition applied:** {cond_input_val}")
                
                st.balloons()
            except Exception as e:
                st.error(f"Error during calculation: {e}")
        else:
            st.error("Please select a valid model from the sidebar.")

# Footer
st.markdown("---")
st.caption("EU Car Predict v2.2 | English Version | 2026")