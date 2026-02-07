import numpy as np
import joblib
import json
import os

def load_resources():
    """Loads models and the mapping dictionary once at startup."""
    try:
        # Load the trained XGBoost models
        tech_model = joblib.load('tech_model.pkl')
        brand_model = joblib.load('brand_model.pkl')
        
        # Load the mappings.json for automatic encoding lookup
        mappings = None
        if os.path.exists('mappings.json'):
            with open('mappings.json', 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            print("[SUCCESS] Resources and mappings loaded correctly.")
        else:
            print("[INFO] mappings.json not found. Manual encoding will be required.")
            
        return tech_model, brand_model, mappings
    except Exception as e:
        print(f"[ERROR] Could not load resources: {e}")
        return None, None, None

def estimate_kw_from_engine(engine_size, fuel_type):
    """Fallback logic: Estimates kW based on engine size if power is unknown."""
    if 'diesel' in fuel_type.lower():
        if engine_size <= 1.5: return 75   # ~102 HP
        elif engine_size <= 2.0: return 110 # ~150 HP
        return 140                         # ~190 HP
    else: # Petrol / Hybrid / Others
        if engine_size <= 1.2: return 60   # ~82 HP
        elif engine_size <= 1.6: return 92  # ~125 HP
        elif engine_size <= 2.0: return 132 # ~180 HP
        return 184                         # ~250 HP

def get_prediction(m_tech, m_brand, mappings):
    print("\n" + "="*55)
    print("      EUROPEAN VEHICLE VALUATION (2026)")
    print("="*55)
    
    try:
        # 1. BRAND INPUT
        brand = input("1. Brand (e.g. Ford): ").strip()
        brand_key = brand.lower()

        # 2. SMART MODEL SELECTION
        available_models = []
        full_model_name = ""
        
        if mappings:
            available_models = sorted([m for m in mappings['models'].keys() if brand_key in m])

        if available_models:
            print(f"\n   [INFO] I found {len(available_models)} models for {brand.upper()}:")
            for i, m in enumerate(available_models, 1):
                display_name = m.replace(brand_key, "").strip().title()
                print(f"    {i:2}. {display_name}")
            
            print("-" * 30)
            model_choice = input(f"2. Select Model (number or type name): ").strip().lower()
            
            if model_choice.isdigit() and 1 <= int(model_choice) <= len(available_models):
                full_model_name = available_models[int(model_choice) - 1]
            else:
                if f"{brand_key} {model_choice}" in mappings['models']:
                    full_model_name = f"{brand_key} {model_choice}"
                elif model_choice in mappings['models']:
                    full_model_name = model_choice
                else:
                    full_model_name = model_choice
        else:
            full_model_name = input("2. Model Name (e.g. Mondeo): ").strip().lower()

        # 3. TECHNICAL SPECIFICATIONS
        year = int(input("3. Year of Manufacture: "))
        engine = float(input("4. Engine Capacity (L, e.g. 2.0): "))
        fuel = input("5. Fuel Type (petrol/diesel/hybrid/cng/lpg/electric): ").strip().lower()

        # 4. POWER LOGIC (kW & HP Fallback)
        kw_input = input("6. Power in kW [Press Enter to skip]: ").strip()
        hp_input = input("7. Power in Horsepower (PS) [Press Enter to skip]: ").strip()
        
        if kw_input != "":
            kw = int(kw_input)
            hp = int(kw * 1.36)
        elif hp_input != "":
            hp = int(hp_input)
            kw = int(hp / 1.36)
        else:
            kw = estimate_kw_from_engine(engine, fuel)
            hp = int(kw * 1.36)
            print(f"   --> Auto-calculated from engine: {kw} kW / {hp} PS")

        km = int(input("8. Total Mileage (km): "))
        trans = input("9. Transmission (manual/automatic): ").strip().lower()
        cond = float(input("10. Condition Factor (0.1 to 1.0): "))

        # 5. ENCODING LOOKUP
        b_enc, m_enc = None, None
        if mappings:
            b_enc = mappings['brands'].get(brand_key)
            m_enc = mappings['models'].get(full_model_name)

        if b_enc is None:
            b_enc = float(input("   [!] Brand not found. Enter Brand Encoded Value manually: "))
        if m_enc is None:
            m_enc = float(input("   [!] Model not found. Enter Model Encoded Value manually: "))

        # 6. FEATURE VECTOR PREPARATION
        is_auto = 1 if trans == 'automatic' else 0
        is_manual = 1 if trans == 'manual' else 0
        is_diesel = 1 if fuel == 'diesel' else 0
        is_petrol = 1 if fuel == 'petrol' else 0
        is_hybrid = 1 if fuel == 'hybrid' else 0
        is_cng = 1 if fuel == 'cng' else 0
        is_lpg = 1 if fuel == 'lpg' else 0
        is_electric = 1 if fuel == 'electric' else 0
        
        # Order must match training: [year, kw, hp, km, engine, auto, manual, cng, diesel, electric, hybrid, lpg, petrol]
        tech_vector = np.array([[year, kw, hp, km, engine, is_auto, is_manual, is_cng, is_diesel, is_electric, is_hybrid, is_lpg, is_petrol]])
        brand_vector = np.array([[b_enc, m_enc]])

        # 7. PREDICTION
        log_base = m_tech.predict(tech_vector)[0]
        log_brand = m_brand.predict(brand_vector)[0]
        
        price = np.expm1(log_base + log_brand) * cond

        # 8. RESULTS OUTPUT
        print("\n" + "*"*55)
        print(f" VALUATION: {brand.upper()} {full_model_name.upper()}")
        print(f" Specs: {engine}L | {fuel.upper()} | {kw} kW / {hp} PS")
        print(f" ESTIMATED PRICE: {price:,.2f} €")
        print("-" * 55)
        print(f" MARKET RANGE: {max(0, price - 2000):,.0f} € - {price + 2000:,.0f} €")
        print(" [INFO] This valuation includes a +/- 2,000 € margin")
        print(" based on local market trends and vehicle condition.")
        print("*"*55 + "\n")

    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")

def main():
    m_tech, m_brand, mappings = load_resources()
    
    if m_tech is None or m_brand is None:
        print("[CRITICAL] Could not load models. Exiting...")
        return

    while True:
        get_prediction(m_tech, m_brand, mappings)
        
        choice = input("Would you like to predict another car? (y/n): ").strip().lower()
        if choice != 'y':
            print("\nThank you for using EU Car Predict! Goodbye, dear friend!")
            break

if __name__ == "__main__":
    main()