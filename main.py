"""
===============================================================================
EcoScan AI - Main Application & Training Pipeline
Multimodal AI Platform for Supply Chain Traceability & Anti-Greenwashing Intelligence
===============================================================================
Author: Jalaja Sunil Borkar
Department: Artificial Intelligence & Data Science, DMCE
===============================================================================
"""

import os
import re
import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("EcoScanAI")


# -----------------------------------------------------------------------------
# 1. AUTOMATED DOCUMENT PARSING ENGINE (NLP / Regex Extraction)
# -----------------------------------------------------------------------------
class DocumentParsingEngine:
    """Simulates NLP/OCR extraction from unstructured shipping manifests, customs logs, and invoices."""
    
    @staticmethod
    def parse_unstructured_text(text_document: str) -> dict:
        """Extracts structured ESG and supply chain features from raw text logs."""
        logger.info("Parsing unstructured logistics document using NLP Engine...")
        
        # Regex patterns to extract key operational metrics
        dist_match = re.search(r'(\d+[\d,]*)\s*km', text_document, re.IGNORECASE)
        recycled_match = re.search(r'(\d+)\%\s*recycled', text_document, re.IGNORECASE)
        cert_match = re.search(r'certified:\s*(yes|no)', text_document, re.IGNORECASE)
        mode_match = re.search(r'mode:\s*(maritime|air_freight|rail|road_truck)', text_document, re.IGNORECASE)
        grid_match = re.search(r'grid_renewable:\s*(\d+)\%', text_document, re.IGNORECASE)
        
        extracted_features = {
            'tier1_energy_kwh_per_unit': 12.5,  # Default baseline extraction
            'tier2_recycled_content_pct': float(recycled_match.group(1)) if recycled_match else 20.0,
            'tier3_raw_extraction_co2_kg': 4.5,
            'logistics_transit_km': float(dist_match.group(1).replace(',', '')) if dist_match else 5000.0,
            'freight_transport_mode': mode_match.group(1).lower() if mode_match else 'maritime',
            'vendor_certified': 'YES' if (cert_match and cert_match.group(1).lower() == 'yes') else 'NO',
            'power_grid_renewable_share': float(grid_match.group(1)) / 100.0 if grid_match else 0.35
        }
        
        logger.info(f"Extracted Features: {extracted_features}")
        return extracted_features


# -----------------------------------------------------------------------------
# 2. SYNTHETIC DATASET GENERATOR
# -----------------------------------------------------------------------------
def generate_synthetic_supply_chain_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generates a synthetic dataset of multi-tier supply chain records for model training."""
    logger.info(f"Generating {n_samples} synthetic supply chain records...")
    np.random.seed(42)
    
    modes = ['maritime', 'air_freight', 'rail', 'road_truck']
    certified_opts = ['YES', 'NO']
    
    tier1_energy = np.random.uniform(2.0, 30.0, n_samples)
    tier2_recycled = np.random.uniform(0.0, 100.0, n_samples)
    tier3_co2 = np.random.uniform(0.5, 12.0, n_samples)
    transit_km = np.random.uniform(100.0, 15000.0, n_samples)
    transport_mode = np.random.choice(modes, size=n_samples, p=[0.5, 0.2, 0.15, 0.15])
    vendor_certified = np.random.choice(certified_opts, size=n_samples, p=[0.6, 0.4])
    grid_renewable = np.random.uniform(0.05, 0.95, n_samples)
    
    # Ground truth equation for Eco-Verity Score (1–100 scale)
    base_score = 50.0
    score = (
        base_score
        + (tier2_recycled * 0.25)
        + (grid_renewable * 20.0)
        + (np.where(vendor_certified == 'YES', 15.0, -10.0))
        - (tier1_energy * 0.5)
        - (tier3_co2 * 1.8)
        - (transit_km * 0.001)
        - (np.where(transport_mode == 'air_freight', 15.0, 0.0))
        + np.random.normal(0, 2.5, n_samples)
    )
    
    # Clamp scores between 1.0 and 100.0
    eco_verity_score = np.clip(score, 1.0, 100.0)
    
    df = pd.DataFrame({
        'tier1_energy_kwh_per_unit': tier1_energy,
        'tier2_recycled_content_pct': tier2_recycled,
        'tier3_raw_extraction_co2_kg': tier3_co2,
        'logistics_transit_km': transit_km,
        'freight_transport_mode': transport_mode,
        'vendor_certified': vendor_certified,
        'power_grid_renewable_share': grid_renewable,
        'eco_verity_score': eco_verity_score
    })
    
    return df


# -----------------------------------------------------------------------------
# 3. MACHINE LEARNING PIPELINE SCHEMA
# -----------------------------------------------------------------------------
CATEGORICAL_COLS = ['freight_transport_mode', 'vendor_certified']
NUMERICAL_COLS = [
    'tier1_energy_kwh_per_unit',
    'tier2_recycled_content_pct',
    'tier3_raw_extraction_co2_kg',
    'logistics_transit_km',
    'power_grid_renewable_share'
]


def build_ml_pipeline():
    """Constructs a Scikit-learn Pipeline with ColumnTransformer and RandomForestRegressor."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    return pipeline


# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION WORKFLOW
# -----------------------------------------------------------------------------
def main():
    logger.info("Starting EcoScan AI Training & Audit Pipeline...")
    
    # Load local CSV or generate synthetic dataset
    dataset_path = os.path.join('data', 'ecoscan_supply_chain.csv')
    if os.path.exists(dataset_path):
        logger.info(f"Loading dataset from '{dataset_path}'...")
        df = pd.read_csv(dataset_path)
    else:
        logger.warning("No CSV supplied; generating 1,000 synthetic supply chain records.")
        df = generate_synthetic_supply_chain_data(1000)
    
    X = df.drop(columns=['eco_verity_score'])
    y = df['eco_verity_score']
    
    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    
    # Train Pipeline
    pipeline = build_ml_pipeline()
    pipeline.fit(X_train, y_train)
    logger.info("RandomForestRegressor pipeline trained successfully!")
    
    # Evaluate Pipeline
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "=" * 70)
    print("         ECOSCAN AI MODEL PERFORMANCE METRICS")
    print("=" * 70)
    print(f"R² Score (Variance Explained): {r2:.4f} ({r2 * 100:.2f}%)")
    print(f"Mean Absolute Error (MAE):    {mae:.4f} score points")
    print(f"Root Mean Squared Error (RMSE):{rmse:.4f} score points")
    print("=" * 70 + "\n")
    
    # Save Model Artifact
    model_path = 'ecoscan_model.joblib'
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved trained pipeline artifact to '{model_path}'")
    
    # -------------------------------------------------------------------------
    # 5. REAL-TIME SAMPLE INFERENCE TEST
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("        SAMPLE INFERENCE: LOGISTICS MANIFEST AUDIT")
    print("=" * 70)
    
    sample_manifest_text = """
    CUSTOMS DECLARATION & BILL OF LADING #ECO-9042
    Origin: Overseas Port | Destination: Enterprise Logistics Hub
    Logistics Route: 8420 km | Mode: maritime
    Tier 2 Material Audit: 82% recycled cotton blend
    Facility Certification: Certified: YES
    Local Power Grid Profile: Grid_Renewable: 65%
    """
    
    parsed_features = DocumentParsingEngine.parse_unstructured_text(sample_manifest_text)
    df_sample = pd.DataFrame([parsed_features])
    
    score = pipeline.predict(df_sample)[0]
    status = "VERIFIED SUSTAINABLE 🌱" if score >= 70.0 else "HIGH GREENWASHING RISK 🚨"
    
    print("\n----------------------------------------------------------------------")
    print(f"Manifest Document        : Bill of Lading #ECO-9042")
    print(f"Predicted Eco-Verity Score: {score:.2f} / 100")
    print(f"Compliance Audit Status   : {status}")
    print("----------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()