"""
Tools for Part 3 Support Agent:
1. check_return_risk(order_features: dict) -> dict
   Loads Part 1's models/return_risk_model.pkl (tuned Random Forest) and anchors risk buckets
   to t*_rf (from models/threshold_rf.txt).
2. classify_product_image(image_path: str) -> dict
   Loads Part 2's models/product_classifier.pt and classifies sample images.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

# Cached models
_return_risk_model = None
_t_star_rf = None

FEATURE_COLUMNS = [
    'product_category', 'price_inr', 'discount_pct', 'payment_method',
    'customer_tenure_days', 'num_previous_orders', 'num_previous_returns',
    'delivery_distance_km', 'delivery_days', 'is_weekend_order', 'rating_given'
]

def load_threshold_rf(threshold_path: str = "models/threshold_rf.txt") -> float:
    global _t_star_rf
    if _t_star_rf is not None:
        return _t_star_rf
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            _t_star_rf = float(f.read().strip())
    else:
        _t_star_rf = 0.50
    return _t_star_rf

def get_return_risk_model(model_path: str = "models/return_risk_model.pkl"):
    global _return_risk_model
    if _return_risk_model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Part 1 return risk model not found at {model_path}")
        _return_risk_model = joblib.load(model_path)
    return _return_risk_model

def check_return_risk(order_features: dict, model_path: str = "models/return_risk_model.pkl") -> dict:
    """
    Task 3: Tool to score return risk using Part 1's tuned Random Forest model.
    Cut points are calibrated dynamically relative to t*_rf (F1-maximizing threshold):
    - Low Risk: return_probability < t*_rf
    - High Risk: return_probability >= t*_rf + 0.15
    - Medium Risk: otherwise (t*_rf <= return_probability < t*_rf + 0.15)
    
    Args:
        order_features (dict): Dictionary with order feature values.
        
    Returns:
        dict: {
            'order_id': any,
            'return_probability': float,
            'risk_bucket': str, # 'Low', 'Medium', 'High'
            't_star_rf': float,
            'cut_points': dict,
            'is_high_risk': bool
        }
    """
    model = get_return_risk_model(model_path)
    t_star = load_threshold_rf()
    
    order_id = order_features.get("order_id", "N/A")
    
    # Prepare DataFrame matching training feature columns
    row_data = {}
    for col in FEATURE_COLUMNS:
        row_data[col] = [order_features.get(col, None)]
    
    input_df = pd.DataFrame(row_data)
    
    # Predict probabilities
    proba = float(model.predict_proba(input_df)[0, 1])
    
    # Risk bucket anchored to t*_rf
    high_threshold = round(t_star + 0.15, 4)
    if proba < t_star:
        risk_bucket = "Low"
    elif proba >= high_threshold:
        risk_bucket = "High"
    else:
        risk_bucket = "Medium"
        
    return {
        "order_id": order_id,
        "return_probability": round(proba, 4),
        "risk_bucket": risk_bucket,
        "t_star_rf": t_star,
        "cut_points": {
            "low_max": t_star,
            "high_min": high_threshold
        },
        "is_high_risk": (proba >= t_star)
    }

# Re-export classify_product_image from part2.predict
from part2.predict import classify_product_image

if __name__ == "__main__":
    sample_order = {
        "order_id": 4,
        "product_category": "Home",
        "price_inr": 7930.0,
        "discount_pct": 49.7,
        "payment_method": "COD",
        "customer_tenure_days": 1479,
        "num_previous_orders": 33,
        "num_previous_returns": 4,
        "delivery_distance_km": 143.1,
        "delivery_days": 5,
        "is_weekend_order": 0,
        "rating_given": 3.0
    }
    risk_res = check_return_risk(sample_order)
    print("Return Risk Tool Result:")
    print(risk_res)
    
    img_path = "data/sample_images/00_ankle_boot.png"
    img_res = classify_product_image(img_path)
    print("\nImage Classifier Tool Result:")
    print(img_res)
