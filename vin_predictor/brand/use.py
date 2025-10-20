#!/usr/bin/env python3

import sys
from train import VINBrandPredictor

if __name__ == "__main__":
    # Initialize predictor
    predictor = VINBrandPredictor()

    predictor.load_model("vin_brand_predictor")

    args = sys.argv[1:]

    predictions = predictor.predict(args)
    
    for pred in predictions:
        print(f"VIN: {pred['vin']}")
        print(f"Predicted Brand: {pred['Predicted_Brand']}")
        print(f"Confidence: {pred['Confidence']:.4f}")
        print("-" * 40)