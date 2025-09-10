import jellyfish
import pandas as pd

def normalize_model(model: str, model_list: list) -> str:
    """
    Normalize the model name to a standard format.
    If the model is not recognized, return 'Unknown'.
    """
    model = str(model).strip().title()
    if model in model_list:
        return model
    
    # Fuzzy matching for slight misspellings
    highest_ratio = 0
    best_match = "Unknown"
    for known_model in model_list:
        ratio = jellyfish.jaro_similarity(model, str(known_model))
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = known_model
    
    # Consider it a match if similarity is above a threshold (e.g., 0.8)
    if highest_ratio >= 0.8:
        return best_match
    
    return "Unknown"

def available_models() -> list[str]:
    models = pd.read_csv('../data/tj_prod_car_models_over_20.csv')['model'].tolist()
    return models