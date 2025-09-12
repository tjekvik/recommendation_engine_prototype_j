import os, requests
from vin_predictor.train import VINBrandPredictor
from vininfo import Vin


def decode_vin_corgi(vin:str) -> dict:
    url = os.getenv("VIN_API_URL", "http://vin-decoder:4000")

    response = requests.get(f"{url}/{vin}")
    if response.status_code == 200:
        return corgi_to_tjekvik_format(response.json())
    else:
        return {}

def corgi_to_tjekvik_format(corgi_data: dict) -> dict:
    components = corgi_data.get("components", {})
    vehicle = components.get("vehicle", {})
    plant = components.get("plant", {})
    engine = components.get("engine", {})
    vin = corgi_data.get("vin", "")
    valid = corgi_data.get("valid", False)
    manufacturer = vehicle.get("manufacturer", "")
    make = vehicle.get("make", "")
    model = vehicle.get("model", "")
    year = vehicle.get("year", "")
    body_style = vehicle.get("bodyStyle", "")
    fuel_type = vehicle.get("fuelType", "")
    doors = vehicle.get("doors", "")
    plant_country = plant.get("country", "")
    plant_city = plant.get("city", "")
    plant_code = plant.get("code", "")
    engine_fuel = engine.get("fuel", "")
    return {
        "vin": vin,
        "valid": valid,
        "year": year,
        "make": make,
        "model": model,
        "bodyStyle": body_style,
        "fuelType": fuel_type,
        "doors": doors,
        "plant": {
            "country": plant_country,
            "city": plant_city,
            "code": plant_code
        },
        "engine": {
            "fuel": engine_fuel
        },
        "manufacturer": manufacturer
    }
    
def decode_vin_vininfo(vin: str) -> dict:

    try:
        v = Vin(vin)
        model = v.details.model if v.details is not None else "N/A"
        plant = v.details.plant if v.details is not None else {}
        body_style = v.details.body if v.details is not None else "N/A"
        engine = v.details.engine if v.details is not None else {}
        return {
            "vin": vin,
            "valid": True,
            "year": v.years[0],
            "make": v.manufacturer,
            "model": model,
            "plant": {
                "country": "N/A",
                "city": "N/A",
                "code": "N/A"
            },
            "body_style": body_style,
            "engine": { "fuel": "N/A"},
            "country": v.country,
        }
    except Exception as e:
        return {
            "vin": vin,
            "valid": False,
            "year": False
        }

def decode_vin_b95(vin: str) -> dict:
    model = VINBrandPredictor()

    # yeah load model per request to be optimized later ;)
    model.load_model("vin_predictor/vin_brand_predictor")
    results = []
    predictions = model.predict([vin])
    for pred in predictions:
        results.append({
            "vin": pred['vin'],
            "brand": pred['Predicted_Brand'],
            "confidence": "{:.2f}".format(pred['Confidence'])
        })
    return results