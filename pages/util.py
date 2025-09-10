import os, requests
from dash import html
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
    
def car_info_card(car_data: dict):
    year = car_data.get("year", "N/A")
    make = car_data.get("make", "N/A")
    model = car_data.get("model", "N/A")
    vin = car_data.get("vin", "N/A")
    valid = car_data.get("valid", False)
    manufacturer = car_data.get("manufacturer", "N/A")
    body_style = car_data.get("bodyStyle", "N/A")
    fuel_type = car_data.get("fuelType", "N/A")
    doors = car_data.get("doors", "N/A")
    plant = car_data.get("plant", {})
    plant_country = plant.get("country", "N/A")
    plant_city = plant.get("city", "N/A")
    plant_code = plant.get("code", "N/A")
    engine = car_data.get("engine", {})
    engine_fuel = engine.get("fuel", "N/A")

    validity_element = html.Li(f"Valid VIN: YES", style={'color': 'green', 'font-weight': 'bold'}) if valid else html.Li("Valid VIN: No", style={'color': 'red', 'font-weight': 'bold'})

    return html.Div(
        className="car-info-card",
        children=[
            html.H3(f"{year} {make} {model}"),
            html.Ul([
                html.Li(f"VIN: {vin}"),
                validity_element,
                html.Li(f"Manufacturer: {make}"),
                html.Li(f"Model: {model}"),
                html.Li(f"Year: {year}"),
                html.Li(f"Body Style: {body_style}"),
                html.Li(f"Fuel Type: {fuel_type}"),
                html.Li(f"Doors: {doors}"),
                html.Li(f"Engine: {engine_fuel}"),
                html.Li(f"Plant: {plant_city}, {plant_country} (Code: {plant_code})"),
            ])
        ]
    )