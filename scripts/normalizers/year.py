from pyvin import VIN

def normalize_year(vin: str) -> int | None:
    try:
        vehicle = VIN(vin)
        return vehicle.year
    except Exception:
        return None