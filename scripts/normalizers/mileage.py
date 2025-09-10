

def normalize_mileage(mileage, unit: str = "km") -> int | None:
    if mileage is None:
        return None
    try:
        mileage = int(mileage)
        if mileage < 0:
            return None
        
        match str(unit).lower():
            case "km":
                return mileage
            case "miles" | "mi":
                return int(mileage * 1.60934)  # Convert miles to kilometers
            case "hours" | "h":
                # Assuming average speed of 60 km/h for conversion
                return int(mileage * 60)
            case _:
                return 0
    except ValueError:
        return None