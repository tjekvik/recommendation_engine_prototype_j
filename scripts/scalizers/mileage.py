def scale_mileage(mileage: int) -> float | None:
    if mileage is None:
        return None
    try:
        mileage = int(mileage)
        if mileage < 0:
            return None
        # Scale mileage to a 0-1 range assuming max mileage of 300,000 km
        return min(mileage / 300000, 1.0)
    except ValueError:
        return None