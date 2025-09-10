
def scale_brand(brand: str) -> float | None:
    if brand is None:
        return None
    brand = str(brand).strip().title()
    brand_list = [
        "Toyota", "Honda", "Ford", "Chevrolet", "Nissan", "Volkswagen", "Hyundai",
        "Kia", "Mercedes-Benz", "BMW", "Audi", "Lexus", "Mazda", "Subaru", "Dodge",
        "Jeep", "GMC", "Ram", "Chrysler", "Buick"
    ]
    if brand in brand_list:
        index = brand_list.index(brand)
        return index / (len(brand_list) - 1)
    else:
        return None