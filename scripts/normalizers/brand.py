import jellyfish

BRANDS = [
    "Toyota", "Ford", "Chevrolet", "Honda", "Nissan", "BMW", "Mercedes-Benz",
    "Volkswagen", "Audi", "Hyundai", "Kia", "Mazda", "Subaru", "Lexus",
    "Jeep", "Dodge", "Ram", "GMC", "Cadillac", "Buick", "Chrysler",
    "Volvo", "Jaguar", "Land Rover", "Porsche", "Ferrari", "Lamborghini",
    "Mitsubishi", "Infiniti", "Acura", "Lincoln", "Mini", "Fiat", "Alfa Romeo",
    "Tesla", "Suzuki", "Saab", "Renault", "Peugeot", "Citroën",
    "Skoda", "Seat", "Opel", "Vauxhall", "Dacia", "Tata", "Mahindra",
    "Isuzu", "Hino", "Maserati", "Bentley", "Rolls-Royce", "Aston Martin",
    "McLaren", "Bugatti", "Pagani", "Koenigsegg", "Genesis", "Rivian", "Lucid", "Chevy"
]

def normalize_brand(brand: str) -> str:
    """
    Normalize the brand name to a standard format.
    If the brand is not recognized, return 'Unknown'.
    """
    brand = str(brand).strip().replace(' ', '').lower()
    if brand in BRANDS:
        return brand
    
    # Fuzzy matching for slight misspellings
    highest_ratio = 0
    best_match = "Unknown"
    for known_brand in BRANDS:
        ratio = jellyfish.jaro_similarity(brand, known_brand.lower())
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = known_brand
    
    # Consider it a match if similarity is above a threshold (e.g., 0.8)
    if highest_ratio >= 0.8:
        return best_match
    
    return "Unknown"