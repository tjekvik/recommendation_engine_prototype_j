def unify_brand_name(brand: str) -> str:
    audi_aliases = ['a', 'wau', '#au']
    byd_aliases = ['by']
    cadillac_aliases = ['cad']
    bmw_aliases = ["b", "bmw cars", "bmw_motorrad", "bmwi", 'bm', 'bw', 'm5', 'l', 'bmw i', 'bmw motorrad', 'bmw_i'] # m5 most likely refers to BMW m5 L are lagerkraftwagen Caddy etc
    citroen_aliases = ['citroën', 'ci']
    ford_aliases = ['fo']
    honda_aliases = ['hk', 'ho']
    fiat_aliases = ['fi']
    hyundai_aliases = ['hy']
    kia_aliases = ['k', 'kg', 'ki']
    mercedes_aliases = ["wdb","mb", "mb pkw","mercedes", "mercedes-benz", "mercedes turismos", "mercedes-benz automóveis", "mercedes-benz personbil", "mercedes-benz varebil", "mercedes-benz cars", 'mercedes benz passenger cars',
    'mercedes-benz lastbil', 'mercedes-benz pkw', 'mer']
    mg_aliases = ['mg motor uk'] # few records under 'm' as well but m is mostly volkswagen
    nissan_aliases = ['n']
    opel_aliases = ['op']
    seat_aliases = ['s','sea'] # some dealerships use S for Cupra as well
    porsche_aliases = ['p'] # porsche mostly
    skoda_aliases = ['škoda', 'c'] # c mostly skoda with some VW as well
    subaru_aliases = ['sh']
    peugeot_aliases = ['pe', 'pg', 'peu']
    renault_aliases = ['rn']
    volkswagen_aliases = ["v", "w", "vw trp", "vw", "vw pkw","wvw", 'm', 'vw nutzfahrzeuge', 'volkswagen commercials', 'volkswagen cars', 'wvg', 'va', 'volkswagen comerciales'] # few Vauxhall but not many under v
    volvo_aliases = ["vo"]
    toyota_aliases = ['to']
    if brand in bmw_aliases:
        return "bmw"
    elif brand in byd_aliases:
        return "byd"
    elif brand in cadillac_aliases:
        return "cadillac"
    elif brand in mercedes_aliases:
        return "mercedes benz" 
    elif brand in volkswagen_aliases:
        return "volkswagen"
    elif brand in audi_aliases:
        return 'audi'
    elif brand in citroen_aliases:
        return 'citroen'
    elif brand in toyota_aliases:
        return 'toyota'
    elif brand in ford_aliases:
        return 'ford'
    elif brand in hyundai_aliases:
        return 'hyundai'
    elif brand in honda_aliases:
        return 'honda'
    elif brand in kia_aliases:
        return 'kia'
    elif brand in mg_aliases:
        return 'mg'
    elif brand in nissan_aliases:
        return 'nissan'
    elif brand in opel_aliases:
        return 'opel'
    elif brand in skoda_aliases:
        return 'skoda'
    elif brand in peugeot_aliases:
        return 'peugeot'
    elif brand in seat_aliases:
        return 'seat'
    elif brand in renault_aliases:
        return 'renault'
    elif brand in volvo_aliases:
        return 'volvo'
    elif brand in porsche_aliases:
        return 'porsche'
    elif brand in fiat_aliases:
        return 'fiat'
    elif brand in subaru_aliases:
        return 'subaru'
    else:
        return brand
    