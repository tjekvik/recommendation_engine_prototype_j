from nltk import RegexpTagger
from info_extraction.models import CAR_MODELS
from info_extraction.makes import CAR_MAKES

class InfoTagger(RegexpTagger):
    BASE_PATTERNS = [
        (r"(?i)(\-)*\d\-door(s)*", "DOOR"), # DOOR COUNT
        (r"[a-zA-Z0-9]{17}", "VIN"), # VIN
        (r"^(20[0-2]\d)\.*\d*\d*$", "YEAR"), # YEAR OF MANUFACTURE
        (r"^(19\d{2})\.*\d*\d*$", "YEAR"), # YEAR OF MANUFACTURE
        ((r'.*', 'INF') ) # UNIDENTIFIED INFO
    ]

    CAR_BODY_PATTERNS = [
        (r"(?i)SEDAN", "BODY"), # CAR BODY TYPE
        (r"(?i)SUV", "BODY"), # CAR BODY TYPE
        (r"(?i)HATCHBACK", "BODY"), # CAR BODY TYPE
        (r"(?i)HATCH", "BODY"), # CAR BODY TYPE
        (r"(?i)COUP[Eé]{1}", "BODY"), # CAR BODY TYPE
        (r"(?i)WAGON", "BODY") # CAR BODY TYPE
    ]

    FUEL_PATTERNS = [
        (r"(?i)benzyna", "FUEL"),
        (r"(?i)diesel", "FUEL"),
        (r"(?i)hybrid", "FUEL")
    ]




    def __init__(self):
        make_pattern = [ (f"(?i){make}", "MAKE") for make in CAR_MAKES ]
        model_pattern = [ (f"(?i)^{model}$", "MODEL") for model in CAR_MODELS ]
        patterns = make_pattern + model_pattern + InfoTagger.CAR_BODY_PATTERNS + InfoTagger.BASE_PATTERNS
        super().__init__(patterns)

