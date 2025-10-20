from info_tagger import InfoTagger

def test_toyota_yaris():
    tagger = InfoTagger()
    result = tagger.tag(["Toyota", "Yaris", "2005"])

    assert ("Toyota", "MAKE") in result
    assert ("2005", "YEAR") in result

def test_toyota_yaris_caps():
    tagger = InfoTagger()
    result = tagger.tag(["TOYOTA", "Yaris", "2005"])

    assert ("TOYOTA", "MAKE") in result
    assert ("2005", "YEAR") in result

def test_toyota_yaris_future():
    tagger = InfoTagger()
    result = tagger.tag(["TOYOTA", "Yaris", "3005"])

    assert ("TOYOTA", "MAKE") in result
    assert ("3005", "YEAR") not in result

def test_mini_door():
    tagger = InfoTagger()
    result = tagger.tag(["MINI", "F56", "MINI", "3-door", "Hatch", "-i3/Clubman"])

    assert ("MINI", "MAKE") in result
    assert ("3-door", "DOOR") in result

def test_bmw_door():
    tagger = InfoTagger()
    result = tagger.tag(["BMW", "F40", "1", "Series", "-5-door", "Sportshatch"])

    assert ("BMW", "MAKE") in result
    assert ("-5-door", "DOOR") in result

def test_mitsubishi_outlander_body():
    tagger = InfoTagger()
    result = tagger.tag(["MITSUBISHI", "OUTLANDER", "WAGON", "2024"])

    assert ("MITSUBISHI", "MAKE") in result
    assert ("WAGON", "BODY") in result
    assert ("2024", "YEAR") in result

def test_kia_coupe_weird_year():
    tagger = InfoTagger()
    result = tagger.tag(["KIA", "OUTLANDER", "Coupé", "2024.75"])

    assert ("KIA", "MAKE") in result
    assert ("Coupé", "BODY") in result
    assert ("2024.75", "YEAR") in result

def test_seat_future_year():
    tagger = InfoTagger()
    result = tagger.tag(["Seat", "Ibiza", "2030"])

    assert ("Seat", "MAKE") in result
    assert ("2030", "YEAR") not in result
