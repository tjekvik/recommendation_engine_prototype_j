from brand import normalize_brand


def test_bmw():
    assert normalize_brand('BMW') == "BMW"
    assert normalize_brand('bmw') == "BMW"
    assert normalize_brand('Bmw') == "BMW"
    assert normalize_brand(' bmw ') == "BMW"
    assert normalize_brand('B M W') == "BMW"

def test_toyota():
    assert normalize_brand('Toyota') == "Toyota"
    assert normalize_brand('toyota') == "Toyota"
    assert normalize_brand('TOYOTA') == "Toyota"
    assert normalize_brand('Tayota') == "Toyota"
    assert normalize_brand('Toyta') == "Toyota"
    assert normalize_brand('Toyoya') == "Toyota"