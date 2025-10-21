#! /usr/bin/env python3
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from info_extraction.info_tagger import InfoTagger

class Extractor:
    def __init__(self):
        self.tagger = InfoTagger()

    def extract(self, car_info):
        tags = self._tags(car_info)
        return {
            "make": self._find_tag(tags, "MAKE"),
            "model": self._find_tag(tags, "MODEL"),
            "year": self._find_tag(tags, "YEAR"),
            "body": self._find_tag(tags, "BODY"),
            "door": self._find_tag(tags, "DOOR"),
        }
    
    def extract_model(self, car_info):
        return self.extract(car_info)["model"]

    def _find_tag(self, tags, target_tag):
        return next((word for word, tag in tags if tag == target_tag), None)

    def _tags(self, car_info: str) -> list:
        tokens = word_tokenize(car_info)
        tags = self.tagger.tag(tokens)

        return tags

if __name__ == "__main__":
    nltk.download('punkt_tab')
    print("Info extraction")
    extractor = Extractor()

    df = pd.read_csv("data/brand_info_model/bim_all_oct.csv")
    df['info'] = df['info'].apply(lambda x: str(x))
    df['model'] = df['info'].apply(extractor.extract_model)
    df = df[df['model'].notna()]
    df.to_csv("data/brand_info_model_extracted/bim_all.csv", index = False)
    print(df['model'].value_counts())
