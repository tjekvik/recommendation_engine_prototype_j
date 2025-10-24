#! /usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Before lower 12493
def exclude_vins(info: str):
    no_vins = [word.lower() for word in info.split() if len(word) != 17]
    no_km = [word for word in no_vins if not (word.endswith("km") or word.endswith("kw"))]
    no_ps = [word for word in no_km if not word.endswith("ps") or word.endswith("cv")]
    return " ".join(no_ps)

if __name__ == "__main__":
    df = pd.read_csv("data/brand_info_model/bim_all.csv")
    infos = df['info'].astype("str").apply(exclude_vins)


    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(infos)
    print(len(vectorizer.get_feature_names_out()))
    print(vectorizer.get_feature_names_out()[:1000])

