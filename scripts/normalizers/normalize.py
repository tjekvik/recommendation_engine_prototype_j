#!/usr/bin/env python3

from brand import normalize_brand
from mileage import normalize_mileage
from model import normalize_model
from year import normalize_year
import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('../data/tj_prod_cars_5000.csv')
    model_list = pd.read_csv('../data/tj_prod_car_models_over_20.csv')['model'].tolist()
    df['normalized_brand'] = df['brand'].apply(normalize_brand)
    df['normalized_mileage'] = df.apply(lambda row: normalize_mileage(row['mileage'], row.get('mileage_unit', 'km')), axis=1)
    df['normalized_model'] = df['model'].apply(lambda x: normalize_model(x, model_list))
    df['normalized_year'] = df['vin'].apply(normalize_year)
    print("Brand Normalization DONE:")
    print(df[['brand', 'normalized_brand']].head(10))
    print("Mileage Normalization DONE:")
    print(df[['mileage', 'normalized_mileage']].head(10))

    df.to_csv('../data/tj_prod_cars_5000_normalized.csv', index=False)
