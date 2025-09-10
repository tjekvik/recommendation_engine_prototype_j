import os
import requests
import dash
from dash import html, dash_table, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px

from pages.util import decode_vin_corgi, decode_vin_vininfo

# dash.register_page(__name__)

def corgi_success(vin: str) -> int:
    result = decode_vin_corgi(vin)
    return 1 if result.get("valid", False) else None

def vininfo_success(vin: str) -> int:
    result = decode_vin_vininfo(vin)
    return 1 if result.get("valid", False) else None

def corgi_success_year(vin: str) -> int:
    result = decode_vin_corgi(vin)
    return 1 if result.get("year", False) else None

def vininfo_success_year(vin: str) -> int:
    result = decode_vin_vininfo(vin)
    return 1 if result.get("year", False) else None

def corgi_success_model(vin: str) -> int:
    result = decode_vin_corgi(vin)
    return 1 if result.get("model", False) else None

def vininfo_success_model(vin: str) -> int:
    result = decode_vin_vininfo(vin)
    return 1 if result.get("model", False) else None


def success_rate_chart(df):
    df['corgi_success'] = df['vin'].apply(corgi_success)
    df['vininfo_success'] = df['vin'].apply(vininfo_success)
    df = df.groupby('brand', as_index=False).count()
    df['vinfo_success_rate'] = df['vininfo_success'] / df['id']
    df['corgi_success_rate'] = df['corgi_success'] / df['id']
    return html.Div([
        dcc.Graph(figure=px.bar(df, x="brand", y='vinfo_success_rate', title='Comparison of successful decodings VININFO').update_xaxes(categoryorder='total ascending')),
        dcc.Graph(figure=px.bar(df, x="brand", y='corgi_success_rate', title='Comparison of successful decodings CORGI').update_xaxes(categoryorder='total ascending')),
        
        ])

def success_rate_chart_year(df):
    df['corgi_success'] = df['vin'].apply(corgi_success_year)
    df['vininfo_success'] = df['vin'].apply(vininfo_success_year)
    df = df.groupby('brand', as_index=False).count()
    df['vinfo_success_rate'] = df['vininfo_success'] / df['id']
    df['corgi_success_rate'] = df['corgi_success'] / df['id']
    return html.Div([
        dcc.Graph(figure=px.bar(df, x="brand", y='vinfo_success_rate', title='Comparison of successful decodings VININFO Fetching Year').update_xaxes(categoryorder='total ascending')),
        dcc.Graph(figure=px.bar(df, x="brand", y='corgi_success_rate', title='Comparison of successful decodings CORGI Fetching Year').update_xaxes(categoryorder='total ascending')),
        
        ])

def success_rate_chart_model(df):
    df['corgi_success'] = df['vin'].apply(corgi_success_model)
    df['vininfo_success'] = df['vin'].apply(vininfo_success_model)
    df = df.groupby('brand', as_index=False).count()
    df['vinfo_success_rate'] = df['vininfo_success'] / df['id']
    df['corgi_success_rate'] = df['corgi_success'] / df['id']

    return html.Div([
        dcc.Graph(figure=px.bar(df, x="brand", y='vinfo_success_rate', title='Comparison of successful decodings VININFO Fetching Model').update_xaxes(categoryorder='total ascending')),
        dcc.Graph(figure=px.bar(df, x="brand", y='corgi_success_rate', title='Comparison of successful decodings CORGI Fetching Model').update_xaxes(categoryorder='total ascending')),
        
        ])

# layout = html.Div([
#     html.H1('VIN Tools Benchmark'),
#     html.Div('''
#         This page benchmarks different VIN decoding tools and libraries to evaluate their accuracy and reliability.
#     '''),
#     html.H1('Dataset Overview'),
#     success_rate_chart(pd.read_csv('/app/data/tj_prod_cars_5000.csv')),
#     success_rate_chart_year(pd.read_csv('/app/data/tj_prod_cars_5000.csv')),
#     success_rate_chart_model(pd.read_csv('/app/data/tj_prod_cars_5000.csv')),
# ])