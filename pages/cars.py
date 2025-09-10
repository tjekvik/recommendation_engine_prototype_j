import pandas as pd
import plotly.express as px
import dash
from dash import html, dash_table, dcc

dash.register_page(__name__)

def cars_chapter():
    df = pd.read_csv('/app/data/tj_prod_cars_5000.csv')

    return([
        html.Div(
            className="app-header",
            children=[
                html.Div('Recommendation Engine', className="app-header--title")
            ]
        ),
        html.Div(
            children=html.Div([
                html.H1('Overview'),
                html.Div('''
                    Basic car 🚗 data below
                ''')
            ])
        ),
        dash_table.DataTable(data=df.to_dict('records'), page_size=10),
        html.Div(
            className="app-header",
            children=[
                html.Div('Car Brands 🚗 / 🚙', className="app-header--title")
            ]
        ),
        dcc.Graph(figure=px.histogram(df, x="brand", histfunc='count', title='Number of Cars by Brand').update_xaxes(categoryorder='total descending')),
        html.Div(
            className="app-header",
            children=[
                html.Div('Car Models', className="app-header--title")
            ]
        ),
        dcc.Graph(figure=px.histogram(df, x="model", histfunc='count', title='Number of Cars by Model').update_xaxes(categoryorder='total descending')),
        html.Div(
            className="app-header",
            children=[
                html.Div('Car Mileage 🛣️', className="app-header--title")
            ]
        ),
        dcc.Graph(figure=px.histogram(df, x="mileage", histfunc='count', title='Car mileage distribution', nbins=80)),
        html.Div('''
            The unnormalized data about cars is very fragmented. First step in performing the analysis is to normalize the data.
        ''', className="app-paragraph"),
    ])

def cars_normalized_chapter():
    df = pd.read_csv('/app/data/tj_prod_cars_5000_normalized.csv')

    return([
        html.Div(
            children=html.Div([
                html.H1('Step 1: Data Normalization'),
                html.Div('''
                    Performed on car data
                ''')
            ])
        ),
        html.Div(
            className="app-header",
            children=[
                html.Div('Normalized Car Brands 🚗 / 🚙', className="app-header--title")
            ]
        ),
        dcc.Graph(figure=px.histogram(df, x="normalized_brand", histfunc='count', title='Number of Cars by Normalized Brand').update_xaxes(categoryorder='total descending')),
        html.Div(
            className="app-header",
            children=[
                html.Div('Normalized Car Models', className="app-header--title")
            ]
        ),
        dcc.Graph(figure=px.histogram(df, x="normalized_model", histfunc='count', title='Number of Cars by Model').update_xaxes(categoryorder='total descending')),
        html.Div(
            className="app-header",
            children=[
                html.Div('Normalized Car Mileage 🛣️', className="app-header--title")
            ]
        ),
        dcc.Graph(figure=px.histogram(df, x="normalized_mileage", histfunc='count', title='Normalized Car mileage distribution', nbins=80)),
        html.Div('''
            The normalized data is much more consistent and almost ready for analysis.
            However, pruning and norlizing data further proved to be challanging task - It was decided to change approach.
        ''', className="app-paragraph"),
    ])

layout = [*cars_chapter(), *cars_normalized_chapter()]