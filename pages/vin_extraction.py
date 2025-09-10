import dash
from dash import html, dash_table, dcc
import pandas as pd
import plotly.express as px

dash.register_page(__name__)

def preface():
    return([
        html.Div(
            className="app-header",
            children=[
                html.Div('VIN Extaction Approach', className="app-header--title")
            ]
        ),
        html.Div(
            children=html.Div([
                html.H1('Preface'),
                html.Div('''
                    I have found an offline tool for VIN data extration. which is part of this dockerized environment- to see the data extraction, lets first go to the demo page.
                '''),
            ])
        ),
    ])

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

layout = [*preface(), *cars_chapter()]