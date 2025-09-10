import os
import requests
import dash
from dash import html, dash_table, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px

from pages.util import decode_vin_corgi, decode_vin_vininfo, car_info_card

dash.register_page(__name__)

def car_data_header():
    df = pd.read_csv('/app/data/tj_prod_cars_5000.csv')

    return([
        html.Div(
            children=html.Div([
                html.H1('Overview'),
                html.Div('''
                    Basic car 🚗 data below
                ''')
            ])
        ),
        dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    ])

def vin_extraction_form():
    return html.Div([
        html.H6("Change the value in the text box to see callbacks in action!"),
        html.Div([
            "Input: ",
            dcc.Input(id='vin-input', value='Paste VIN here', type='text')
        ]),
        html.Br(),
        html.Div(id='car-stats'),

    ])


@callback(
    Output(component_id='car-stats', component_property='children'),
    Input(component_id='vin-input', component_property='value')
)
def update_output_div(input_value):
    df = pd.read_csv('/app/data/tj_prod_cars_5000.csv')
    tjekvik_df = df.loc[df['vin'] == input_value]
    if tjekvik_df.empty:
        return html.Div("No car found with the provided VIN.", style={'color': 'red', 'font-weight': 'bold'})
    
    return html.Div([
        html.H4("Car Data from Tjekvik Dataset:"),
        dash_table.DataTable(data=tjekvik_df.to_dict('records'), page_size=1),
        html.Hr(),
        html.H4("VIN Decoding Result from CORGI 🐶:"),
        car_info_card(decode_vin_corgi(input_value)),
        html.H4("VIN Decoding Result from VINFO lib 🐍:"),
        car_info_card(decode_vin_vininfo(input_value))
    ])

layout = [*car_data_header(), vin_extraction_form()]
