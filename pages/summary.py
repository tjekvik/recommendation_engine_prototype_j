import dash
from dash import html, dash_table, dcc, Input, Output, callback

dash.register_page(__name__)



layout = html.Div([
        html.H4("1. Data extraction and normalization researched with infrastructure ready."),
        html.Hr(),
        html.H4("2. Vector generation to be implemented."),
        html.H4("3. Currently working on: 1. Observability with elasticsearch 2. Service importing and vectorization 3. Trial integration with local Tjekvik"),
    ])