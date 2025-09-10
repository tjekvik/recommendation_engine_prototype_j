import dash
from dash import html, dash_table, dcc, Input, Output, callback

dash.register_page(__name__)

layout = html.Div([
        html.H4("1. Investigated PostgresSQL vector extensions: pgvector seems to be easiest and aligned with our needs."),
        html.Hr(),
        html.H4("2. Vector generation to be done from the normalized car data."),
        html.H4("3. Service and their archetypes - potential use for LLMs"),
    ])