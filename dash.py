import dash
from dash import html, dcc, Dash
from dotenv import load_dotenv

load_dotenv()


app = Dash(use_pages=True)

app.layout = html.Div([
    html.H1('Recommendation Engine Prototype Presentation'),
    html.Div([
        html.Div(
            dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
        ) for page in dash.page_registry.values()
    ]),
    dash.page_container
])

# app.

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')