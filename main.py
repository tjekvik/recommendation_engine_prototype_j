# from flask import Flask, render_template, jsonify
# from flask_sqlalchemy import SQLAlchemy
import dash
from dash import Dash, html, dcc
from dotenv import load_dotenv

load_dotenv()

# app = Flask(__name__)

# # Database configuration
# app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
#     'DATABASE_URL', 
#     'postgresql://user:password@localhost:5432/tjekvik_recommendation'
# )
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db.init_app(app)


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