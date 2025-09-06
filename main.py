import os

from flask import Flask, render_template, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

from models import db, Car

load_dotenv()

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL', 
    'postgresql://user:password@localhost:5432/tjekvik_recommendation'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cars', methods=['GET'])
def cars():
    cars = Car.query.all()  # <-- Get all records
    return jsonify([{"id": c.id, "model": c.model, "brand": c.brand, "mileage": c.mileage} for c in cars])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)