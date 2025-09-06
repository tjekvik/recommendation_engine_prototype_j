from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Simple model example
class Car(db.Model):
    __tablename__ = 'cars'
    id = db.Column(db.Uuid, primary_key=True)
    model = db.Column(db.String(100), nullable=False)
    mileage = db.Column(db.Integer, nullable=False)
    brand = db.Column(db.String(50), nullable=False)