from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config
from routes import bp as routes_bp
from models import db

app = Flask(__name__)
app.config.from_object(Config)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Para desabilitar o rastreamento de modificações, o que evita warnings.

app.secret_key = 'ioio2014'

db.init_app(app)
app.register_blueprint(routes_bp)

def cria_tabelas():
    with app.app_context():
        db.create_all()

if __name__ == "__main__":
    cria_tabelas()
    app.run(debug=True)
