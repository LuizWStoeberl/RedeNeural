from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from RedeNeural.config import Config
from routes import bp as routes_bp
from RedeNeural.models import db

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
app.register_blueprint(routes_bp)

def cria_tabelas():
    with app.app_context():
        db.create_all()

if __name__ == "__main__":
    cria_tabelas()
    app.run(debug=True)
