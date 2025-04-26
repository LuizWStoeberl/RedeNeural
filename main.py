from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config
from routes import bp as routes_bp
from models import db
from cores import cores

app = Flask(__name__)
app.config.from_object(Config)

app = Flask(__name__)
app.register_blueprint(bp_uploads)
app.register_blueprint(bp_cores)

db.init_app(app)
app.register_blueprint(routes_bp)

def cria_tabelas():
    with app.app_context():
        db.create_all()

if __name__ == "__main__":
    cria_tabelas()
    app.run(debug=True)
