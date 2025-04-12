from main import app
from flask import render_template

@app.route("/")
def home():
   return render_template("home.html")

@app.route("/templates/rede1.html")
def rede1():
   return render_template("rede1.html")

@app.route("/templates/rede2.html")
def rede2():
   return render_template("rede2.html")

@app.route("/templates/variaveisRede1.html")
def variaveisRede1():
   return render_template("variaveisRede1.html")

@app.route("/templates/variaveisRede2.html")
def variaveisRede2():
   return render_template("variaveisRede2.html")