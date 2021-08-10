from cs50 import SQL
from flask import Flask, redirect, render_template
from flask_session import Session
from tempfile  import mkdtemp




app = Flask(__name__)

app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

db = SQL("sqlite:///mydb.db")


@app.route('/', methods=['GET']) 
def home() : 
    return redirect("/login")

@app.route('/login', methods=['GET','POST'])
def login():
    return render_template("login.html")