from cs50 import SQL
from flask import  Flask, flash, redirect, render_template, request, session, jsonify
from flask_session import Session
from tempfile  import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
scalar = StandardScaler
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
    return render_template('layout.html')

@app.route('/login', methods=['GET','POST'])
def login():
    session.clear() 
    
    if request.method == "GET" : 
        return render_template("login.html")
    else:
        print("here in login route") 
        email = request.form.get("email")
        password = request.form.get("password")
        rows = db.execute("SELECT * FROM users WHERE email = :email", email = email)
        if len(rows) == 0:
            return render_template("sorry.html", text = "Invalid User")
        if not check_password_hash(rows[0]["pasword"], password):
            return render_template("sorry.html", text="Invalid User")
        session["user_id"] = rows[0]["Id"]
        print(rows[0]["Id"])
        return redirect("/")

@app.route("/prediction", methods=["GET","POST"])
def prediction():
    knn_model = pickle.load(open('production/knn_model.pkl', 'rb'))
    if request.method == "GET":
        print("we are in pred")
        return render_template("predictionForm.html")
    else:
        print("we are in prediction with post")
        features = [float(x) for x in request.form.values()]
        final_features = np.asarray(features).reshape(1, -1)
        prediction = knn_model.predict(final_features)
        output = round(prediction[0], 2)
        print("final features",output)
        if output == 0:
            return render_template('Heart Disease Classifier.html',result = 'The patient is not likely to have heart disease!')
        else:
            return render_template('Heart Disease Classifier.html',result = 'The patient is likely to have heart disease!')
        
@app.route("/logout", methods = ["GET","POST"])
# @login_required
def logout():
    session.clear()
    return redirect("/")


@app.route("/register", methods=["GET", "POST"])
def register():
    print("I am in register route")
    if request.method == "GET":
        return render_template("register.html") 
    else : 
        # TODO : Add register functionality here
        username = request.form.get("username")
        emailId = request.form.get("emailId")
        password = request.form.get("password")
        confirmation = request.form.get("confirm")
        contactNo = request.form.get("contactNo")
        if not username or not emailId or not password or not confirmation or not contactNo :
            return render_template("sorry.html", text="Please Enter complete details")
        if password != confirmation:
            return render_template("sorry.html", text = "Password doesn't match .. !!")
        user = db.execute("SELECT * FROM users WHERE username=:username", username = username)
        if len(user) != 0:
            return render_template("sorry.html", text = "Username Already Exists")
        email = db.execute("SELECT * FROM users WHERE email=:email", email = emailId)
        if len(email) != 0:
            return render_template("Email Already Used")
        hashed = generate_password_hash(password)
        db.execute("INSERT INTO users (username, email, pasword, contactNo ) VALUES(:username, :emailId, :pasword, :contactNo)", username = username, emailId = emailId, pasword = hashed, contactNo = contactNo)
        return redirect("/")




if __name__ == "__main__":
    app.run(debug = True)

#source venv/scripts/activate
#export FLASK_APP=application.py
#flask run