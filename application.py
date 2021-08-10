from cs50 import SQL
from flask import  Flask, flash, redirect, render_template, request, session, jsonify
from flask_session import Session
from tempfile  import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash



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
    if request.method == "GET":
        print("we are in pred")
        return render_template("predictionForm.html")
    else:
        print("we are in prediction with post")
        predictionData = []
        age = int(request.form.get("age"))
        predictionData.append(age) 
        gender = int(request.form.get("Gender"))
        #print(age,gender)
        predictionData.append(gender)
        #print("Varun kalra was here")
        #print(predictionData)
        cp = int(request.form.get("cp"))
        predictionData.append(cp)
        rbp = int(request.form.get("trestbps"))
        predictionData.append(rbp)
        chol = int(request.form.get("chol"))
        predictionData.append(chol)
        fbs = int(request.form.get("fbs"))
        predictionData.append(fbs)
        recg = int(request.form.get("recg"))
        predictionData.append(recg)
        mhr = int(request.form.get("mhr"))
        predictionData.append(mhr)
        dind = int(request.form.get("dind"))
        predictionData.append(dind)
        ia = int(request.form.get("ia"))
        predictionData.append(ia)
        slope = int(request.form.get("slope"))
        predictionData.append(slope)
        cf = int(request.form.get("cf"))
        predictionData.append(cf)
        thales = int(request.form.get("thales"))
        predictionData.append(thales)
        print(predictionData)
        #predictionData is the list in which form values are present
        return render_template("home.html", pred = predictionData)
    
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