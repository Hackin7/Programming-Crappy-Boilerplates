# https://flask-login.readthedocs.io/en/latest/

# Derived from https://github.com/shekhargulati/flask-login-example/blob/master/flask-login-example.py
import flask
from flask import Response, redirect, url_for, request, session, abort
from flask_login import LoginManager, UserMixin, \
                               current_user, login_required, login_user, logout_user 
import user_management as um

user_manager = um.UserManager()

app = flask.Flask(__name__)

# config
app.config.update(
    DEBUG = True,
    SECRET_KEY = 'secret_xxx'
)

# flask-login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

### Login System : Form Based ##########################################

# some protected url
@app.route('/')
@login_required
def home():
    return Response("Hello World!")


LOGIN_SECTION = '/login_system' 
# somewhere to login
@app.route(LOGIN_SECTION + "/login", methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = user_manager.get(username, password)   
        if user != None:
            login_user(user)
            return redirect(url_for("home"))
        else:
            return abort(401)
    else:
        return flask.render_template('login.html')

@app.route(LOGIN_SECTION + '/register' , methods = ['GET' , 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_manager.register(username, password)
        return Response("Registered Successfully")
    else:
        return flask.render_template('register.html')

# somewhere to logout
@app.route(LOGIN_SECTION + "/logout")
@login_required
def logout():
    logout_user()
    return Response('<p>Logged out</p>')


# handle login failed
@app.errorhandler(401)
def page_not_found(e):
    return Response('<p>Login failed</p>')
    
    
# callback to reload the user object        
@login_manager.user_loader
def load_user(username):
    return user_manager.get_session(username)

### Login System : GET Based ##########################################
@app.route(LOGIN_SECTION + "/get/user")
def get_user():
    return str(current_user)

### Running the Program #################################### 
if __name__ == "__main__":
    app.run()
