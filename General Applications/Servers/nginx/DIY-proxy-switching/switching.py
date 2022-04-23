import flask
import os

def switch_nginx_conf(infile):
    os.system("echo Switching NGINX conf")
    os.system(f"cp {infile} nginx.conf")
    restart_nginx()
    
def restart_nginx():
    os.system("sudo docker-compose restart")

app = flask.Flask(__name__)

@app.route('/')
def index():
    return 'Hello World'

@app.route('/aa')
def aa():
    switch_nginx_conf("aa.conf")
    return 'aa'
    
@app.route('/bb')
def bb():
    switch_nginx_conf("bb.conf")
    return 'bb'
    

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)
