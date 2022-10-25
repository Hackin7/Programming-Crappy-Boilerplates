import flask
app = flask.Flask(__name__)

@app.route('/')
def main():
    return flask.render_template('form.html')

@app.route('/table', methods=['GET'])
def table():
    form_stuff = flask.request.args['textfield']
    # for POST use flask.request.form
    
    # Templating
    headers = ['Name','Value']
    data = [
        ['Text Field', form_stuff]
    ]
    return flask.render_template('table.html',
                                 headers=headers,
                                 data=data)
    
@app.route('/quit')
def quit():
    exit()

if __name__ == "__main__":
    app.run('0.0.0.0', 3000, debug=True)
