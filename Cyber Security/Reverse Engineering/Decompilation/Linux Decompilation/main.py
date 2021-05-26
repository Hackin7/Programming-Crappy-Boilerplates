### Decompilation ##############################################################
import os
APP_PATH = "/root"
PORT = os.environ.get("PORT")
if PORT == None:
    PORT = 8000
    print("Default port used, which is",PORT)

def retdec(file_path, folder_path, filename, output_path):
    os.system(f"rm -rf {folder_path}")
    os.system(f"mkdir {folder_path}")
    os.system(f"mv {file_path} {folder_path}/{filename}")
    os.system(f"{APP_PATH}/retdec/bin/retdec-decompiler.py {folder_path}/{filename} > {folder_path}/retdec_output.txt")
    os.system(f"zip -r {output_path} {folder_path}")
    return output_path

# https://reverseengineering.stackexchange.com/questions/21207/use-ghidra-decompiler-with-command-line
def ghidra(file_path, output_path):
    os.system("rm -rf /tmp/ghidra_test.gpr")
    os.system("rm -rf /tmp/ghidra_test.rep")
    os.system(f"{APP_PATH}/ghidra/support/analyzeHeadless /tmp ghidra_test -import {file_path} -postScript /root/ghidra_post_script.py > {output_path}")
    return output_path

### Web App ####################################################################
import flask
app = flask.Flask(__name__)

os.system("mkdir /tmp/decompilation")
@app.route("/", methods=["GET"])
def page():
    return """
    <title>Online C Decompiler</title>
    <h1>Online Decompiler</h1>
    <p>Decompile Linux Binary/ exe files into pseudo C language</p>
    <p>Made by <a href='https://hackin7.github.io/'>Hackin7</a></p>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" id="file">
        <select name="decompiler" id="decompiler">
          <option value="ghidra">Ghidra</option>
          <option value="retdec">Retdec</option>
        </select>
        <input type="submit"/>
    </form>
    """

from werkzeug.utils import secure_filename
@app.route("/upload", methods=["POST"])
def upload():
    if flask.request.files:
        file = flask.request.files['file']
        filename = secure_filename(file.filename)
        if filename == "":
            return "Nothing uploaded"
        file_path = '/tmp/'+filename
        file.save(file_path)


        if flask.request.form['decompiler'] == 'ghidra':
            output_directory = '/tmp/'
            output_file = 'decompiled.txt'
            ghidra(file_path, output_directory + output_file)
        elif flask.request.form['decompiler'] == 'retdec':
            working_path = "/tmp/decompilation"
            output_directory = '/tmp/'
            output_file = 'decompiled.zip'
            retdec(file_path, working_path, filename, output_directory + output_file)
        else:
            return 'Decompiler does not exist as of the moment'
        return flask.send_from_directory(output_directory,output_file)
        #return "Uploaded "+file.filename
    else:
        return "Nothing uploaded"

@app.route("/download", methods=["GET"])
def download():
    return flask.send_from_directory('./','file_upload_download.py')

app.run('0.0.0.0', port=PORT)#, debug=True)
