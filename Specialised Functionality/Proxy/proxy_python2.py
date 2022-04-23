# https://levelup.gitconnected.com/how-to-build-a-super-simple-http-proxy-in-python-in-just-17-lines-of-code-a1a09192be00
# Run in Python 2
import SocketServer
import SimpleHTTPServer
import urllib
PORT = 9097

class MyProxy(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        url=self.path[1:]
        self.send_response(200)
        self.end_headers()
        self.copyfile(urllib.urlopen(url), self.wfile)
        
httpd = SocketServer.ForkingTCPServer(('', PORT), MyProxy)
httpd.serve_forever()
