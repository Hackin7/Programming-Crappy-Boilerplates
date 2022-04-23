# Run in Python 2
import SocketServer
import SimpleHTTPServer
import urllib
PORT = 9096

RHOST="https://chatty-bobcat-13.telebit.io"
class MyProxy(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        print(self.path, RHOST+self.path)
        url=self.path[1:]
        self.send_response(200)
        self.end_headers()
        self.copyfile(urllib.urlopen(RHOST+self.path), self.wfile)
        
httpd = SocketServer.ForkingTCPServer(('', PORT), MyProxy)
httpd.serve_forever()
