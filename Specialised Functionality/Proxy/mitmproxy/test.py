"""Redirect HTTP requests to another server."""
from mitmproxy import http
from mitmproxy import ctx

# Exposed URL
LSCHEME = "http"
LHOST = "localhost"
LPORT = 8080

# List of Hosts
RHOSTS = {
    "1" : ("https", "www.google.com", 443), 
    "2" : ("https", "chatty-bobcat-13.telebit.io", 443),
    "nextcloud" : ("http", "192.168.1.4", 8080),
}
RSCHEME, RHOST, RPORT = RHOSTS["1"]

def request(flow: http.HTTPFlow) -> None:
    global RSCHEME
    global RHOST
    global RPORT
    # Switching between endpoints
    if flow.request.path[:8] == "/switch/":
        selection = flow.request.path[8:]
        if selection in RHOSTS.keys():
            RSCHEME, RHOST, RPORT = RHOSTS[selection]
        flow.request.path = "/"
    # Directing to Endpoint
    if flow.request.pretty_host == LHOST:
        flow.request.scheme = RSCHEME
        flow.request.host = RHOST
        flow.request.port = RPORT

def response(flow: http.HTTPFlow):
    if flow.response.headers.get("location") != None:
        loc_replace = []
        final_url = f"{LSCHEME}://{LHOST}:{LPORT}/"
        loc_replace.append(("http://localhost/", final_url)) # Convert localhost to correct URL
        loc_replace.append((f"{RSCHEME}://{RHOST}/", final_url))
        loc_replace.append((f"{RSCHEME}://{RHOST}:{RPORT}/", final_url))
        
        
        for inloc, outloc in loc_replace:
            flow.response.headers["location"] = flow.response.headers["location"].replace(inloc, outloc)
