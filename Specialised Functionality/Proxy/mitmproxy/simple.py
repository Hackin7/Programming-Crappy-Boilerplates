"""Redirect HTTP requests to another server."""
from mitmproxy import http
from mitmproxy import ctx

LHOST="localhost"


def request(flow: http.HTTPFlow) -> None:
    # pretty_host takes the "Host" header of the request into account,
    # which is useful in transparent mode where we usually only have the IP
    # otherwise.
    ctx.log.info(flow.request.pretty_host)
    ctx.log.info(flow.request.path)
    ctx.log.info(flow.request.port)
    if flow.request.pretty_host == "localhost":
        flow.request.scheme = "https"
        flow.request.host = "chatty-bobcat-13.telebit.io" #"hcloud.loca.lt"
        flow.request.port = 443
    if flow.request.path == "/switch":
        flow.request.host = "www.google.com"
        flow.request.port = 80
        flow.request.path = "/"

def response(flow: http.HTTPFlow):
    print(flow.response.headers)
    if flow.response.status_code == 302:
        print(flow.response.headers)
    flow.response.headers["location"] = flow.response.headers["location"].replace("http://localhost/", "http://localhost:8080/")
    flow.response.headers["location"] = flow.response.headers["location"].replace("https://chatty-bobcat-13.telebit.io/", "http://localhost:8080/")
    
    #ctx.log.info(flow.response)
    #print(flow.response.content)
