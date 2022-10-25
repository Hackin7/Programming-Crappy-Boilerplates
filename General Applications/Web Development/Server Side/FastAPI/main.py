'''
pip3 install fastapi
uvicorn main:app
'''
import time
from typing import Union  # Union[str,None] is just a type suggestion
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

### Initialisation #####################################################
tags_metadata = [
    {
        "name": "admin",
        "description": "Administrative purposes"
    },
]
  
app = FastAPI(
    title = "Demo",
    description = "Backend Server",
    version = "1.0.0",
    openapi_tags=tags_metadata
)
    
### GET Params #########################################################

@app.get("/app")
def read_main():
    return {"message": "Hello World from main app"}
    
# http://localhost:8000/params/1?query=2
@app.get("/params/{path}") # if not path param, is query param
async def read_item(path: int, query:Union[str, None]  = None): 
    return {"path": path, "query": query} 

### POST Params ########################################################

# curl -H "Content-Type: application/json" -X POST  localhost:8000/post/1?query=a -d '{"name":"a", "price":1}'  | python -m json.tool
class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None

@app.post("/post/{path}")
async def create_item(item: Item, path: int = None, query:Union[str, None]  = None):
    return { "item": item, "itemname": item.name, "path": path, "query": query }

### Middleware #########################################################
app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

### Sub API ############################################################
# No docs
# https://fastapi.tiangolo.com/advanced/sub-applications/
subapi = FastAPI()
@subapi.get("/sub")
def read_sub():
    return {"message": "Hello World from sub API"}
app.mount("/subapi", subapi)

app.mount("/static", StaticFiles(directory="./"), name="static")

### Router #############################################################
# https://fastapi.tiangolo.com/tutorial/bigger-applications/#another-module-with-apirouter
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter()

@router.get("/users/", tags=["users"])
async def read_users():
    return [{"username": "Rick"}, {"username": "Morty"}]
    return fake_items_db

app.include_router(
    router,
    prefix="/admin",
    tags=["admin"],
    responses={418: {"description": "I'm a teapot"}},
)
