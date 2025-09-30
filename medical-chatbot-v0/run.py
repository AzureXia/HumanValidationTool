from main import app
from settings import HOST, PORT
import uvicorn



if __name__ == '__main__':
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False) #True -> False for prevent from reloading 
