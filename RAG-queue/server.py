# Flake8(E501)
from fastapi import FastAPI, Query, Path
from .queue.connection import queue
from .queue.worker import process_querry


app = FastAPI()


@app.get('/')
def root():
    return {"status": 'server is up and running'}


@app.post('/chat')
def chat(
    query: str = Query(..., description="Chat Message")
):
    # Query ko Queue mein daal do aur phir
    # job is nothing but uss message ke baare mein information uski id, enqueue hua ya nahi hua, oo basically return kardeta hai
    job = queue.enqueue(process_querry, query)  # calls this function  "process_querry" with this parameter "query" - "process_querry(query)"

    # User ko bolo "your job recieved"
    return {"status": "queued", "job_id": job.id}


@app.get("/result/{job_id}")
def get_result(
    job_id: str = Path(..., description="Job ID")
):
    job = queue.fetch_job(job_id=job_id)
    result = job.return_value()

    return {"result": result}
