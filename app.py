from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from search_mm import wxd_search_basic, generate_answer, connect_wxd


app = FastAPI()

# Add CORS middleware to allow requests from any origin with any method (you can customize as needed)
origins = ["*"]  # Allow requests from any origin
# Add CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/retrieve/")
async def retrieve(query: str):
    client = connect_wxd()
    index_names = ["mm-banking-url"]
    response = wxd_search_basic(client, query, index_names)
    hits = response["hits"]["hits"]
    answer = generate_answer(query, hits)
    titles = [hit["_source"]["document_title"] for hit in hits]
    urls = [hit["_source"]["document_url"] for hit in hits]
    return {"answer":answer, "titles":titles, "urls":urls}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)