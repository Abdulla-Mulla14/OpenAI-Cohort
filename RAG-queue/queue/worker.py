# flake8: noqa

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

client = OpenAI()

# Vector Embeddings - (yeh isliye banaya gaya hai kyunke jab user ki jab query aayegi tab hum usko jo hai numbers mein convert karenge.)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# (making connection with the database. because when query comes from the user so that it can search from the database and provide the relevant information)
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://vector-db:6333",
    collection_name="learning_vectors",
    embedding=embedding_model
)

async def process_querry(query: str):
    print("Searching Chunks", query)
    search_result = vector_db.similarity_search(
        query=query
    )

    context = "\n\n\n".join(
        [f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_result])
    
    SYSTEM_PROMPT = f"""

        You are an helpful AI Assistant who answers user query based on the available context retrieved from a PDF file along with page_contents and page number.

        You should only answer the user based on the following context and navigate the user to open the right page number to know more.

        Context:
        {context}
    """

    chat_completion = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
    )

    # Save to DB
    print(f"🤖: {query}", chat_completion.choices[0].message.content, "\n\n\n")
    return chat_completion.choices[0].message.content
