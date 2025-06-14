from langchain_openai import OpenAIEmbeddings 
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

# Vector Embeddings - (yeh isliye banaya gaya hai kyunke jab user ki jab query aayegi tab hum usko jo hai numbers mein convert karenge.)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# (making connection with the database. because when query comes from the user so that it can search from the database and provide the relevant information)
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model
)


# Take User query
query = input("> ")


# Vector Simalirity Search (query) in DB
search_result = vector_db.similarity_search(
    query=query
)
# print("search_result", search_result)


# SYSTEM_PROMPT and giving context to SYSTEM_PROMPT
context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_result]) # (hum kya kar rahe hai agar aap dekh paarahe ho toh hum har search_result ke andar loop kar rahe hai(for every result in search_result). Har ek result ke liye har ek entry ke liye hum result ke andar se "page_content, metadata, and source" ko utha rahe hai aur phir hum usko finally "\n\n" se join karke ek string bana rahe hai.)
SYSTEM_PROMPT = f"""

    You are an helpful AI Assistant who answers user query based on the available context retrieved from a PDF file along with page_contents and page number.

    You should only answer the user based on the following context and navigate the user to open the right page number to know more.

    Context:
    {context}
"""
# print("SYSTEM_PROMPT", SYSTEM_PROMPT)


# Calling an LLM (and you can use any LLM (Gemini, ChatGPT, etc...))
chat_completion = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
)

print(f"🤖: {chat_completion.choices[0].message.content}")
