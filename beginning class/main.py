from openai import OpenAI 
client = OpenAI()

response = client.embeddings.create(
    input = "My name is hitesh"
    model = "gpt-4o"
)

print(response.data[0].embedding)

