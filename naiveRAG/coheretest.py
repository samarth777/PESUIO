import cohere
import os
from dotenv import load_dotenv

load_dotenv()


co = cohere.ClientV2(os.getenv('COHERE_API_KEY'))

response = co.embed(
    texts=["hello", "goodbye"],
    model="embed-english-v3.0",
    input_type="classification",
    embedding_types=["float"]
)
print(print(response.embeddings.float_[0]))  # Print all embedding vectors)