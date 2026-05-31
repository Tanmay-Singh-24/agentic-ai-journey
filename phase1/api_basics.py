from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()  # picks up GROQ_API_KEY from .env

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "What is a large language model in 2 sentences?"}
    ]
)

print(response.choices[0].message.content)