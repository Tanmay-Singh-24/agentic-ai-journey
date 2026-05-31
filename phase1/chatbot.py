from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()

conversation_history = []

system_prompt = "You are a sarcastic assistant who answers everything with a joke."
print("Chat started. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        break

    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            *conversation_history
        ]
    )

    assistant_message = response.choices[0].message.content

    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })

    print(f"\nAssistant: {assistant_message}\n")