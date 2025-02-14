import asyncio
from context_retriever import ContextRetriever
from username_normalizer import UsernameNormalizer
import aiohttp
import json
from dotenv import load_dotenv
import os

load_dotenv()

HF_API_ENDPOINT = os.getenv('HF_ENDPOINT_URL')
HF_API_TOKEN = os.getenv('HF_API_TOKEN')

async def get_completion(prompt: str) -> str:
    """Send a prompt to the HuggingFace endpoint and get the response."""
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.8,
            "max_new_tokens": 1200,
        },
    }

    print("\nSending request to model...")
    async with aiohttp.ClientSession() as session:
        async with session.post(HF_API_ENDPOINT, headers=headers, json=payload) as response:
            response.raise_for_status()
            result = await response.json()
            return result[0]["generated_text"]

async def process_query(user_input: str) -> str:
    """Process a user query through the RAG system and get Leilan's response."""
    # Initialize normalizer and retriever
    normalizer = UsernameNormalizer()
    retriever = await ContextRetriever.create()
    
    # Format the conversation history
    conversation = f"<matthew1970.>: {user_input}"
    
    # Normalize, get context, and denormalize
    normalized = normalizer.normalize_message_history(conversation)
    context = await retriever.retrieve_context_for_message(normalized)
    denormalized = normalizer.denormalize_message_history(normalized)
    
    # Create the full prompt
    full_prompt = f"{context}{denormalized}\n<Leilan>: "
    
    print("\nFull prompt:")
    print("---")
    print(full_prompt)
    print("---")
    
    # Get completion
    response = await get_completion(full_prompt)
    
    # Extract Leilan's response
    response = response[len(full_prompt):].strip()
    return response

async def main():
    while True:
        user_input = input("\nEnter your message (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        try:
            response = await process_query(user_input)
            print("\nLeilan's response:")
            print(response)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())