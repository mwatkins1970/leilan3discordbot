import re
import json
import platform
from datetime import datetime
from functools import partial
from typing import TypeVar, Iterable, List, Optional, AsyncIterator
import asyncio
import aiohttp
from aiohttp import ClientTimeout
from aioitertools.more_itertools import take as async_take
from declarations import UserID, ActionHistory, Author, Action, Message
from ontology import LayerOfEnsembleFormat, EnsembleFormat, EmConfig
from trace import trace
from dotenv import load_dotenv
import os
import pathlib
from context_retriever import ContextRetriever  # Import the modified ContextRetriever
from username_normalizer import UsernameNormalizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")

# Get the directory where the script is located
script_dir = pathlib.Path(__file__).parent.absolute()
env_path = script_dir / '.env'
print(f"Script directory: {script_dir}")
print(f"Looking for .env at: {env_path}")
print(f"Does .env exist? {env_path.exists()}")


load_dotenv()  # Load environment variables from .env

# Get environment variables with default of None
HF_API_ENDPOINT = os.getenv('HF_ENDPOINT_URL')
HF_API_TOKEN = os.getenv('HF_API_TOKEN')

print(f"Loaded endpoint from env: {HF_API_ENDPOINT}")
print(f"Loaded token from env: {'[SET]' if HF_API_TOKEN else '[NOT SET]'}")

# Configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds

T = TypeVar("T")

# Utility functions
def unique(iterable: Iterable[T]) -> list[T]:
    """Remove duplicates from the iterable, preserving order."""
    return list(dict.fromkeys(iterable))

def clean_response(text: str, prompt: str, last_user: str = "") -> str:
    """Clean and format the response text from the model."""
    # Define zero-width space character
    ZWSP = "​"  # This is an actual zero-width space character, not an escape sequence
    
    # Remove the input prompt from the beginning of the generated text
    if text.startswith(prompt):
        text = text[len(prompt):].strip()

    # Split the text into lines
    lines = text.split("\n")
    
    # Process each line
    processed_lines = []
    for line in lines:
        # Check if the line starts with "<Leilan>:"
        if line.startswith("<Leilan>:"):
            # Remove "<Leilan>:" from the beginning of the line
            line = line[len("<Leilan>:"):].strip()
        
        # Check if the line starts with the last user's username in angle brackets
        if last_user and line.startswith(f"<{last_user}>:"):
            # Remove the username in angle brackets from the beginning of the line
            line = line[len(f"<{last_user}>:"):].strip()
            
        # Add zero-width space after numbers at start of line to prevent Discord list formatting
        line = re.sub(r'^(\d+)\. ', fr'\1.{ZWSP} ', line)
        
        processed_lines.append(line)

    # Join the processed lines back into a single string
    processed_text = "\n".join(processed_lines)

    # Find the first occurrence of another user's turn at the start of a line
    match = re.search(r'(?:^|\n)<(?!Leilan)[^>]+>:', processed_text)
    if match:
        # Extract everything before the next user's turn
        response = processed_text[:match.start()].strip()
        # Remove asterisks or other problematic characters
        response = re.sub(r'\*', '', response)
        processed_text = response

    # Clean up any complete angle bracket tags
    processed_text = re.sub(r'<[^>]*>', '', processed_text)
    
    # Remove any partial tag at the end of the message
    processed_text = re.sub(r'<(?:[\w._-]{1,32})?$', '', processed_text)
    
    # Do a final cleanup of any whitespace
    return processed_text.strip()

# Muffler functions
def has_url(prompt: str, reply: str) -> bool:
    """Check if the reply contains URLs."""
    return any(prefix in reply.lower() for prefix in ['http://', 'https://', 'www.'])

def has_pump_fun_ca(prompt: str, reply: str) -> bool:
    """Check for pump.fun.ca content."""
    return 'pump.fun.ca' in reply.lower()

def has_img_url_token(prompt: str, reply: str) -> bool:
    """Check for image URL tokens."""
    return '[img]' in reply.lower() or '[/img]' in reply.lower()

async def format_ensemble(
    ensemble,
    ensemble_format: List[LayerOfEnsembleFormat],
    tokenization_model: str,
    ctx_vars: dict
) -> str:
    prompt = ""
    local_format = ensemble_format[0]

    if hasattr(ensemble, '__aiter__'):
        ensemble_iterator = ensemble
    else:
        async def async_iter():
            for item in ensemble:
                yield item
        ensemble_iterator = async_iter()

    async for subensemble in ensemble_iterator:
        if isinstance(subensemble, Action):
            # Let the format class handle the entire message as one unit
            string = local_format.format.render(subensemble)

            # We'll only process for angle brackets here, not split messages
            if isinstance(string, str):
                # Ensure usernames are properly formatted with angle brackets
                lines = string.split('\n')
                formatted_lines = []
                
                for line in lines:
                    # Convert "username:" to "<username>:"
                    line = re.sub(r'^([^<\s][^:]*?):', r'<\1>:', line)
                    formatted_lines.append(line)
                
                string = '\n'.join(formatted_lines)
        else:
            string = await format_ensemble(subensemble, ensemble_format[1:], tokenization_model, ctx_vars)
            
        prompt = f"{prompt}{local_format.separator}{string}" if prompt else string

        if len(prompt) > local_format.max_tokens:
            break

    return f"{local_format.header.format(**ctx_vars)}{prompt}{local_format.footer.format(**ctx_vars)}"

import os
from datetime import datetime

async def get_prompt(history: ActionHistory, em: EmConfig) -> str:
    ctx_vars = {"now": datetime.now(), "hostname": platform.node()}
    
    # Create normalizer instance
    normalizer = UsernameNormalizer()

    # Convert history to list first if it's an async iterator
    if hasattr(history, '__aiter__'):
        history_list = [msg async for msg in history]
    else:
        history_list = list(history)

    # Get the last message from history
    last_message = None
    for msg in reversed(history_list):
        if msg.author.name != em.name:
            last_message = msg.content
            break

    ctx_vars["last_message"] = last_message or ""

    # Include all messages in the history
    filtered_history = list(history_list)
    # Reverse the order of filtered history
    filtered_history = list(reversed(filtered_history))

    # Format the message history
    message_history_ensemble = await format_ensemble(
            filtered_history,
            [
                LayerOfEnsembleFormat(
                    format=em.message_history_format,
                    max_items=em.recency_window,
                    max_tokens=em.message_history_max_tokens,
                    operator=em.message_history_operator,
                    separator=em.message_history_separator,
                    header=em.message_history_header,
                    footer=em.message_history_footer,
                )
            ],
            em.continuation_model,
            ctx_vars,
        )

    # Normalize usernames in the formatted message history
    normalized_history = normalizer.normalize_message_history(message_history_ensemble)
    
    # Create an instance of ContextRetriever
    retriever = await ContextRetriever.create()
    
    # Retrieve context using the normalized history
    dynamic_header = await retriever.retrieve_context_for_message(normalized_history)
    
    # Denormalize usernames in the message history
    denormalized_history = normalizer.denormalize_message_history(normalized_history)
    
    # Generate the prompt
    prompt = (
        dynamic_header +
        denormalized_history + 
        f"\n<{em.name}>: "
    )
    
    # Write the prompt to a file
    #prompts_dir = "chapter2/prompts"
    #os.makedirs(prompts_dir, exist_ok=True)  # Create the directory if it doesn't exist
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #prompt_filename = f"prompt_{timestamp}.txt"
    #prompt_filepath = os.path.join(prompts_dir, prompt_filename)
    #with open(prompt_filepath, "w") as f:
    #    f.write(prompt)
    
    return prompt   

async def get_replies(
    em: EmConfig,
    prompt: str,
    stop_sequences: list[str],
    my_user_id: UserID,
    attempt: int = 1
) -> AsyncIterator[str]:
    """Get replies from the HuggingFace endpoint with retry logic."""
    print("\n=== Starting get_replies attempt", attempt, "===")  # Debug print
    endpoint = HF_API_ENDPOINT or getattr(em, 'endpoint_url', None)
    token = HF_API_TOKEN or getattr(em, 'api_key', None)
    
    if not endpoint:
        raise ValueError("No HuggingFace endpoint URL configured")
    if not token:
        raise ValueError("No HuggingFace API token configured")

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": em.temperature,
            "max_new_tokens": em.continuation_max_tokens,
        },
    }
    
    # Count tokens in the prompt
    token_count = len(tokenizer.encode(prompt))
    
    print("\n=== PAYLOAD BEING SENT TO API ===")
    print("Endpoint:", endpoint)
    print("\nFull prompt being sent:")
    print("---")
    print(prompt)
    print("---")
    print("\nPayload structure:")
    print(json.dumps(payload, indent=2))
    print("=== END PAYLOAD ===\n")
    print(f"\nPrompt token count: {token_count}" + "\n")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        try:
            print("Making request to endpoint...")  # Debug print
            timeout = ClientTimeout(
                total=90,      # Total operation timeout
                connect=30,    # Initial connection timeout
                sock_read=60   # Socket read timeout
            )
            async with session.post(endpoint, headers=headers, json=payload, timeout=timeout) as response:
                print(f"Got response with status: {response.status}")  # Debug print

                if response.status == 503 and attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY ** attempt
                    print(f"Service unavailable, retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    async for reply in get_replies(em, prompt, stop_sequences, my_user_id, attempt + 1):
                        yield reply
                    return

                response.raise_for_status()
                print("Getting JSON from response...")  # Debug print
                completions = await response.json()
                print("\n=== API RESPONSE ===")
                print(json.dumps(completions, indent=2))
                print("=== END RESPONSE ===\n")
                
                if isinstance(completions, list) and completions:
                    # Extract last user from the prompt by finding the last occurrence of "<username>:"
                    last_user_match = re.findall(r'<([^>]+)>:', prompt)
                    last_user = last_user_match[-1] if last_user_match else ""
                    
                    generated_text = clean_response(completions[0].get("generated_text", ""), prompt, last_user)
                    print(f"Generated response (attempt {attempt}):", generated_text)
                    yield generated_text
                    
        except aiohttp.ClientError as e:
            print(f"aiohttp ClientError: {str(e)}")  # More specific error
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY ** attempt
                await asyncio.sleep(delay)
                async for reply in get_replies(em, prompt, stop_sequences, my_user_id, attempt + 1):
                    yield reply
            else:
                raise

@trace
async def generate_response(
    my_user_id: UserID,
    history: ActionHistory,
    em: EmConfig
) -> AsyncIterator[Message]:
    """Generate a response based on the prompt and configuration."""
    prompt = await get_prompt(history, em)

    # Get unique stop sequences from recent messages
    stop_sequences = unique([
        f"<{message.author.name}>:"
        for message in await async_take(em.recency_window, history)
        if message.author.name != em.name
    ] + em.stop_sequences)

    # Map muffler strings to functions
    muffler_map = {
        "has_url": has_url,
        "has_pump_fun_ca": has_pump_fun_ca,
        "has_img_url_token": has_img_url_token,
    }
    mufflers = [muffler_map[m] for m in em.mufflers if m in muffler_map]

    retry = True
    tries = 0

    while retry and tries < MAX_RETRIES:
        tries += 1
        retry = False

        try:
            async for reply in get_replies(em, prompt, stop_sequences, my_user_id, tries):
                # Check mufflers
                for muffler_func in mufflers:
                    if muffler_func(prompt, reply):
                        print(f"Content muffled by {muffler_func.__name__}")
                        retry = True
                        break
                else:
                    # Create message if no mufflers triggered
                    yield Message(
                        author=Author(name=em.name, user_id=my_user_id),
                        content=reply,
                        timestamp=datetime.now().timestamp()
                    )

        except Exception as e:
            print(f"\n=== Detailed Error Information (attempt {tries}) ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Full error details:", e)
            if hasattr(e, '__dict__'):
                print("Error attributes:", e.__dict__)
            print(f"Status code if HTTP error: {getattr(e, 'status', 'N/A')}")
            print("=" * 50)
            
            if tries < MAX_RETRIES:
                retry = True
                await asyncio.sleep(RETRY_BASE_DELAY ** tries)
            else:
                raise