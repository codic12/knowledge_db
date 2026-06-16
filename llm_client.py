import os
import time
import threading
from openai import OpenAI

# Configuration
# Support switching between providers: 'gemini', 'nvidia'
LLM_PROVIDER = "nvidia"

# Providers configuration
PROVIDERS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "default_model": "meta/llama-3.1-70b-instruct"
    },
    "gemini": {
        # Gemini OpenAI-compatible endpoint
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GOOGLE_API_KEY",
        "default_model": "gemma-4-31b-it"
    }
}

# Fallback to gemini if provider is unknown
if LLM_PROVIDER not in PROVIDERS:
    print(f"WARNING: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Defaulting to 'gemini'.")
    LLM_PROVIDER = "nvidia"

config = PROVIDERS[LLM_PROVIDER]
API_KEY = os.environ.get(config["api_key_env"])

# Also check for GEMINI_API_KEY if provider is gemini
if LLM_PROVIDER == "gemini" and not API_KEY:
    API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL_NAME = os.environ.get("LLM_MODEL_NAME", config["default_model"])
BASE_URL = config["base_url"]

if not API_KEY:
    print(f"WARNING: {config['api_key_env']} environment variable is not set for provider '{LLM_PROVIDER}'.")

# Initialize the OpenAI-compatible client
client = OpenAI(
    api_key=API_KEY if API_KEY else "dummy_key",
    base_url=BASE_URL
)

# Global rate limiting state
_last_call_time = 0
_call_lock = threading.Lock()

import re

def safe_chat_completion(messages, temperature=0.0, max_tokens=512, retries=5):
    """
    Wrapper for chat.completions.create with rate limiting and retries.
    Enforces a minimum 4s interval (15 RPM) for Gemini.
    """
    global _last_call_time
    
    # 15 RPM limit = 4.1 seconds per request to be safe
    RATE_LIMIT_INTERVAL = 4.1 if LLM_PROVIDER == "gemini" else 0.1
    
    for attempt in range(retries + 1):
        with _call_lock:
            now = time.time()
            elapsed = now - _last_call_time
            if elapsed < RATE_LIMIT_INTERVAL:
                wait_time = RATE_LIMIT_INTERVAL - elapsed
                time.sleep(wait_time)
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                _last_call_time = time.time()
                
                return response
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg:
                    # Exponential backoff: 5, 10, 20, 40, 80s
                    backoff = (2 ** attempt) * 5
                    print(f"[{LLM_PROVIDER}] Rate limit hit. Retrying in {backoff}s... (Attempt {attempt+1}/{retries})")
                    time.sleep(backoff)
                    continue
                else:
                    # For other errors, log and re-raise or return a failure info
                    print(f"[{LLM_PROVIDER}] API Error: {e}")
                    raise e
                    
    raise Exception(f"Failed to get response from {LLM_PROVIDER} after {retries} retries.")

def generate_answer(query: str, contexts: list) -> str:
    """
    Generate an answer using the configured LLM based on the provided contexts.
    contexts is a list of dicts: [{'filepath': '...', 'page': 1, 'text': '...'}]
    """
    
    if not contexts:
        return "I cannot answer this based on the provided documents. No relevant information was found."
        
    # Construct context string with citations
    context_text = ""
    for idx, ctx in enumerate(contexts):
        filename = os.path.basename(ctx['filepath'])
        page = ctx['page']
        text = ctx['text'].strip()
        context_text += f"--- Document: {filename}, Page: {page} ---\n{text}\n\n"

    system_prompt = (
        "You are a highly accurate Document Intelligence Assistant.\n"
        "Your task is to answer the user's question based strictly on the provided context.\n"
        "Rules:\n"
        "1. You MUST NOT use outside knowledge. If the answer is not in the context, say 'I cannot answer this based on the provided documents.'\n"
        "2. Keep the answer short, accurate, and to the point.\n"
        "3. You MUST include citations for every piece of information you provide. "
        "Format citations as [DocumentName, Page X] at the end of the relevant sentence.\n"
        "4. Pay close attention to section hierarchies. If a user asks for '6A', look for 'Section 6, Part A' or sub-items under 'Section 6'.\n"
        "5. Do not be overly literal; understand that '6a' and '6. A.' refer to the same logical section.\n"
        "6. START your response IMMEDIATELY with the answer. Do not use phrases like 'Based on the context' or 'The document states'. Just provide the facts with citations.\n"
        "7. DO NOT include any internal reasoning, <thought> blocks, or step-by-step calculations in your output. Provide only the final answer."
    )

    user_prompt = f"Context Information:\n{context_text}\nQuestion: {query}\nAnswer:"

    print(f"\n--- LLM GENERATE ANSWER ({LLM_PROVIDER}): SYSTEM PROMPT ---")
    print(system_prompt)
    print("-------------------------------------------\n")
    print(f"--- LLM GENERATE ANSWER ({LLM_PROVIDER}): USER PROMPT ---")
    print(user_prompt)
    print("-----------------------------------------\n")

    try:
        response = safe_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error contacting {LLM_PROVIDER} API. Details: {str(e)}"
# "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=0.0,
#             max_tokens=512,
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return f"Error contacting {LLM_PROVIDER} API. Details: {str(e)}"
