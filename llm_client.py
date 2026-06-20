import os
import time
import threading
import re
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configuration
# Support switching between providers: 'gemini', 'nvidia', 'deepseek'
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "deepseek")
DISABLE_THINKING = os.environ.get("DISABLE_THINKING", "true").lower() == "true"

# Providers configuration
PROVIDERS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "default_model": "meta/llama-3.1-70b-instruct"
    },
    "gemini": {
        "api_key_env": "GOOGLE_API_KEY",
        "default_model": "gemma-4-31b-it"
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat"
    }
}

# Fallback to deepseek if provider is unknown
if LLM_PROVIDER not in PROVIDERS:
    print(f"WARNING: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Defaulting to 'deepseek'.")
    LLM_PROVIDER = "deepseek"

config = PROVIDERS[LLM_PROVIDER]
API_KEY = os.environ.get(config["api_key_env"])

# Also check for GEMINI_API_KEY if provider is gemini
if LLM_PROVIDER == "gemini" and not API_KEY:
    API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL_NAME = os.environ.get("LLM_MODEL_NAME", config["default_model"])

if not API_KEY:
    print(f"WARNING: {config['api_key_env']} environment variable is not set for provider '{LLM_PROVIDER}'.")

# Initialize clients
openai_client = None
gemini_client = None

if LLM_PROVIDER in ["nvidia", "deepseek"]:
    openai_client = OpenAI(
        api_key=API_KEY if API_KEY else "dummy_key",
        base_url=config.get("base_url")
    )
elif LLM_PROVIDER == "gemini":
    gemini_client = genai.Client(api_key=API_KEY if API_KEY else "dummy_key")

# Global rate limiting state
_last_call_time = 0
_call_lock = threading.Lock()

def safe_chat_completion(messages, temperature=0.0, max_tokens=512, retries=5):
    """
    Wrapper for chat completions with rate limiting and retries.
    Enforces a minimum 4s interval (15 RPM) for Gemini.
    """
    global _last_call_time
    
    # Rate limit interval
    if LLM_PROVIDER == "gemini":
        RATE_LIMIT_INTERVAL = 4.1 # 15 RPM
    elif LLM_PROVIDER == "deepseek":
        RATE_LIMIT_INTERVAL = 0.5 # Fast
    else:
        RATE_LIMIT_INTERVAL = 0.1
    
    for attempt in range(retries + 1):
        with _call_lock:
            now = time.time()
            elapsed = now - _last_call_time
            if elapsed < RATE_LIMIT_INTERVAL:
                wait_time = RATE_LIMIT_INTERVAL - elapsed
                time.sleep(wait_time)
            
            try:
                if LLM_PROVIDER in ["nvidia", "deepseek"]:
                    response = openai_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    _last_call_time = time.time()
                    content = response.choices[0].message.content.strip()
                    
                    if DISABLE_THINKING:
                        # DeepSeek V3/R1 might return reasoning in some cases
                        # though deepseek-chat is usually direct.
                        # R1 (distill or original) might use thought tags.
                        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
                        content = re.sub(r'<\|think\|>.*?<\|thought\|>', '', content, flags=re.DOTALL)
                        content = re.sub(r'<\|think\|>', '', content)
                        content = re.sub(r'</?thought>', '', content)
                    
                    return content.strip()
                
                elif LLM_PROVIDER == "gemini":
                    # Convert OpenAI messages format to Gemini contents
                    system_instruction = None
                    contents = []
                    for msg in messages:
                        if msg['role'] == 'system':
                            system_instruction = msg['content']
                        else:
                            contents.append({'role': msg['role'], 'parts': [{'text': msg['content']}]})
                    
                    # Configure thinking if supported by the model
                    config_params = {
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                    
                    if DISABLE_THINKING:
                        if "thinking" in MODEL_NAME.lower():
                            config_params["thinking_config"] = types.ThinkingConfig(
                                include_thoughts=False,
                                thinking_budget=0
                            )
                    
                    try:
                        response = gemini_client.models.generate_content(
                            model=MODEL_NAME,
                            contents=contents,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                **config_params
                            )
                        )
                    except Exception as e:
                        if "thinking" in str(e).lower() or "budget" in str(e).lower():
                            config_params.pop("thinking_config", None)
                            response = gemini_client.models.generate_content(
                                model=MODEL_NAME,
                                contents=contents,
                                config=types.GenerateContentConfig(
                                    system_instruction=system_instruction,
                                    **config_params
                                )
                            )
                        else:
                            raise e
                    _last_call_time = time.time()
                    
                    # Manually assemble and strip ALL reasoning markers
                    full_text = ""
                    thought_text_accumulated = ""
                    if response and hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if candidate.content and candidate.content.parts:
                            for i, part in enumerate(candidate.content.parts):
                                t_val = None
                                if hasattr(part, 'thought') and part.thought:
                                    if isinstance(part.thought, str): t_val = part.thought
                                    elif hasattr(part.thought, 'text'): t_val = part.thought.text
                                    else: t_val = str(part.thought)
                                
                                if t_val:
                                    thought_text_accumulated += t_val + "\n"
                                    if DISABLE_THINKING: continue
                                
                                if hasattr(part, 'text') and part.text:
                                    full_text += part.text
                    
                    if not full_text.strip() and thought_text_accumulated.strip():
                        full_text = thought_text_accumulated
                    
                    if not full_text and response and response.text:
                        full_text = response.text

                    if full_text:
                        full_text = re.sub(r'<thought>.*?</thought>', '', full_text, flags=re.DOTALL)
                        full_text = re.sub(r'<\|think\|>.*?<\|thought\|>', '', full_text, flags=re.DOTALL)
                        full_text = re.sub(r'<\|think\|>', '', full_text)
                        full_text = re.sub(r'<\|thought\|>', '', full_text)
                        full_text = re.sub(r'</?thought>', '', full_text)
                        return full_text.strip()
                    
                    return "No response text received from Gemini."
                    
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg:
                    backoff = (2 ** attempt) * 5
                    print(f"[{LLM_PROVIDER}] Rate limit hit. Retrying in {backoff}s... (Attempt {attempt+1}/{retries})")
                    time.sleep(backoff)
                    continue
                else:
                    print(f"[{LLM_PROVIDER}] API Error: {e}")
                    raise e
                    
    raise Exception(f"Failed to get response from {LLM_PROVIDER} after {retries} retries.")

from sentence_transformers import SentenceTransformer

# Initialize local embedding model
local_embedding_model = None

def get_local_embedding_model():
    global local_embedding_model
    if local_embedding_model is None:
        print("Initializing local embedding model (all-MiniLM-L6-v2)...")
        local_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return local_embedding_model

def generate_embeddings(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using a local model.
    """
    if not texts:
        return []
        
    try:
        model = get_local_embedding_model()
        embeddings = model.encode(texts)
        # Convert numpy array to list of lists for JSON serialization
        return embeddings.tolist()
    except Exception as e:
        print(f"Local Embedding Error: {e}")
        raise e

def generate_answer(query: str, contexts: list) -> str:
    """
    Generate an answer using the configured LLM based on the provided contexts.
    """
    if not contexts:
        return "I cannot answer this based on the provided documents. No relevant information was found."
        
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
        "4. Pay close attention to section hierarchies.\n"
        "5. START your response IMMEDIATELY with the answer.\n"
        "6. DO NOT include any internal reasoning, <thought> blocks, or step-by-step calculations."
    )

    user_prompt = f"Context Information:\n{context_text}\nQuestion: {query}\nAnswer:"

    try:
        content = safe_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        return content
    except Exception as e:
        return f"Error contacting {LLM_PROVIDER} API. Details: {str(e)}"
