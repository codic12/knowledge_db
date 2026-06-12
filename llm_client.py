import os
from openai import OpenAI

# Initialize the OpenAI client for NVIDIA NIM
# The user needs to set NVIDIA_API_KEY environment variable.
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "meta/llama-3.1-70b-instruct"

if not NVIDIA_API_KEY:
    print("WARNING: NVIDIA_API_KEY environment variable is not set.")

client = OpenAI(
    api_key=NVIDIA_API_KEY if NVIDIA_API_KEY else "dummy_key",
    base_url=BASE_URL
)

def generate_answer(query: str, contexts: list) -> str:
    """
    Generate an answer using Nvidia NIM LLM based on the provided contexts.
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
        "5. Do not be overly literal; understand that '6a' and '6. A.' refer to the same logical section."
    )

    user_prompt = f"Context Information:\n{context_text}\nQuestion: {query}\nAnswer:"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return f"Error contacting LLM API. Please ensure NVIDIA_API_KEY is set. Details: {str(e)}"
