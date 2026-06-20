import os
from llm_client import safe_chat_completion

def test_summary_generation():
    text = "This is a sample document about a contract between Company A and Company B. It involves payment of $1000 for services rendered."
    prompt = (
        "Summarize the following document page text in EXACTLY one concise sentence. "
        "Focus on the main topic, section headings, and key entities mentioned. "
        "Return ONLY the summary.\n\n"
        f"TEXT:\n{text}\n\nSUMMARY:"
    )
    
    print("--- TESTING SUMMARY GENERATION ---")
    content = safe_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100
    )
    print(f"RAW OUTPUT:\n{content}")
    print("--- END TEST ---")

def test_summary_generation_v2():
    text = "This is a sample document about a contract between Company A and Company B. It involves payment of $1000 for services rendered."
    system_message = "You are a concise summarizer. You output ONLY the summary, no explanations, and no restating of instructions."
    user_prompt = f"Summarize the following text in EXACTLY one concise sentence:\n\n{text}\n\nSUMMARY:"
    
    print("\n--- TESTING SUMMARY GENERATION V2 (with system message) ---")
    content = safe_chat_completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=100
    )
    print(f"RAW OUTPUT:\n{content}")
    print("--- END TEST V2 ---")

if __name__ == "__main__":
    # Ensure we use the same model as the app
    test_summary_generation()
    test_summary_generation_v2()
