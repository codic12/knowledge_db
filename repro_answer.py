
import os
from llm_client import generate_answer

def test_generate_answer():
    query = "What is the payment amount mentioned?"
    contexts = [
        {
            'filepath': 'contract.pdf',
            'page': 1,
            'text': 'The contract states that Company A will pay Company B a total of $1000 for the consulting services.'
        }
    ]
    
    print("--- TESTING GENERATE ANSWER ---")
    answer = generate_answer(query, contexts)
    print(f"ANSWER:\n{answer}")
    print("--- END TEST ---")

if __name__ == "__main__":
    test_generate_answer()
