import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_gpt(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that selects the best Canvas section for a student's question."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] GPT call failed: {e}"

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    try:
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return [0.0] * 1536  # Dummy fallback vector
