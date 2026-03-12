import json
import ollama

schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3
        }
    },
    "required": ["title", "summary", "findings"]
}

prompt = """
Analyze the following text and return a JSON object with:
- title: string
- summary: string
- findings: list of 3 short bullet points

Text:
AI systems must be transparent, accountable, and safe.
"""

def run_test(schema_enabled, temp, prompt_strength):
    fmt = schema if schema_enabled else "json"
    p = prompt if prompt_strength == "strong" else "Return JSON."

    try:
        resp = ollama.generate(
            model="llama3.1:8b",
            prompt=p,
            format=fmt,
            options={"temperature": temp}
        )
        data = json.loads(resp["response"])
        return True, data
    except Exception as e:
        return False, str(e)

tests = [
    (True, 0, "strong"),
    (True, 0.2, "strong"),
    (True, 0, "weak"),
    (False, 0, "strong"),
    (False, 0.2, "strong"),
]

for t in tests:
    ok, result = run_test(*t)
    print(f"{t}: {'OK' if ok else 'FAIL'}")
    print(result)
    print()