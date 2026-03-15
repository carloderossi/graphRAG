"""
Test Ollama context length by asking the model to count tokens in a long prompt.
Run this while your Ollama server is running with llama3.1-16k loaded.
"""
import ollama
import json
import time
import traceback

MODEL = "llama3.1-16k:latest"

def count_characters_and_estimate_tokens(text):
    chars = len(text)
    # Very rough estimate: ~1.3–1.4 chars per token for English-ish text
    est_tokens = int(chars / 1.33)
    return chars, est_tokens

def ask_model_to_count_tokens(prompt_text):
    try:
        start = time.time()
        response = ollama.generate(
            model=MODEL,
            prompt=prompt_text,
            #format=KG_SCHEMA,
            options={"temperature": 0}
        )
        # print(f"response: {response}")
        # response.raise_for_status()
        # result = json.loads(response["response"])
        # print(f"result: {result}")
        full_response = response.get("response", "").strip()
        # print(f"response: {full_response}")
        total_duration = time.time() - start

        print("\n" + "="*70)
        print("Model answer:")
        print(full_response)
        print(f"\nGeneration took: {total_duration:.1f} seconds")
        print("="*70 + "\n")

        return full_response

    except Exception as e:
        print(f"Error talking to Ollama: {e}")
        traceback.print_exc()
        return None

def main():
    # Create a long repetitive text (~17–18k characters)
    # Using repeating pattern so it's easy to estimate tokens
    base = "The quick brown fox jumps over the lazy dog. " * 777
    # Add some extra length + instruction at the beginning
    long_text = (
        "Please count exactly how many tokens are in the TEXT below. "
        "If the token exceed the max windows, report a warning"
        "Return ONLY the number of the tokens in the text below, for example: TOKENS: 123456\n\n"
        "TEXT:\n" + (base) + " ---END-OF-REPEATED-TEXT--- " + "Final sentence. " * 42
    )

    char_count, est_tokens = count_characters_and_estimate_tokens(long_text)
    print(f"Generated prompt length: {char_count:,} characters")
    print(f"Rough token estimate:   ~{est_tokens:,} tokens\n")

    print(f"Sending to model '{MODEL}' ...\n")

    model_answer = ask_model_to_count_tokens(long_text)

    if model_answer:
        # Try to extract the number the model reported
        if "TOKENS:" in model_answer.upper():
            try:
                reported = model_answer.upper().split("TOKENS:")[1].strip().split()[0]
                reported = int(''.join(c for c in reported if c.isdigit()))
                print(f"Model reported: {reported} tokens")
                if reported > 10000:
                    print("→ Looks good! Context window is clearly larger than 8k.")
                elif reported > 4000:
                    print("→ At least bigger than old 2048/4096 default.")
                else:
                    print("→ Still seems limited — check Modelfile / ollama show")
            except:
                print("Could not parse the number from model answer.")
        else:
            print("Model did not return a clear token count. Try re-running or making prompt clearer.")

if __name__ == "__main__":
    main()