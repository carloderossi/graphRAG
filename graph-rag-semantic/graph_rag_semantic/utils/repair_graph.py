import json
from pathlib import Path
import ollama

INPUT = Path("./docs/reg_kg_triples_repaired.jsonl")
OUTPUT = Path("./docs/reg_kg_triples_rescued.jsonl")

MODEL = "llama3.1:8b"

RESCUE_PROMPT="""
You extract a knowledge graph from legal text. 
Be aggressive: infer obligations, actors, and regulatory relationships even if implicit.

Schema:
entities: [{local_id, name, type}]
relations: [{source_local_id, target_local_id, type}]

Allowed entity types:
- Regulation
- Provider
- User
- AI_System
- High_Risk_System
- Authority
- Obligation
- Process
- Concept

Allowed relation types:
- REQUIRES
- MUST_COMPLY_WITH
- IS_DEFINED_IN
- IS_PART_OF
- APPLIES_TO
- SUPERVISED_BY

Rules:
- Always include the regulation if identifiable.
- Prefer short canonical names.
- Infer obligations when the text implies duties, prohibitions, or requirements.
- Infer processes when the text describes actions, assessments, monitoring, reporting.
- If actors interact, infer MUST_COMPLY_WITH or SUPERVISED_BY.
- If nothing is extractable, return empty arrays.

Examples:

Example 1:
Text:
"Providers of high-risk AI systems must implement risk management."

Output:
{
 "entities": [
   {"local_id":"e1","name":"Provider","type":"Provider"},
   {"local_id":"e2","name":"High-risk AI system","type":"High_Risk_System"},
   {"local_id":"e3","name":"Risk management","type":"Process"},
   {"local_id":"e4","name":"EU AI Act","type":"Regulation"}
 ],
 "relations":[
   {"source_local_id":"e1","target_local_id":"e4","type":"MUST_COMPLY_WITH"},
   {"source_local_id":"e3","target_local_id":"e2","type":"APPLIES_TO"},
   {"source_local_id":"e4","target_local_id":"e3","type":"REQUIRES"}
 ]
}

Example 2:
Text:
"Member States shall supervise providers."

Output:
{
 "entities":[
   {"local_id":"e1","name":"Member State","type":"Authority"},
   {"local_id":"e2","name":"Provider","type":"Provider"},
   {"local_id":"e3","name":"EU AI Act","type":"Regulation"}
 ],
 "relations":[
   {"source_local_id":"e2","target_local_id":"e1","type":"SUPERVISED_BY"},
   {"source_local_id":"e2","target_local_id":"e3","type":"MUST_COMPLY_WITH"}
 ]
}

Now extract entities and relations from this text:

"{{CHUNK}}"
"""

def call_rescue_llm(text):
    prompt = RESCUE_PROMPT.replace("{{CHUNK}}", text)

    response = ollama.generate(
        model=MODEL,
        prompt=prompt,
        options={"temperature": 0},
    )

    try:
        return json.loads(response["response"])
    except:
        return {"entities": [], "relations": []}

def rescue():
    rescued = 0
    total_empty = 0

    i=0

    with open(INPUT, "r", encoding="utf-8") as f_in, \
         open(OUTPUT, "w", encoding="utf-8") as f_out:

        for line in f_in:
            obj = json.loads(line)
            if i % 10 == 0:
              print("•", end="", flush=True)   # bullet every 10
            else:
              print(".", end="", flush=True)
            i=i+1  
            if len(obj["relations"]) == 0:
                total_empty += 1
                result = call_rescue_llm(obj["text"])

                if result["relations"]:
                    rescued += 1
                    obj["entities"] = result["entities"]
                    obj["relations"] = result["relations"]

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Total empty chunks: {total_empty}")
    print(f"Rescued chunks with new relations: {rescued}")
    print(f"Output written to {OUTPUT}")

if __name__ == "__main__":
    rescue()