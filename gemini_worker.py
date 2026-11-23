import os
import sys
import json
from PIL import Image
from google import genai
from dotenv import load_dotenv

# Pre-load environment variables
load_dotenv(override=True)

client = None
try:
    if "GEMINI_API_KEY" in os.environ:
        client = genai.Client()
        sys.stderr.write("Gemini client initialized successfully in persistent worker.\n")
        sys.stderr.flush()
    else:
        sys.stderr.write("Warning: GEMINI_API_KEY not found in environment.\n")
        sys.stderr.flush()
except Exception as e:
    sys.stderr.write(f"Error creating Gemini client: {e}\n")
    sys.stderr.flush()

DESCRIPTIVE_PROMPT = (
    "Remove glare and overexposed reflections using non-destructive image restoration. "
    "Preserve exact geometry and dimensional accuracy. Do not modify object boundaries, "
    "measurements, positions, counts, spacing, perspective, or structural relationships. "
    "Recover clipped highlight detail while maintaining realistic texture and material appearance. "
    "Preserve all manufacturing marks, surface defects, edges, seams, labels, and inspection-relevant details. "
    "Output must remain visually and geometrically identical to the source image except for reduced glare and "
    "recovered highlight detail. This is an edit-only operation, not image generation or scene reconstruction."
)

def process_image(image_path: str) -> str:
    global client
    if client is None:
        load_dotenv(override=True)
        client = genai.Client()
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    raw_image = Image.open(image_path)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[raw_image, DESCRIPTIVE_PROMPT]
    )
    return response.text

if __name__ == "__main__":
    # Signal readiness to the Node parent process
    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            payload = json.loads(line)
            req_id = payload.get("id")
            image_path = payload.get("imagePath")
            analysis = process_image(image_path)
            print(json.dumps({"id": req_id, "success": True, "analysis": analysis}), flush=True)
        except Exception as e:
            sys.stderr.write(f"Error processing {req_id}: {e}\n")
            sys.stderr.flush()
            print(json.dumps({"id": req_id, "success": False, "error": str(e)}), flush=True)
