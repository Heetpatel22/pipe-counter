import os
import sys
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load local environment variables from .env file
load_dotenv(override=True)

def process_industrial_pipes(image_path: str) -> str:
    # 1. Initialize the client (picks up GEMINI_API_KEY from environment/load_dotenv)
    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError("Please set your GEMINI_API_KEY environment variable in a .env file.")
    
    client = genai.Client()

    # 2. Verify and load the local image file using Pillow
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find the image at: {image_path}")
    
    raw_image = Image.open(image_path)

    # 3. Construct the prompt matching structural and atmospheric requirements
    descriptive_prompt = (
        "Remove glare and overexposed reflections using non-destructive image restoration. "
        "Preserve exact geometry and dimensional accuracy. Do not modify object boundaries, "
        "measurements, positions, counts, spacing, perspective, or structural relationships. "
        "Recover clipped highlight detail while maintaining realistic texture and material appearance. "
        "Preserve all manufacturing marks, surface defects, edges, seams, labels, and inspection-relevant details. "
        "Output must remain visually and geometrically identical to the source image except for reduced glare and "
        "recovered highlight detail. This is an edit-only operation, not image generation or scene reconstruction."
    )
    
    # 4. Execute the request using the multimodal model
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[raw_image, descriptive_prompt]
    )

    # 5. Output the model's structural evaluation
    print(response.text)
    return response.text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gemini_process.py <image_path>")
        sys.exit(1)
        
    target_image = sys.argv[1]
    try:
        process_industrial_pipes(target_image)
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)
