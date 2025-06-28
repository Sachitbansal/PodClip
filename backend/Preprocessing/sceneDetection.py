from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()  # Load GOOGLE_API_KEY from .env

def analyze_podcast_segment(start_index: int, end_index: int, file_path: str = "youtube_chunks.json") -> dict:
    # Load specified chunk range from JSON
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)[start_index:end_index]

    # Prepare transcript text with timestamps
    transcript_text = "\n".join([
        f"[{i+1}] {chunk['metadata']['start_timestamp']} - {chunk['metadata']}s: {chunk['transcript']}"
        for i, chunk in enumerate(chunks)
    ])

    # Prompt template with escaped JSON braces
    prompt_template = PromptTemplate.from_template("""
    You are analyzing a podcast transcript (2 minutes long). Your goal is to:

    1. Detect if a question is asked, and when it starts (timestamp).
    2. Detect if there is a valuable or insightful response.
    3. Identify if any sentence stands out as motivational, emotional, shocking, or deeply insightful.
    4. Suggest whether this clip is worth turning into a YouTube Short.
    
    you need to extract the best clip from the transcript, which is less than 60 seconds long.
    identify a hook line from the transcript 

    Transcript:
    {transcript}

    Return the result in strict JSON format:

    {{
      "highlight": true/false,
      "reason": "...",
      "hook_line": "...",
      "start_time": ...,
      "end_time": ...,
    }}

    Return only the JSON.
    """)

    # Initialize Gemini model
    llm = GoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        temperature=0.8
    )

    # Compose chain and invoke
    chain = prompt_template | llm
    response = chain.invoke({"transcript": transcript_text})
    
    return response
