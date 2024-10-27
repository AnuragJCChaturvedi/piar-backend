from fastapi import FastAPI, File, UploadFile
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import subprocess
from pathlib import Path
from openai import OpenAI
import shutil
import uuid
import base64
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("API_KEY"))
# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (use a specific list for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths for saving files
AUDIO_UPLOAD_DIR = Path("/Users/anuragchaturvedi/Downloads/Learning/piar-api/input/audio")
IMAGE_UPLOAD_DIR = Path("/Users/anuragchaturvedi/Downloads/Learning/piar-api/input/images")
VIDEO_TRANSCRIPTION_OUTPUT_DIR = Path("/Users/anuragchaturvedi/Downloads/Learning/piar-api/output/images")

# Ensure directories exist
AUDIO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_TRANSCRIPTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Path to the Whisper binary
binary_path = "../whisper/main"

@app.post("/transcribe")
async def upload_files(audio: UploadFile = File(...), images: Optional[List[UploadFile]] = File(None)):
    try:
        # Save the audio file
        audio_video_uuid = uuid.uuid4()
        audio_filename = f"{audio_video_uuid}.wav"
        audio_path = AUDIO_UPLOAD_DIR / audio_filename
        with audio_path.open("wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        # Prepare a unique content list for each request
        image_to_text_content = [
            {
                "type": "text",
                "text": (
                    "You are observing a paramedics/EMTs POV. "
                    "Focus on anything related to the patient's condition and the EMT's actions. "
                    "Provide relevant information or general observations about the dispatch environment."
                ),
            }
        ]

        # Save each image file if provided
        image_paths = []
        if images:
            for image in images:
                image_filename = f"{uuid.uuid4()}.jpeg"
                image_path = IMAGE_UPLOAD_DIR / image_filename
                with image_path.open("wb") as buffer:
                    shutil.copyfileobj(image.file, buffer)
                image_paths.append(str(image_path))

                # Encode the image and add to content
                base64_image = encode_image(image_path)
                image_to_text_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )

        
        params = ["-f", str(audio_path), "-otxt"]
        
        # Run Whisper binary
        try:
            result = subprocess.run(
                [binary_path] + params,
                cwd="../whisper",  # Set working directory
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            error_message = f"Subprocess failed with error code {e.returncode}:\n{e.stderr}"
            print(error_message)
            return JSONResponse(content={"error": error_message}, status_code=500)

        # Call the analysis function asynchronously
        await analyze_images(audio_video_uuid, image_to_text_content)

        # Combine analysis and send for brief analysis
        result = await combine_transcribed_data(audio_video_uuid)

        # Return success response with paths
        return JSONResponse(content={"result": result})

    except Exception as e:
        print("An error occurred:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Function to encode image as base64 for analysis
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def get_image_transcribe_filename(uuid):
    return (VIDEO_TRANSCRIPTION_OUTPUT_DIR / f"{uuid}.txt") 

def get_audio_transcribe_filename(uuid):
    return (AUDIO_UPLOAD_DIR / f"{uuid}.wav.txt") 


# Async function to analyze images
# Async function to analyze images and save response to a text file
async def analyze_images(uuid, content):
    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=3000,
    )

    # Check if the response is successful
    if response:
        response_content = response.choices[0].message.content
        output_file_path = get_image_transcribe_filename(uuid)
        
        # Write the response to a text file
        with open(output_file_path, "w") as file:
            file.write(response_content)
        
        print(f"Analysis saved to {output_file_path}")
    else:
        print("Analysis failed.")


async def combine_transcribed_data(uuid):
    # Define paths for the audio and image analysis text files
    audio_transcription_path = get_audio_transcribe_filename(uuid)
    image_analysis_path = get_image_transcribe_filename(uuid)
    
    # Read the contents of each file
    combined_text = ""
    if audio_transcription_path.exists():
        with open(audio_transcription_path, "r") as audio_file:
            audio_content = audio_file.read()
            combined_text += "Audio Transcription:\n" + audio_content + "\n\n"

    if image_analysis_path.exists():
        with open(image_analysis_path, "r") as image_file:
            image_content = image_file.read()
            combined_text += "Image Analysis:\n" + image_content

    # Prepare the query content for GPT-4o with improved prompt engineering
    combine_transcribe_data = [
        {
            "role": "system",
            "content": (
                "You are an AI specializing in medical assistance, analyzing paramedic transcriptions and observations. "
                "Focus on relevant patient information and EMT actions to generate an initial summary."
            ),
        },
        {
            "role": "user",
            "content": (
                "Use the combined data below to generate a valid JSON string output only. "
                "Ensure the JSON string has fields 'Name', 'Occupation', and 'Summary'. Avoid extra information or explanations."
            ),
        },
        {
            "role": "user",
            "content": combined_text
        }
    ]

    # Send the combined query to GPT-4o
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=combine_transcribe_data,
        max_tokens=1000
    )

    # Print or handle JSON response content directly
    if response:
        return response.choices[0].message.content
    else:
        return "GPT-4o analysis failed."

