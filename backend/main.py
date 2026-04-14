import os
import json
import sys
import vertexai
from dotenv import load_dotenv
from google.cloud import vision
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import Part

'''
PSEUDO CODE for detect_web function:
function detect_web(image_path):
    create a Vision client
    read the image file as bytes
    construct a Vision Image object from those bytes
    call web_detection on the client, passing the image
    check for errors in the response
    return response.web_detection
'''
def detect_web(image_path):
    client = vision.ImageAnnotatorClient() #create a Vision client
    
    with open(image_path, 'rb') as image_file:
        content = image_file.read() #read the image file as bytes
    
    image = vision.Image(content=content) #construct a Vision Image object from those bytes
    
    response = client.web_detection(image=image) #call web_detection on the client, passing the image
    
    if response.error.message:
        raise Exception(f'{response.error.message}') #check for errors in the response
    
    return response.web_detection   #return response.web_detection


def format_web_detection(web_detection):

    result = {
        "best_guess_labels": [],
        "web_entities": [],
        "full_matching_images": [],
        "partial_matching_images": [],
        "visually_similar_images": [],
        "pages_with_matching_images": []
    }

    for entity in web_detection.web_entities:
        result["web_entities"].append({
            "description": entity.description,
            "score": entity.score
        })
    for image in web_detection.full_matching_images:
        result["full_matching_images"].append({
            "url": image.url
        })
    for image in web_detection.partial_matching_images:
        result["partial_matching_images"].append({
            "url": image.url
        })
    for page in web_detection.pages_with_matching_images:
        result["pages_with_matching_images"].append({
            "url": page.url,
            "title": page.page_title
        })
    return result

def build_prompt(web_detection_data):
    web_json = json.dumps(web_detection_data, indent=2)
    prompt = f"""You are an image forensics analyst. You have been given an image and 
    the results of a Google Cloud Vision web detection scan on that image.

    Here are the web detection results:
    {web_json}

    Based on both the image itself and the web detection data, provide the following analysis:

    1. **Image Description**: Describe what you see in the image — the subject, setting, 
    style (photo, illustration, meme, screenshot, etc.), and any notable details.

    2. **Identification**: What is this image of? Identify any people, places, objects, 
    logos, artwork, or events shown. Be as specific as possible using both your visual 
    analysis and the web entity data.

    3. **Origin & Source**: Based on the matching pages and URLs, where did this image 
    likely originate? Is it a news photo, stock image, social media post, meme, 
    promotional material, or something else?

    4. **Internet Presence**: Summarize where this image appears online. Are there 
    patterns in the types of sites hosting it? Has it spread widely or is it relatively 
    obscure?

    5. **Key Takeaway**: In one or two sentences, give the most important thing someone 
    should know about this image.

    If the web detection results are mostly empty, note that the image appears to have 
    limited or no public internet presence and focus your analysis on what you can see 
    in the image itself.

    Also, keep responses to one paragraph. Dont hedge unnecessarily. """
    return prompt

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("LOCATION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-1.5-flash-002")


def generate_analysis(web_detection_data):
    prompt = build_prompt(web_detection_data)
    response = model.generate_content(prompt)
    return response
