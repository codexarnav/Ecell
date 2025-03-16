import requests
import json
import torch
import librosa
import soundfile as sf
import PyPDF2
import spacy
import sqlite3
import math
from typing import List, Dict
import pandas as pd
import os
from functools import lru_cache
import streamlit as st
from twilio.rest import Client
from PIL import Image
import hashlib
import tempfile

# ----------------- LAZY LOADING & INITIALIZATION ----------------- #

# Global variables for models and processors
models = {
    "tokenizer": None,
    "summarization": None,
    "nlp": None,
    "entity_ruler": None,
    "clip_model": None,
    "clip_processor": None
}

# Configuration
config = {
    # API keys
    "TWILIO_ACCOUNT_SID": "AC5ef80d1da418750186d7eca80e3f8f79",
    "TWILIO_AUTH_TOKEN": "a47d99e14115cc235ddf63a8ef6c5a0a",
    "TWILIO_PHONE_NUMBER": "+15179553782",
    "HF_API_TOKEN": "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",
    "GEMINI_API_KEY": "AIzaSyCf7rWXF7j2UlBhxTvXbThNRDsnbH5UA58",
    "OPENCAGE_API_KEY": "d63508503f9042be8ccedd15b26f07ec",

    # Database
    "DB_PATH": "disaster_management.db",

    # Model parameters
    "ASR_MODEL": "openai/whisper-small",
    "SUMMARIZATION_MODEL": "facebook/bart-large-cnn",
    "SPACY_MODEL": "en_core_web_lg",
    "CLIP_MODEL": "openai/clip-vit-base-patch32",
    "WHISPER_MODEL": "openai/whisper-large-v3",
    "BLIP_MODEL": "Salesforce/blip-image-captioning-large",
    "GEMINI_MODEL": "models/gemini-1.5-pro"
}

# Headers for API requests
headers = {"Authorization": f"Bearer {config['HF_API_TOKEN']}"}

# ----------------- LAZY LOADING FUNCTIONS ----------------- #

@lru_cache(maxsize=1)
def get_nlp():
    """Lazy load spaCy NLP model"""
    if models["nlp"] is None:
        st.info("Loading language model... This may take a moment.")
        models["nlp"] = spacy.load(config["SPACY_MODEL"])

        # Add entity ruler if not already present
        if "entity_ruler" not in [pipe for pipe, _ in models["nlp"].pipeline]:
            ruler = models["nlp"].add_pipe("entity_ruler", before="ner")
            patterns = [
                {"label": "EMERGENCY_TYPE", "pattern": [{"lower": "earthquake"}]},
                {"label": "EMERGENCY_TYPE", "pattern": [{"lower": "fire"}]},
                {"label": "EMERGENCY_TYPE", "pattern": [{"lower": "flood"}]},
                {"label": "EMERGENCY_TYPE", "pattern": [{"lower": "hurricane"}]},
                {"label": "EMERGENCY_TYPE", "pattern": [{"lower": "tornado"}]},
                {"label": "EMERGENCY_TYPE", "pattern": [{"lower": "tsunami"}]},
                {"label": "EMERGENCY_TYPE", "pattern": [{"lower": "landslide"}]},
                {"label": "SEVERITY", "pattern": [{"lower": "critical"}]},
                {"label": "SEVERITY", "pattern": [{"lower": "severe"}]},
                {"label": "SEVERITY", "pattern": [{"lower": "urgent"}]},
                {"label": "SEVERITY", "pattern": [{"lower": "major"}]},
                {"label": "SEVERITY", "pattern": [{"lower": "minor"}]},
                {"label": "VICTIM_CONDITION", "pattern": [{"lower": "injured"}]},
                {"label": "VICTIM_CONDITION", "pattern": [{"lower": "unconscious"}]},
                {"label": "VICTIM_CONDITION", "pattern": [{"lower": "stuck"}]},
                {"label": "VICTIM_CONDITION", "pattern": [{"lower": "trapped"}]},
                {"label": "VICTIM_CONDITION", "pattern": [{"lower": "missing"}]},
                {"label": "DAMAGE", "pattern": [{"lower": "collapsed"}]},
                {"label": "DAMAGE", "pattern": [{"lower": "destroyed"}]},
                {"label": "DAMAGE", "pattern": [{"lower": "damaged"}]},
                {"label": "DAMAGE", "pattern": [{"lower": "flooded"}]},
                {"label": "DAMAGE", "pattern": [{"lower": "burned"}]}
            ]
            ruler.add_patterns(patterns)
            models["entity_ruler"] = ruler
    return models["nlp"]

@lru_cache(maxsize=1)
def get_tokenizer_and_summarization_model():
    """Lazy load summarization model and tokenizer"""
    if models["tokenizer"] is None or models["summarization"] is None:
        st.info("Loading summarization model... This may take a moment.")
        from transformers import BartTokenizer, BartForConditionalGeneration
        models["tokenizer"] = BartTokenizer.from_pretrained(config["SUMMARIZATION_MODEL"])
        models["summarization"] = BartForConditionalGeneration.from_pretrained(config["SUMMARIZATION_MODEL"])
    return models["tokenizer"], models["summarization"]

@lru_cache(maxsize=1)
def get_clip_model_and_processor():
    """Lazy load CLIP model and processor"""
    if models["clip_model"] is None or models["clip_processor"] is None:
        st.info("Loading CLIP model... This may take a moment.")
        from transformers import CLIPProcessor, CLIPModel
        models["clip_model"] = CLIPModel.from_pretrained(config["CLIP_MODEL"])
        models["clip_processor"] = CLIPProcessor.from_pretrained(config["CLIP_MODEL"])
    return models["clip_model"], models["clip_processor"]

@lru_cache(maxsize=1)
def get_asr_pipeline():
    """Lazy load ASR pipeline"""
    from transformers import pipeline
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return pipeline(
        task="automatic-speech-recognition",
        model=config["ASR_MODEL"],
        device=device
    )

def get_genai():
    """Lazy load Google GenerativeAI"""
    import google.generativeai as genai
    genai.configure(api_key=config["GEMINI_API_KEY"])
    return genai

def get_twilio_client():
    """Get Twilio client"""
    return Client(config["TWILIO_ACCOUNT_SID"], config["TWILIO_AUTH_TOKEN"])

# ----------------- DATABASE FUNCTIONS ----------------- #

def init_db():
    """Initialize database with tables if they don't exist"""
    conn = None
    try:
        conn = sqlite3.connect(config["DB_PATH"])
        cursor = conn.cursor()

        # Create emergency table
        cursor.execute('''CREATE TABLE IF NOT EXISTS emergency
                     (eid INTEGER PRIMARY KEY,
                      location TEXT,
                      latitude REAL,
                      longitude REAL,
                      text TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # Create resource table
        cursor.execute('''CREATE TABLE IF NOT EXISTS resource
                     (resourceid INTEGER PRIMARY KEY,
                      amenity TEXT,
                      name TEXT,
                      latitude REAL,
                      longitude REAL,
                      created_by INTEGER,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # Create volunteer table with password field
        cursor.execute('''CREATE TABLE IF NOT EXISTS volunteer
                     (id INTEGER PRIMARY KEY,
                      name TEXT,
                      email TEXT UNIQUE,
                      password_hash TEXT,
                      location TEXT,
                      latitude REAL,
                      longitude REAL,
                      speciality TEXT,
                      phone TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

def get_db_connection():
    """Get database connection with proper configuration"""
    conn = sqlite3.connect(config["DB_PATH"], check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.create_function("HAVERSINE", 4, haversine)
    return conn

def execute_query(query: str, params: tuple = (), commit: bool = True) -> List[Dict]:
    """Execute a database query with proper connection handling"""
    conn = None
    results = []
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query, params)

        if query.strip().upper().startswith("SELECT"):
            results = [dict(row) for row in cur.fetchall()]

        if commit:
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        if conn and commit:
            conn.rollback()
    finally:
        if conn:
            conn.close()
    return results

# ----------------- HELPER FUNCTIONS ----------------- #

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on earth"""
    # Ensure inputs are floats
    lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
    R = 6371  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def hash_password(password):
    """Create a secure password hash"""
    return hashlib.sha256(password.encode()).hexdigest()

def get_lat_lon(location_name):
    """Get latitude and longitude from location name using OpenCage Geocoder"""
    url = f'https://api.opencagedata.com/geocode/v1/json?q={location_name}&key={config["OPENCAGE_API_KEY"]}'
    try:
        response = requests.get(url)
        data = response.json()
        if data['status']['code'] == 200 and data['results']:
            lat = data['results'][0]['geometry']['lat']
            lon = data['results'][0]['geometry']['lng']
            return lat, lon
        else:
            return None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None

# ----------------- CORE FUNCTIONS ----------------- #

def english_speech_to_text(file_path):
    """Convert audio file to text using ASR"""
    try:
        model_asr = get_asr_pipeline()
        audio, sr = librosa.load(file_path, sr=16000)
        sf.write("resampled.wav", audio, sr, format='wav')
        result = model_asr(
            "resampled.wav",
            generate_kwargs={"task": "translate"}  # Forces English output
        )
        return result["text"]
    except Exception as e:
        st.error(f"Speech-to-text error: {e}")
        return ""

def extract_entities(text):
    """Extract entities from text using spaCy"""
    nlp = get_nlp()
    doc = nlp(text)
    entities = {
        "location": [],
        "date": [],
        "emergency_type": [],
        "severity": [],
        "victim_condition": [],
        "damage": []
    }
    for ent in doc.ents:
        if ent.label_ == "EMERGENCY_TYPE":
            entities["emergency_type"].append(ent.text)
        elif ent.label_ == "SEVERITY":
            entities["severity"].append(ent.text)
        elif ent.label_ == "VICTIM_CONDITION":
            entities["victim_condition"].append(ent.text)
        elif ent.label_ == "DAMAGE":
            entities["damage"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["date"].append(ent.text)
        elif ent.label_ in ['GPE', 'LOC']:
            entities["location"].append(ent.text)
    return entities

def transcribe_audio(audio_path):
    """Transcribe audio using Hugging Face Whisper model"""
    API_URL = f"https://api-inference.huggingface.co/models/{config['WHISPER_MODEL']}"
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        payload = {"options": {"task": "translate"}}
        response = requests.post(API_URL, headers=headers, data=audio_data, json=payload)
        if response.status_code == 200:
            return response.json()["text"]
        else:
            return f"Error: {response.text}"
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""

def process_image(image_path):
    """Process image using BLIP model"""
    API_URL = f"https://api-inference.huggingface.co/models/{config['BLIP_MODEL']}"
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        response = requests.post(API_URL, headers=headers, data=image_data)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and result and 'generated_text' in result[0]:
            return result[0]['generated_text']
        else:
            return "Unexpected API response format"
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return ""

def process_text(text_input):
    """Process text using CLIP model"""
    text_options = ["fire", "earthquake", "flood", "car accident", "building collapse",
                    "cyclone", "landslide", "medical emergency"]
    try:
        model, processor = get_clip_model_and_processor()
        inputs = processor(text=text_options, return_tensors="pt", padding=True)
        text_features = model.get_text_features(**inputs)
        input_text = processor(text=[text_input], return_tensors="pt", padding=True)
        input_features = model.get_text_features(**input_text)
        similarities = torch.nn.functional.cosine_similarity(input_features, text_features, dim=1)
        predicted_label = text_options[similarities.argmax().item()]
        return predicted_label, similarities.max().item()
    except Exception as e:
        st.error(f"Text processing error: {e}")
        return "unknown", 0.0

def generate_summary(pdf_path):
    """Generate summary from PDF using BART model"""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        tokenizer, model = get_tokenizer_and_summarization_model()
        inputs = tokenizer.encode('summarize: ' + text, return_tensors="pt",
                                  max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=1000, min_length=50,
                                    length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"PDF summarization error: {e}")
        return "Error generating summary"

def get_first_aid_response(disaster_type, input_text):
    """Get first aid response using Google's Gemini model"""
    try:
        genai = get_genai()
        model_gen = genai.GenerativeModel(config["GEMINI_MODEL"])
        prompt = f"What are the first-aid measures for a {disaster_type}? Context provided: {input_text}"
        response = model_gen.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"First aid response error: {e}")
        return "Error generating first aid response"

def send_sms(to, message):
    """Send SMS using Twilio"""
    try:
        client = get_twilio_client()
        msg = client.messages.create(
            body=message,
            from_=config["TWILIO_PHONE_NUMBER"],
            to=to
        )
        return f"SMS sent successfully to {to}: {msg.sid}"
    except Exception as e:
        st.error(f"Error sending SMS: {e}")
        return f"Failed to send SMS: {str(e)}"

def notify_emergency_services(location, severity, user_phone):
    """Notify emergency services and user via SMS"""
    government_contact = "+919625984260"  # Emergency services number
    try:
        # Get location coordinates
        get_lat_lon(location)

        # Message to emergency services
        gov_message = f"🚨 URGENT! Emergency at {location}. Severity: {severity}. Immediate response required."
        gov_status = send_sms(government_contact, gov_message)

        # Message to user
        user_message = f"🔹 Help is on the way! Authorities have been alerted to your emergency at {location} (Severity: {severity}). Stay safe!"
        user_status = send_sms(user_phone, user_message)

        return gov_status, user_status
    except Exception as e:
        st.error(f"Notification error: {e}")
        return f"Failed to notify: {str(e)}", ""

# ----------------- DATABASE OPERATIONS ----------------- #

def add_emergency(location: str, lat: float, lon: float, text: str):
    """Add new emergency to database"""
    return execute_query(
        '''INSERT INTO emergency (location, latitude, longitude, text)
           VALUES (?, ?, ?, ?)''',
        (location, lat, lon, text)
    )

def get_nearest_emergencies(user_lat: float, user_lon: float, max_km=10, limit=10):
    """Get nearest emergencies to location"""
    return execute_query(
        f'''SELECT *, 
            HAVERSINE(?, ?, latitude, longitude) AS distance
            FROM emergency
            WHERE HAVERSINE(?, ?, latitude, longitude) <= ?
            ORDER BY distance
            LIMIT ?''',
        (user_lat, user_lon, user_lat, user_lon, max_km, limit)
    )

def add_resource(amenity: str, name: str, lat: float, lon: float, created_by: int):
    """Add new resource to database"""
    return execute_query(
        '''INSERT INTO resource (amenity, name, latitude, longitude, created_by)
           VALUES (?, ?, ?, ?, ?)''',
        (amenity, name, lat, lon, created_by)
    )

def get_nearest_resources(user_lat: float, user_lon: float, max_km=10, limit=10):
    """Get nearest resources to location"""
    return execute_query(
        f'''SELECT *, 
            HAVERSINE(?, ?, latitude, longitude) AS distance
            FROM resource
            WHERE HAVERSINE(?, ?, latitude, longitude) <= ?
            ORDER BY distance
            LIMIT ?''',
        (user_lat, user_lon, user_lat, user_lon, max_km, limit)
    )

def register_volunteer(name: str, email: str, password: str, location: str,
                       lat: float, lon: float, speciality: str, phone: str):
    """Register new volunteer"""
    # Hash the password
    password_hash = hash_password(password)

    # Check if email already exists
    existing = execute_query(
        "SELECT * FROM volunteer WHERE email = ?",
        (email,)
    )
    if existing:
        return False, "Email already registered"

    # Insert new volunteer
    execute_query(
        '''INSERT INTO volunteer 
           (name, email, password_hash, location, latitude, longitude, speciality, phone)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (name, email, password_hash, location, lat, lon, speciality, phone)
    )

    # Get the new volunteer ID
    result = execute_query(
        "SELECT id FROM volunteer WHERE email = ?",
        (email,)
    )
    if result:
        return True, result[0]['id']
    return False, "Registration failed"

def volunteer_login(email: str, password: str):
    """Login volunteer by email and password"""
    password_hash = hash_password(password)
    volunteers = execute_query(
        'SELECT * FROM volunteer WHERE email = ? AND password_hash = ?',
        (email, password_hash)
    )
    if not volunteers:
        return False, "Invalid email or password"
    return True, volunteers[0]

def get_volunteer_dashboard(volunteer_id: int):
    """Get dashboard data for volunteer"""
    # Get volunteer info
    volunteer = execute_query(
        'SELECT * FROM volunteer WHERE id = ?',
        (volunteer_id,)
    )[0]

    # Get nearby emergencies
    emergencies = get_nearest_emergencies(
        volunteer['latitude'],
        volunteer['longitude']
    )

    # Get nearby resources
    resources = get_nearest_resources(
        volunteer['latitude'],
        volunteer['longitude']
    )

    # Get resources created by this volunteer
    my_resources = execute_query(
        'SELECT * FROM resource WHERE created_by = ?',
        (volunteer_id,)
    )

    return volunteer, emergencies, resources, my_resources

# ----------------- STREAMLIT APP ----------------- #

def main():
    """Main Streamlit application"""
    # Initialize database
    init_db()

    st.title("Disaster Management Application")

    # Sidebar navigation
    workflow = st.sidebar.radio(
        "Select Workflow",
        ("User", "Volunteer Login", "Volunteer Registration")
    )

    if workflow == "User":
        user_workflow()
    elif workflow == "Volunteer Login":
        volunteer_login_workflow()
    elif workflow == "Volunteer Registration":
        volunteer_registration_workflow()

def user_workflow():
    """User workflow for emergency reporting"""
    st.header("Emergency Reporting")
    st.subheader("Submit your emergency report")

    # Input fields
    text_input = st.text_area("Enter emergency details (optional):")
    voice_file = st.audio_input("Record your emergency (optional):")
    image_file = st.file_uploader("Upload image (optional)", type=["jpg", "png", "jpeg"])
    location_input = st.text_input("Enter location:")
    user_phone = st.text_input("Enter your phone number for SMS updates (with country code):")

    # Submit button
    if st.button("Submit Report"):
        if not location_input:
            st.error("Location is required")
            return

        combined_text = ""

        # Process voice note if provided
        if voice_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                temp.write(voice_file.getbuffer())
                temp_path = temp.name

            st.info("Processing voice note...")
            voice_text = english_speech_to_text(temp_path)
            os.unlink(temp_path)
            combined_text += voice_text + "\n"
            st.success("Voice note processed")

        # Add manual text if provided
        if text_input:
            combined_text += text_input + "\n"

        # Process image if provided
        if image_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                temp.write(image_file.getbuffer())
                temp_path = temp.name

            st.info("Processing image...")
            image_text = process_image(temp_path)
            os.unlink(temp_path)
            combined_text += image_text + "\n"
            st.success("Image processed")

        # Display input summary
        if combined_text:
            st.write("**Combined Input Text:**")
            st.write(combined_text)

            # Identify disaster type
            with st.spinner("Analyzing disaster type..."):
                predicted_disaster, confidence = process_text(combined_text)
                st.write("**Predicted Disaster Type:**", predicted_disaster)
                st.write("**Confidence Score:**", confidence)

            # Extract entities
            with st.spinner("Extracting entities..."):
                entities = extract_entities(combined_text)
                st.write("**Extracted Entities:**")
                for entity_type, items in entities.items():
                    if items:
                        st.write(f"- {entity_type.capitalize()}: {', '.join(items)}")

            # Generate first-aid response
            with st.spinner("Generating first-aid information..."):
                first_aid_response = get_first_aid_response(predicted_disaster, combined_text)
                st.subheader("First Aid Response")
                st.write(first_aid_response)
        else:
            st.warning("No input text provided. Please provide text, voice, or image input.")

        # Geocode location
        with st.spinner("Getting location coordinates..."):
            lat, lon = get_lat_lon(location_input)
            if lat is None or lon is None:
                st.error("Could not determine coordinates from location. Please check your input.")
                return
            st.success(f"Location coordinates: {lat}, {lon}")

        # Save emergency report
        add_emergency(location_input, lat, lon, combined_text)
        st.success("Emergency report added to the database.")

        # Retrieve nearest resources
        resources = get_nearest_resources(lat, lon)
        if resources:
            st.subheader("Nearest Resources")
            for idx, res in enumerate(resources):
                st.write(f"{idx+1}. {res['name']} ({res['amenity']}) - {res['distance']:.2f} km away")

            # Map: Show current location and resources
            st.subheader("Map View")
            map_data = []
            # Add current location with label
            map_data.append({"lat": lat, "lon": lon, "label": "Emergency Location"})
            # Add resources with labels
            for res in resources:
                map_data.append({
                    "lat": res["latitude"],
                    "lon": res["longitude"],
                    "label": f"{res['name']} ({res['amenity']})"
                })
            # Create DataFrame for the map
            df_map = pd.DataFrame(map_data)
            st.map(df_map)
            st.map(latitude=lat , longitude= lon)

        # SMS notifications
        if user_phone:
            if st.button("Send SMS Notifications"):
                severity_text = entities.get("severity", ["Unknown"])[0] if entities.get("severity") else "Unknown"
                gov_status, user_status = notify_emergency_services(location_input, severity_text, user_phone)
                st.success("SMS notifications have been sent.")
                st.info(f"Status: {gov_status} | {user_status}")

def volunteer_registration_workflow():
    """Volunteer registration workflow"""
    st.header("Volunteer Registration")

    # Registration form
    with st.form("volunteer_registration"):
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        phone = st.text_input("Phone Number (with country code)")
        location = st.text_input("Location")
        speciality = st.selectbox(
            "Speciality",
            ["Medical", "Search & Rescue", "Fire Fighting", "Logistics", "Communications", "Other"]
        )
        other_speciality = st.text_input("If Other, please specify")

        submitted = st.form_submit_button("Register")

        if submitted:
            if not all([name, email, password, confirm_password, phone, location]):
                st.error("All fields are required")
                return

            if password != confirm_password:
                st.error("Passwords do not match")
                return

            # Get coordinates
            lat, lon = get_lat_lon(location)
            if lat is None or lon is None:
                st.error("Could not determine coordinates from location. Please check your input.")
                return

            # Final speciality value
            final_speciality = other_speciality if speciality == "Other" else speciality

            # Register volunteer
            success, result = register_volunteer(
                name, email, password, location, lat, lon, final_speciality, phone
            )

            if success:
                st.success(f"Registration successful! Your volunteer ID is: {result}")
                st.info("You can now log in using your email and password.")
            else:
                st.error(f"Registration failed: {result}")

def volunteer_login_workflow():
    """Volunteer login workflow"""
    st.header("Volunteer Login")

    # Login form
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if not email or not password:
            st.error("Email and password are required")
            return

        success, result = volunteer_login(email, password)

        if success:
            st.session_state.volunteer_id = result['id']
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error(result)

    # Check if logged in
    if st.session_state.get('logged_in', False):
        volunteer_dashboard()

def volunteer_dashboard():
    """Volunteer dashboard after login"""
    volunteer_id = st.session_state.volunteer_id

    # Get dashboard data
    volunteer, emergencies, resources, my_resources = get_volunteer_dashboard(volunteer_id)

    st.header(f"Welcome, {volunteer['name']}!")

    # Logout button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.pop('volunteer_id', None)
        st.rerun()

    # Dashboard tabs
    tabs = st.tabs(["Emergencies", "Resources", "My Resources", "Add Resource", "Upload Situation Report"])

    # Emergencies tab
    with tabs[0]:
        st.subheader("Nearby Emergencies")
        if emergencies:
            for i, emergency in enumerate(emergencies):
                with st.expander(f"Emergency {i+1}: {emergency['location']} - {emergency['distance']:.2f} km"):
                    st.write(f"**Location:** {emergency['location']}")
                    st.write(f"**Coordinates:** {emergency['latitude']}, {emergency['longitude']}")
                    st.write(f"**Report:** {emergency['text']}")
                    st.write(f"**Reported:** {emergency['timestamp']}")
        else:
            st.info("No emergencies reported in your area.")

        # Map of emergencies
        if emergencies:
            st.subheader("Emergency Map")
            map_data = []
            # Add volunteer location
            map_data.append({
                "lat": volunteer["latitude"],
                "lon": volunteer["longitude"]
            })
            # Add emergencies
            for emerg in emergencies:
                map_data.append({
                    "lat": emerg["latitude"],
                    "lon": emerg["longitude"]
                })
            df_map = pd.DataFrame(map_data)
            st.map(df_map)

    # Resources tab
    with tabs[1]:
        st.subheader("Nearby Resources")
        if resources:
            for i, resource in enumerate(resources):
                with st.expander(f"Resource {i+1}: {resource['name']} - {resource['distance']:.2f} km"):
                    st.write(f"**Name:** {resource['name']}")
                    st.write(f"**Type:** {resource['amenity']}")
                    st.write(f"**Coordinates:** {resource['latitude']}, {resource['longitude']}")
        else:
            st.info("No resources available in your area.")

        # Map of resources
        if resources:
            st.subheader("Resource Map")
            map_data = []
            # Add volunteer location
            map_data.append({
                "lat": volunteer["latitude"],
                "lon": volunteer["longitude"]
            })
            # Add resources
            for res in resources:
                map_data.append({
                    "lat": res["latitude"],
                    "lon": res["longitude"]
                })
            df_map = pd.DataFrame(map_data)
            st.map(df_map)

    # My Resources tab
    with tabs[2]:
        st.subheader("Resources I've Added")
        if my_resources:
            for i, resource in enumerate(my_resources):
                with st.expander(f"Resource {i+1}: {resource['name']}"):
                    st.write(f"**Name:** {resource['name']}")
                    st.write(f"**Type:** {resource['amenity']}")
                    st.write(f"**Coordinates:** {resource['latitude']}, {resource['longitude']}")
                    st.write(f"**Added on:** {resource['timestamp']}")
        else:
            st.info("You haven't added any resources yet.")

    # Add Resource tab
    with tabs[3]:
        st.subheader("Add New Resource")
        with st.form("add_resource_form"):
            amenity = st.text_input("Resource Type (e.g., Hospital, Shelter, Water Supply)")
            name = st.text_input("Resource Name")
            use_current_location = st.checkbox("Use my current location")

            if use_current_location:
                lat_input = volunteer["latitude"]
                lon_input = volunteer["longitude"]
                st.info(f"Using your location: {lat_input}, {lon_input}")
            else:
                location_input = st.text_input("Location")
                if location_input:
                    lat, lon = get_lat_lon(location_input)
                    if lat is not None and lon is not None:
                        lat_input = lat
                        lon_input = lon
                        st.success(f"Coordinates found: {lat_input}, {lon_input}")
                    else:
                        st.error("Could not determine coordinates. Please check the location.")
                        lat_input = st.number_input("Latitude", format="%.6f")
                        lon_input = st.number_input("Longitude", format="%.6f")
                else:
                    lat_input = st.number_input("Latitude", format="%.6f")
                    lon_input = st.number_input("Longitude", format="%.6f")

            submitted = st.form_submit_button("Add Resource")

            if submitted:
                if not amenity or not name:
                    st.error("Resource type and name are required")
                else:
                    add_resource(amenity, name, lat_input, lon_input, volunteer_id)
                    st.success("Resource added successfully!")

    # Upload Situation Report tab
    with tabs[4]:
        st.subheader("Upload Situation Report")
        pdf_file = st.file_uploader("Upload Situation PDF", type=["pdf"])
        if pdf_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
                temp.write(pdf_file.getbuffer())
                temp_path = temp.name

            with st.spinner("Generating summary from PDF..."):
                summary = generate_summary(temp_path)
                os.unlink(temp_path)

                st.subheader("PDF Summary")
                st.write(summary)

                with st.spinner("Generating response..."):
                    first_aid_response = get_first_aid_response("situation", summary)
                    st.subheader("First Aid Response")
                    st.write(first_aid_response)

                # Extract entities from summary
                entities = extract_entities(summary)
                if any(entities.values()):
                    st.subheader("Extracted Entities")
                    for entity_type, items in entities.items():
                        if items:
                            st.write(f"- {entity_type.capitalize()}: {', '.join(items)}")

                # Generate report
                if st.button("Generate Action Plan"):
                    st.success("Action plan generated successfully!")
                    st.info("Emergency services have been notified of your situation report.")

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'volunteer_id' not in st.session_state:
        st.session_state.volunteer_id = None

# Main application entry point
if __name__ == '__main__':
    init_session_state()
    main()
