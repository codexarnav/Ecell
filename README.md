# Disaster Management Application

## Overview
The **Disaster Management Application** is a comprehensive tool designed to assist in emergency situations by providing real-time information, resource management, and communication capabilities. The application leverages AI models and APIs to process and analyze data from multiple sources, including text, audio, and images, to provide actionable insights during emergencies.

---

## Features
- **Emergency Reporting:** Users can report emergencies through text, voice recordings, and images. The application processes these inputs to identify the type of emergency and extract relevant entities.
- **First Aid Information:** Provides first-aid measures based on the type of disaster reported.
- **Resource Management:** Volunteers can add and manage resources such as hospitals, shelters, and water supplies.
- **Volunteer Dashboard:** Volunteers can view nearby emergencies, resources, and manage their contributions.
- **SMS Notifications:** Sends SMS notifications to emergency services and users with updates on the situation.
- **PDF Summarization:** Generates summaries from uploaded PDF documents and provides relevant first-aid responses.

---

## Technologies Used
- **Python:** Primary programming language.
- **Streamlit:** For building the web application interface.
- **SQLite:** For database management.
- **Hugging Face Transformers:** For natural language processing tasks such as summarization and entity extraction.
- **Twilio:** For sending SMS notifications.
- **Google GenerativeAI:** For generating first-aid responses.
- **OpenCage Geocoder:** For geocoding location names to coordinates.
- **Librosa:** For audio processing.
- **PyPDF2:** For PDF text extraction.

---

## Installation
### Clone the Repository:
```bash
git clone https://github.com/your-username/disaster-management-app.git
cd disaster-management-app
```

### Set Up a Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Set Up Environment Variables:
Create a `.env` file in the root directory and add the necessary API keys and configurations:
```plaintext
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
HF_API_TOKEN=your_huggingface_api_token
GEMINI_API_KEY=your_gemini_api_key
OPENCAGE_API_KEY=your_opencage_api_key
```

### Run the Application:
```bash
streamlit run app.py
```

---

## Usage
### User Workflow:
1. Navigate to the **"User"** section.
2. Enter emergency details, record a voice note, or upload an image.
3. Provide location and phone number for SMS updates.
4. Submit the report to receive first-aid information and nearby resources.

### Volunteer Registration:
1. Navigate to the **"Volunteer Registration"** section.
2. Fill in the registration form with your details.
3. Submit the form to register as a volunteer.

### Volunteer Login:
1. Navigate to the **"Volunteer Login"** section.
2. Enter your email and password to log in.
3. Access the volunteer dashboard to view and manage emergencies and resources.

### Volunteer Dashboard:
- View nearby emergencies and resources.
- Add new resources and manage existing ones.
- Upload situation reports in PDF format for summarization and analysis.

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourFeatureName
   ```
5. Open a pull request.

---

## License
This project is licensed under the **MIT License** - see the `LICENSE` file for details.

---

## Acknowledgments
- Thanks to **Hugging Face** for providing pre-trained models.
- Thanks to **Twilio** for their SMS API.
- Thanks to **Google** for their GenerativeAI API.
- Thanks to **OpenCage** for their geocoding service.


