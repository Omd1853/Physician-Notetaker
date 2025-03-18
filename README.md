# Physician-Notetaker

Medical Transcription Analyzer
Overview
The Medical Transcription Analyzer is a Python-based tool designed to process medical transcripts and extract structured information, perform sentiment and intent analysis, and generate SOAP notes. It leverages advanced Natural Language Processing (NLP) techniques using libraries such as spaCy , NLTK , and Hugging Face Transformers .

This tool is particularly useful for healthcare professionals, researchers, and developers working with medical transcriptions to automate the extraction of key insights and generate standardized reports.

Features
Medical Entity Extraction :
Extracts symptoms, treatments, diagnoses, and prognoses from medical transcripts.
Combines spaCy's named entity recognition (NER) with custom rule-based extraction for improved accuracy.
Sentiment and Intent Analysis :
Analyzes patient sentiment (e.g., reassured, anxious, neutral).
Detects patient intent (e.g., reporting symptoms, seeking reassurance, asking questions).

Summarization :
Generates concise summaries of patient and physician statements using extractive summarization techniques.

SOAP Note Generation :
Automatically generates structured SOAP notes (Subjective, Objective, Assessment, Plan) based on the transcript.

Severity Determination :
Determines the severity of the patient's condition based on keywords in the transcript.

JSON Report Export :
Saves the analysis results into a JSON file for easy integration with other systems.
Installation
Prerequisites
Python 3.8 or higher

Install the required dependencies using the following command:
pip install -r requirements.txt

Required Libraries
The project depends on the following Python libraries:

spacy : For NLP tasks like named entity recognition.
transformers : For sentiment analysis using pre-trained models.
nltk : For tokenization, stopwords, and similarity calculations.
networkx : For graph-based summarization.
numpy : For numerical operations.

To install spaCy's English language model, run:
python -m spacy download en_core_web_trf

Usage
Running the Script
Clone the repository or copy the code into your local environment.
Place your medical transcript in a text file (e.g., transcript.txt).
Run the script:

python medical_transcription_analyzer.py

Example Input
The script includes a sample transcript for demonstration purposes. You can replace it with your own transcript by modifying the sample_transcript variable in the script.

Output
The script generates the following outputs:

Medical Report :
Structured report containing patient name, symptoms, diagnosis, treatment, current status, and prognosis.

Sentiment Analysis :
Sentiment and intent analysis of the patient's statements.

SOAP Note :
A standardized SOAP note generated from the transcript.

JSON File :
All outputs are saved in a JSON file (medical_report.json) for further use.


Code Structure
Classes and Methods
MedicalTranscriptionAnalyzer :
The main class responsible for processing medical transcripts.
Key methods include:
preprocess_transcript: Cleans and structures the transcript into dialogue format.

extract_medical_entities: Extracts medical entities like symptoms, treatments, and diagnoses.

analyze_sentiment: Performs sentiment and intent analysis on patient statements.

generate_soap_note: Generates a SOAP note from the transcript.

summarize_text: Creates a summary of the transcript using extractive summarization.

save_report_to_json: Saves the analysis results to a JSON file.

Helper Methods :
_sentence_similarity: Calculates cosine similarity between two sentences.
_extract_physical_exam: Extracts physical examination details.
_determine_severity: Determines the severity of the patient's condition.
_extract_follow_up: Extracts follow-up instructions.
Sample Output
JSON Report (medical_report.json)

{
  "Medical_Report": {
    "Patient_Name": "Ms. Jones",
    "Symptoms": ["pain in my neck and back", "discomfort"],
    "Diagnosis": ["whiplash injury"],
    "Treatment": ["physiotherapy", "painkillers"],
    "Current_Status": "I do get occasional backaches. It's nothing like before, though.",
    "Prognosis": "I'd expect you to make a full recovery within six months of the accident."
  },
  "Sentiment_Analysis": {
    "Sentiment": "Reassured",
    "Intent": "Reporting symptoms"
  },
  "SOAP_Note": {
    "Subjective": {
      "Chief_Complaint": "pain in my neck and back, discomfort",
      "History_of_Present_Illness": "I still have some discomfort now and then... The first four weeks were rough..."
    },
    "Objective": {
      "Physical_Exam": "Everything looks good. Your neck and back have a full range of movement...",
      "Observations": "No specific observations noted"
    },
    "Assessment": {
      "Diagnosis": "whiplash injury",
      "Severity": "Mild, improving"
    },
    "Plan": {
      "Treatment": "physiotherapy, painkillers",
      "Follow-Up": "If anything changes or you experience worsening symptoms, you can always come back for a follow-up."
    }
  }
}

Contributing
Contributions are welcome! If you would like to contribute to this project, please follow these steps:

Fork the repository.
Create a new branch for your feature or bug fix.
Submit a pull request with a detailed description of your changes.
License
This project is licensed under the MIT License . See the LICENSE file for more details.

Contact
For questions or feedback, please contact:

Email: onkardharmadhikari3@gmail.com
GitHub:https://github.com/Omd1853
