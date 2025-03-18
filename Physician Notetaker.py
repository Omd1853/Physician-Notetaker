# Import necessary libraries
import spacy
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_trf")

class MedicalTranscriptionAnalyzer:
    def __init__(self):
        # Initialize models
        self.sentiment_model = pipeline("text-classification", 
                                       model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Define medical entities
        self.medical_entities = {
            "SYMPTOM": ["pain", "discomfort", "ache", "stiffness", "difficulty", "trouble"],
            "TREATMENT": ["physiotherapy", "therapy", "medication", "painkillers", "sessions"],
            "DIAGNOSIS": ["whiplash", "injury", "strain", "sprain"],
            "PROGNOSIS": ["recovery", "improve", "better", "progress", "future"]
        }
        
    def preprocess_transcript(self, transcript):
        """Clean and structure the transcript"""
        # Split into doctor and patient segments
        segments = re.split(r'(?:\n\n|\*\*)(Physician|Patient|Doctor)(?:\:\*\*|\:)', transcript)
        
        # Process segments into a dialogue format
        dialogue = []
        for i in range(1, len(segments), 2):
            if i+1 < len(segments):
                speaker = segments[i].strip()
                text = segments[i+1].strip()
                dialogue.append({"speaker": speaker, "text": text})
        
        return dialogue
    
    def extract_medical_entities(self, text):
        """Extract medical entities using spaCy and custom rules"""
        doc = nlp(text)
        
        entities = {
            "Symptoms": [],
            "Treatment": [],
            "Diagnosis": [],
            "Prognosis": []
        }
        
        # Extract using spaCy named entities
        for ent in doc.ents:
            if ent.label_ == "PROBLEM":
                entities["Symptoms"].append(ent.text)
            elif ent.label_ == "TREATMENT":
                entities["Treatment"].append(ent.text)
        
        # Custom rule-based extraction
        for entity_type, keywords in self.medical_entities.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    # Find the complete phrase containing the keyword
                    sentences = sent_tokenize(text)
                    for sentence in sentences:
                        if keyword.lower() in sentence.lower():
                            # Extract noun phrases containing the keyword
                            doc = nlp(sentence)
                            for chunk in doc.noun_chunks:
                                if keyword.lower() in chunk.text.lower():
                                    if entity_type == "SYMPTOM":
                                        entities["Symptoms"].append(chunk.text)
                                    elif entity_type == "TREATMENT":
                                        entities["Treatment"].append(chunk.text)
                                    elif entity_type == "DIAGNOSIS":
                                        entities["Diagnosis"].append(chunk.text)
                                    elif entity_type == "PROGNOSIS":
                                        entities["Prognosis"].append(chunk.text)
        
        # Remove duplicates and clean up
        for key in entities:
            entities[key] = list(set(entities[key]))
            entities[key] = [item.strip() for item in entities[key] if len(item.strip()) > 0]
        
        return entities
    
    def extract_patient_name(self, transcript):
        """Extract patient name from transcript"""
        # Look for patterns like "Ms. X" or "Mr. X"
        name_match = re.search(r'(Ms\.|Mrs\.|Mr\.|Dr\.) ([A-Z][a-z]+)', transcript)
        if name_match:
            return name_match.group(1) + " " + name_match.group(2)
        return "Unknown"
    
    def summarize_text(self, text):
        """Summarize the text using extractive summarization"""
        sentences = sent_tokenize(text)
        
        # Create a similarity matrix
        stop_words = set(stopwords.words('english'))
        sentence_similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sentence_similarity_matrix[i][j] = self._sentence_similarity(sentences[i], sentences[j], stop_words)
        
        # Use NetworkX to find the most important sentences
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
        scores = nx.pagerank(sentence_similarity_graph)
        
        # Sort sentences by importance
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        
        # Select top 30% of the sentences
        summary_sentences_count = max(1, int(len(sentences) * 0.3))
        summary_sentences = [ranked_sentences[i][1] for i in range(summary_sentences_count)]
        
        # Sort the selected sentences by their original order
        summary_sentences_ordered = [s for s in sentences if s in summary_sentences]
        
        # Join the sentences
        summary = ' '.join(summary_sentences_ordered)
        
        return summary
    
    def _sentence_similarity(self, sent1, sent2, stop_words):
        """Calculate the cosine similarity between two sentences"""
        words1 = [word.lower() for word in nltk.word_tokenize(sent1) if word.lower() not in stop_words]
        words2 = [word.lower() for word in nltk.word_tokenize(sent2) if word.lower() not in stop_words]
        
        all_words = list(set(words1 + words2))
        
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
        for word in words1:
            vector1[all_words.index(word)] += 1
        
        for word in words2:
            vector2[all_words.index(word)] += 1
        
        return 1 - cosine_distance(vector1, vector2)
    
    def analyze_sentiment(self, patient_text):
        """Analyze the sentiment of patient text"""
        # For a real system, we'd use a custom trained model for medical sentiment
        # This is a simplified version using a general sentiment model
        result = self.sentiment_model(patient_text)[0]
        
        # Map the sentiment to medical context
        if result["label"] == "POSITIVE":
            sentiment = "Reassured"
        elif result["label"] == "NEGATIVE":
            sentiment = "Anxious"
        else:
            sentiment = "Neutral"
        
        # Detect intent based on keywords and patterns
        intent = self._detect_intent(patient_text)
        
        return {
            "Sentiment": sentiment,
            "Intent": intent
        }
    
    def _detect_intent(self, text):
        """Detect patient intent based on text patterns"""
        text = text.lower()
        
        if any(word in text for word in ["worry", "worried", "concerned", "afraid", "hope", "hoping"]):
            return "Seeking reassurance"
        elif any(word in text for word in ["pain", "hurt", "ache", "discomfort", "feel", "feeling"]):
            return "Reporting symptoms"
        elif any(word in text for word in ["can i", "should i", "what if", "will it", "is it"]):
            return "Asking questions"
        else:
            return "Providing information"
    
    def generate_soap_note(self, transcript, entities, sentiment_analysis):
        """Generate a SOAP note from the transcript"""
        dialogue = self.preprocess_transcript(transcript)
        
        # Extract patient statements for subjective section
        patient_statements = " ".join([segment["text"] for segment in dialogue if segment["speaker"] == "Patient"])
        
        # Extract physician statements for objective and plan sections
        physician_statements = " ".join([segment["text"] for segment in dialogue if segment["speaker"] == "Physician" or segment["speaker"] == "Doctor"])
        
        # Create SOAP note
        soap_note = {
            "Subjective": {
                "Chief_Complaint": ", ".join(entities["Symptoms"][:2]) if entities["Symptoms"] else "Not specified",
                "History_of_Present_Illness": self.summarize_text(patient_statements)
            },
            "Objective": {
                "Physical_Exam": self._extract_physical_exam(transcript),
                "Observations": self._extract_observations(transcript)
            },
            "Assessment": {
                "Diagnosis": ", ".join(entities["Diagnosis"]) if entities["Diagnosis"] else "Not specified",
                "Severity": self._determine_severity(transcript)
            },
            "Plan": {
                "Treatment": ", ".join(entities["Treatment"]) if entities["Treatment"] else "Not specified",
                "Follow-Up": self._extract_follow_up(transcript)
            }
        }
        
        return soap_note
    
    def _extract_physical_exam(self, transcript):
        """Extract physical exam details from the transcript"""
        # Look for sections after "[Physical Examination Conducted]"
        exam_match = re.search(r'\[.*Physical Examination.*\](.*?)(?=\n\n|\*\*|$)', transcript, re.DOTALL | re.IGNORECASE)
        if exam_match:
            return exam_match.group(1).strip()
        
        # Alternative approach: look for physician statements about examination
        exam_keywords = ["examination", "exam", "range of movement", "mobility", "tenderness"]
        sentences = sent_tokenize(transcript)
        exam_sentences = [s for s in sentences if any(keyword in s.lower() for keyword in exam_keywords)]
        
        if exam_sentences:
            return " ".join(exam_sentences)
        
        return "No physical examination details found"
    
    def _extract_observations(self, transcript):
        """Extract physician observations from the transcript"""
        # Look for descriptive statements about the patient's condition
        observation_keywords = ["appears", "seems", "looks", "shows", "exhibits", "demonstrates"]
        sentences = sent_tokenize(transcript)
        observation_sentences = [s for s in sentences if any(keyword in s.lower() for keyword in observation_keywords)]
        
        if observation_sentences:
            return " ".join(observation_sentences)
        
        return "No specific observations noted"
    
    def _determine_severity(self, transcript):
        """Determine the severity of the condition"""
        # Look for keywords indicating severity
        high_severity = ["severe", "serious", "critical", "major", "significant"]
        medium_severity = ["moderate", "considerable", "notable"]
        low_severity = ["mild", "minor", "slight", "improving"]
        
        text_lower = transcript.lower()
        
        if any(word in text_lower for word in high_severity):
            return "Severe"
        elif any(word in text_lower for word in medium_severity):
            return "Moderate"
        elif any(word in text_lower for word in low_severity):
            return "Mild, improving"
        else:
            return "Unspecified"
    
    def _extract_follow_up(self, transcript):
        """Extract follow-up instructions"""
        # Look for follow-up related statements
        follow_up_keywords = ["follow-up", "follow up", "return", "come back", "check", "appointment"]
        sentences = sent_tokenize(transcript)
        follow_up_sentences = [s for s in sentences if any(keyword in s.lower() for keyword in follow_up_keywords)]
        
        if follow_up_sentences:
            return " ".join(follow_up_sentences)
        
        return "No specific follow-up instructions found"
    
    def process_transcript(self, transcript):
        """Process the transcript and generate all outputs"""
        # 1. Medical NLP Summarization
        entities = self.extract_medical_entities(transcript)
        patient_name = self.extract_patient_name(transcript)
        
        # Extract current status
        current_status = self._extract_current_status(transcript)
        
        # Extract prognosis
        prognosis = self._extract_prognosis(transcript)
        
        # Structured medical report
        medical_report = {
            "Patient_Name": patient_name,
            "Symptoms": entities["Symptoms"],
            "Diagnosis": entities["Diagnosis"],
            "Treatment": entities["Treatment"],
            "Current_Status": current_status,
            "Prognosis": prognosis
        }
        
        # 2. Sentiment & Intent Analysis
        dialogue = self.preprocess_transcript(transcript)
        patient_text = " ".join([segment["text"] for segment in dialogue if segment["speaker"] == "Patient"])
        sentiment_analysis = self.analyze_sentiment(patient_text)
        
        # 3. SOAP Note Generation
        soap_note = self.generate_soap_note(transcript, entities, sentiment_analysis)
        
        return {
            "Medical_Report": medical_report,
            "Sentiment_Analysis": sentiment_analysis,
            "SOAP_Note": soap_note
        }
    
    def _extract_current_status(self, transcript):
        """Extract the current status of the patient"""
        # Look for current status indicators
        status_keywords = ["currently", "now", "at present", "at this time", "these days"]
        sentences = sent_tokenize(transcript)
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in status_keywords):
                return sentence
        
        # Alternative approach: look for "occasional" or "still" with symptoms
        for sentence in sentences:
            if "occasional" in sentence.lower() or "still" in sentence.lower():
                if any(symptom in sentence.lower() for symptom in self.medical_entities["SYMPTOM"]):
                    return sentence
        
        return "Not specified"
    
    def _extract_prognosis(self, transcript):
        """Extract the prognosis"""
        # Look for prognosis indicators
        prognosis_keywords = ["expect", "recovery", "future", "prognosis", "outlook"]
        sentences = sent_tokenize(transcript)
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in prognosis_keywords):
                return sentence
        
        return "Not specified"
    
    def save_report_to_json(self, report, filename):
        """Save the analysis report to a JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4, ensure_ascii=False)
            return True, f"Report successfully saved to {filename}"
        except Exception as e:
            return False, f"Error saving report to {filename}: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Sample transcript
    sample_transcript = """
    **Physician:** Good morning, Ms. Jones. How are you feeling today?
    
    **Patient:** Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    
    **Physician:** I understand you were in a car accident last September. Can you walk me through what happened?
    
    **Patient:** Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
    
    **Physician:** That sounds like a strong impact. Were you wearing your seatbelt?
    
    **Patient:** Yes, I always do.
    
    **Physician:** What did you feel immediately after the accident?
    
    **Patient:** At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
    
    **Physician:** Did you seek medical attention at that time?
    
    **Patient:** Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn't do any X-rays. They just gave me some advice and sent me home.
    
    **Physician:** How did things progress after that?
    
    **Patient:** The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
    
    **Physician:** That makes sense. Are you still experiencing pain now?
    
    **Patient:** It's not constant, but I do get occasional backaches. It's nothing like before, though.
    
    **Physician:** That's good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
    
    **Patient:** No, nothing like that. I don't feel nervous driving, and I haven't had any emotional issues from the accident.
    
    **Physician:** And how has this impacted your daily life? Work, hobbies, anything like that?
    
    **Patient:** I had to take a week off work, but after that, I was back to my usual routine. It hasn't really stopped me from doing anything.
    
    **Physician:** That's encouraging. Let's go ahead and do a physical examination to check your mobility and any lingering pain.
    
    [**Physical Examination Conducted**]
    
    **Physician:** Everything looks good. Your neck and back have a full range of movement, and there's no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
    
    **Patient:** That's a relief!
    
    **Physician:** Yes, your recovery so far has been quite positive. Given your progress, I'd expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
    
    **Patient:** That's great to hear. So, I don't need to worry about this affecting me in the future?
    
    **Physician:** That's right. I don't foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you're on track for a full recovery.
    
    **Patient:** Thank you, doctor. I appreciate it.
    
    **Physician:** You're very welcome, Ms. Jones. Take care, and don't hesitate to reach out if you need anything.
    """

    # Initialize the analyzer
    analyzer = MedicalTranscriptionAnalyzer()

    # Process the transcript
    results = analyzer.process_transcript(sample_transcript)

    # Save the results to a JSON file
    success, message = analyzer.save_report_to_json(results, "medical_report.json")
    print(message)
    
    # Print the results to console as well
    print("\nAnalysis Results:")
    print(json.dumps(results, indent=4))