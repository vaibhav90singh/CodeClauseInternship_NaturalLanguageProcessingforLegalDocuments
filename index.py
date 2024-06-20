import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Span
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import re

# Load Spacy and NLTK models
nlp = spacy.load('en_core_web_sm')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    return text

# Function to extract key clauses (using SpaCy)
def extract_key_clauses(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    return sentences

# Function to generate summary (using transformers pipeline)
def generate_summary(text):
    summarizer = pipeline("summarization")
    summary_text = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary_text

# Example legal document text (replace with your own document)
legal_document = """
This Agreement ("Agreement") is made and entered into this 1st day of January, 2024, by and between Company X, a corporation organized and existing under the laws of the State of California, with its principal office located at 123 Main St, City, State, and Client Y, an individual residing at 456 Oak Ave, City, State.

WHEREAS, the parties desire to enter into an agreement for the provision of legal services by Company X to Client Y;

NOW, THEREFORE, in consideration of the mutual covenants and agreements herein contained, the parties hereto agree as follows:

1. Engagement of Services. Client Y hereby engages Company X to provide legal services related to intellectual property matters, including but not limited to trademark registration and protection.

2. Term. This Agreement shall commence on the date first above written and shall continue until terminated as provided herein.

3. Compensation. Client Y agrees to pay Company X a fee of $10,000 per month for the legal services provided under this Agreement.

4. Termination. Either party may terminate this Agreement upon 30 days' written notice to the other party.

5. Governing Law. This Agreement shall be governed by and construed in accordance with the laws of the State of California.

IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the date first above written.

[Signatures follow]
"""

# Preprocess text
legal_document = preprocess_text(legal_document)

# Extract key clauses
key_clauses = extract_key_clauses(legal_document)

# Generate summary from key clauses
summary = generate_summary(legal_document)

# Print key clauses and summary
print("Key Clauses:")
for i, clause in enumerate(key_clauses):
    print(f"{i+1}. {clause.text}")

print("\nSummary:")
print(summary)
