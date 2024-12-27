import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HUGGINGFACE_DATASET_LINK = "hf://datasets/TrainingDataPro/email-spam-classification/email_spam.csv"
SEED = 25