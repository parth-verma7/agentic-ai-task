import os
import pandas as pd
from app.config import GEMINI_API_KEY, HUGGINGFACE_DATASET_LINK, SEED
from app.logging_config import setup_logging
from app.agents import create_agents
from app.group_chat import create_group_chat
from app.report_generator import generate_report

logger = setup_logging()

try:
    logger.info("Loading dataset...")
    df = pd.read_csv(HUGGINGFACE_DATASET_LINK)
    df.to_csv("charts/emails.csv", index=False)
    logger.info("Dataset saved to 'charts/emails.csv'")
except Exception as e:
    logger.error(f"Failed to load or save the dataset: {e}")
    raise

config_list_gemini = [{"model": "gemini-1.5-flash", "api_key": GEMINI_API_KEY, "api_type": "google"}]

try:
    logger.info("Initializing agents...")
    assistant1, assistant2, user_proxy = create_agents(config_list_gemini, SEED)
    logger.info("Agents initialized successfully")
except Exception as e:
    logger.error(f"Error initializing agents: {e}")
    raise

try:
    logger.info("Initializing group chat...")
    groupchat, manager = create_group_chat([assistant1, assistant2, user_proxy], config_list_gemini, SEED)
    logger.info("Group chat initialized successfully")
except Exception as e:
    logger.error(f"Error initializing group chat: {e}")
    raise

user_query = '''
    Perform an in-depth exploratory data analysis (EDA) on the input dataset...
'''
user_proxy.send(f"Input Dataset {df} . -> " + user_query, recipient=manager, request_reply=True)

try:
    logger.info("Generating report...")
    all_files = os.listdir("./charts")
    charts = [file for file in all_files if file.endswith(".png") or file.endswith(".jpg")]
    generate_report(groupchat.messages, charts, "reports/README.md", logger)
except Exception as e:
    logger.error(f"Error generating report: {e}")
    raise
