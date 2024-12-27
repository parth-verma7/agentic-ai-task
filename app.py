import os
import autogen
import logging
import pandas as pd
from autogen.code_utils import content_str
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

load_dotenv()
gemini_api_key=os.getenv('GEMINI_API_KEY')

logging.basicConfig(
    filename='logs/system.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    ## INPUT YOUR HUGGING FACE DATASET LINK HERE
    logger.info("Loading dataset from huggingface link")
    df = pd.read_csv("hf://datasets/TrainingDataPro/email-spam-classification/email_spam.csv")
    df.to_csv("charts/emails.csv", index=False)
    logger.info("Dataset successfully saved as 'charts/emails.csv'")

except Exception as e:
    logger.error(f"Failed to load or save the dataset: {e}")
    raise

## CONFIGURE YOUR GEMINI MODEL THAT YOU PREFER TO USE
config_list_gemini = [
    {
        "model": "gemini-1.5-flash",
        "api_key": gemini_api_key,
        "api_type": "google"
    }
]

seed = 25

try:
    ## CREATE MULTIPLE ASSISTANTS TO AUTOMATE THE OPERATIONS
    logger.info("Initializing assistant agents")

    assistant1 = AssistantAgent(
        name="assistant1",
        system_message='''Provide the (i)th instruction for the data analysis task and pass control to assistant2. 
                            Ensure the instruction is clear, sequential, and builds on any prior inferences made by assistant2.
                                Wait for assistant2 to generate and execute the corresponding code before crafting the next (i+1)th instruction based on the insights and outputs provided.
                                    If assistant2 falls into some error ask it to fix that error by generating the previous code again.''',
        description='''I generate step-by-step instructions for data analysis tasks, ensuring each instruction logically progresses toward the analysis goal. I collaborate with assistant2 for seamless task execution.''',
        llm_config={
            "config_list": config_list_gemini,
            "seed": seed
        },
        max_consecutive_auto_reply=3
    )

    assistant2 = AssistantAgent(
        name="assistant2",
        system_message='''Generate and execute Python code based on the (i)th instruction provided by assistant1. The code should generate visualizations (e.g., charts or graphs) using `matplotlib` or other libraries, and save these visualizations to files (e.g., as PNG or JPEG). After generating the charts, summarize key insights from the results. Once execution is complete, provide the file paths of the saved charts along with inferences to assistant1 for further analysis.''',
        description='''I specialize in generating and executing Python code for data analysis tasks, which includes creating, saving, and providing visualizations. I return key insights and saved chart file paths to guide subsequent steps in the process.''',
        llm_config={
            "config_list": config_list_gemini,
            "seed": seed
        },
        max_consecutive_auto_reply=3
    )
    logger.info("Assistant agents initialized successfully")

except Exception as e:
    logger.error(f"Failed to initialize assistant agents: {e}")
    raise
try:
    user_proxy = UserProxyAgent(
        "user_proxy",
        code_execution_config={"work_dir": "charts", "use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        is_termination_msg=lambda x: content_str(x.get("content")).find("TERMINATE") >= 0
        or content_str(x.get("content")) == "",
        description="Responsible for data analysis of the dataset provided."
    )
    logger.info("UserProxyAgent initialized successfully")
except Exception as e:
    logger.error(f"Error initializing UserProxyAgent: {e}")
    raise

# CREATING A GROUP CHAT THAT AUTOMATICALLY DETERMINES WHICH AGENT SHOULD ACT
try:
    groupchat = autogen.GroupChat(agents=[assistant1, assistant2, user_proxy], messages=[], max_round=10, speaker_selection_method="round_robin")
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list_gemini, "seed": seed})
    logger.info("Group chat initialized successfully")
except Exception as e:
    logger.error(f"Error initializing group chat: {e}")
    raise

# MAIN TASK DEFINING QUERY FOR AGENTS
user_query = f'''
    Perform an in-depth exploratory data analysis (EDA) on the input dataset. \n 
    Examine patterns, trends, and statistical significance, including class imbalances, dataset format, and the number of conversation turns. \n
    Generate graphs and charts to visually represent data patterns, highlight key findings, and detect anomalies. \n
    Provide a detailed analysis report with insights and visualizations. \n
'''

try:
    user_proxy.send(
        f"Input Dataset {df} . -> " + user_query,
        recipient=manager,
        request_reply=True
    )
    logger.info("User query sent successfully")
except Exception as e:
    logger.error(f"Error sending user query: {e}")
    raise

responses = groupchat.messages
responses[0]['content'] = user_query

try:
    all_files = os.listdir("./charts")
    charts = [file for file in all_files if file.endswith(".png") or file.endswith(".jpg")]
    logger.info(f"Charts identified: {charts}")
except Exception as e:
    logger.error(f"Error reading charts directory: {e}")
    raise

# FUNCTION TO GENERATE THE FINAL REPORT
def generate_report(response, output_file):
    try:
        with open(output_file, "w") as f:
            f.write("# Exploratory Data Analysis Report\n\n")
            for response in responses:
                name = response['name']
                content = response['content']
                content = content.replace("exitcode: 0 (execution succeeded)", " ")
                if name == "user_proxy":
                    f.write("## User Proxy\n")
                    f.write(content + "\n\n")

                elif name == "assistant1":
                    f.write("## Assistant 1\n")
                    f.write(content + "\n")
                    if ".jpg" in content or ".png" in content:
                        for chart in charts:
                            if chart in content:
                                f.write(f"![{chart}](./charts/{chart})\n")
                    f.write("\n\n")

                elif name == "assistant2":
                    f.write("## Assistant 2\n")
                    f.write(content + "\n")
                    if ".jpg" in content or ".png" in content:
                        for chart in charts:
                            if chart in content:
                                f.write(f"![{chart}](./charts/{chart})\n")
                    f.write("\n\n")
        logger.info(f"Report generated and saved to {output_file}")
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise

generate_report(responses, "reports/README.md")