import os
import autogen
import pandas as pd
from autogen.code_utils import content_str
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv
load_dotenv()

gemini_api_key=os.getenv('GEMINI_API_KEY')
df = pd.read_csv("hf://datasets/TrainingDataPro/email-spam-classification/email_spam.csv")
df.to_csv("charts/emails.csv", index=False)

config_list_gemini = [
    {
        "model": "gemini-1.5-flash",
        "api_key": gemini_api_key,
        "api_type": "google"
    }
]

seed = 25

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
    max_consecutive_auto_reply=2
)

assistant2 = AssistantAgent(
    name="assistant2",
    system_message='''Generate and execute Python code based on the (i)th instruction provided by assistant1. The code should generate visualizations (e.g., charts or graphs) using `matplotlib` or other libraries, and save these visualizations to files (e.g., as PNG or JPEG). After generating the charts, summarize key insights from the results. Once execution is complete, provide the file paths of the saved charts along with inferences to assistant1 for further analysis.''',
    description='''I specialize in generating and executing Python code for data analysis tasks, which includes creating, saving, and providing visualizations. I return key insights and saved chart file paths to guide subsequent steps in the process.''',
    llm_config={
        "config_list": config_list_gemini,
        "seed": seed
    },
    max_consecutive_auto_reply=2
)

user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config={"work_dir": "charts", "use_docker": False},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: content_str(x.get("content")).find("TERMINATE") >= 0
    or content_str(x.get("content")) == "",
    description="Responsible for data analysis of the dataset provided."
)

groupchat = autogen.GroupChat(agents=[assistant1, assistant2, user_proxy], messages=[], max_round=20, speaker_selection_method="round_robin")
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list_gemini, "seed": seed})

user_query = f'''
    Perform an in-depth exploratory data analysis (EDA) on the input dataset - {df}. 
    Examine patterns, trends, and statistical significance, including class imbalances, dataset format, and the number of conversation turns. 
    Generate graphs and charts to visually represent data patterns, highlight key findings, and detect anomalies. 
    Provide a detailed analysis report with insights and visualizations.
'''

user_proxy.send(
    user_query, 
    recipient=manager, 
    request_reply=True
)

response = groupchat.messages

import pickle
with open('response.pkl', 'wb') as f:
    pickle.dump(response, f)