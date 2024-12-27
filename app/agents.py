from autogen import AssistantAgent, UserProxyAgent

def create_agents(config_list, seed):
    assistant1 = AssistantAgent(
        name="assistant1",
        system_message='''Provide the (i)th instruction for the data analysis task and pass control to assistant2...''',
        description='''Generates step-by-step instructions for data analysis tasks.''',
        llm_config={"config_list": config_list, "seed": seed},
        max_consecutive_auto_reply=3
    )
    assistant2 = AssistantAgent(
        name="assistant2",
        system_message='''Generate and execute Python code based on the (i)th instruction provided by assistant1...''',
        description='''Generates and executes Python code for data analysis tasks.''',
        llm_config={"config_list": config_list, "seed": seed},
        max_consecutive_auto_reply=3
    )
    user_proxy = UserProxyAgent(
        name="user_proxy",
        code_execution_config={"work_dir": "charts", "use_docker": False},
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        description="Handles dataset analysis."
    )
    return assistant1, assistant2, user_proxy
