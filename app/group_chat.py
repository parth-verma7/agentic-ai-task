import autogen

def create_group_chat(agents, config_list, seed):
    groupchat = autogen.GroupChat(
        agents=agents,
        messages=[],
        max_round=10,
        speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": config_list, "seed": seed}
    )
    return groupchat, manager
