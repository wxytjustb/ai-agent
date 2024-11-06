from typing import Annotated, get_type_hints

from langgraph.constants import START, END
from typing_extensions import TypedDict
import os
from langgraph.graph import StateGraph, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage


os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] ="chat-bot"




class State(TypedDict):
    messages: Annotated[list, add_messages]


# 通过打断点了解注解的工作机制
for name, typ in get_type_hints(State, include_extras=True).items():
    print(f"{name}: {typ}")
    print(typ.__metadata__)


graph_builder = StateGraph(State)
chat_model = ChatOpenAI(model="gpt-4o-mini")


def chatbot(state: State):
    return {"messages": [chat_model.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)



graph = graph_builder.compile()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")  # 打印告别信息
        break  # 结束循环，退出聊天

    for event in graph.stream({"messages": ("user", user_input)}):

        # 遍历每个事件的值
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                # 如果消息是 BaseMessage 类型，则打印机器人的回复
                print("Assistant:", value["messages"][-1].content)






