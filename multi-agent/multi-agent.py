import functools
import getpass
import operator
import os
from typing import Annotated, Literal, TypedDict, Sequence

from langchain.chains.question_answering.map_reduce_prompt import messages
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import ToolMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please enter your {var}: ")

# 设置 OpenAI 和 Langchain API 密钥
_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# 可选：在 LangSmith 中添加追踪功能
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Multi-agent Collaboration"


tavily_tool = TavilySearchResults(max_results=5)
repl = PythonREPL()

@tool
def python_repl(code: Annotated[str, "The python code to execute your chart"]):
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

    return f"Successfully executed:\n```python\n{code}\n```\n"

def agent_node(state, agent, name):
    name = name.replace(" ", "_").replace("-", "_")

    result = agent.invoke(state)

    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}, name=name))


    return {
        "message": [result],
        "sender": name,
    }


def create_agent(llm, tools, tool_message: str, custom_notice: str = ""):
    # 定义智能体的提示模板，包含系统消息和工具信息
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                "\n{custom_notice}\n"
                " You have access to the following tools: {tool_names}.\n{tool_message}\n\n",
            ),
            MessagesPlaceholder(variable_name="messages"),  # 用于替换的消息占位符
        ]
    )

    prompt = prompt.partial(tool_message=tool_message, custon_notice=custom_notice)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    return prompt | llm.bind_tools(tools)


research_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
chart_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


research_agent = create_agent(
    research_llm,
    [tavily_tool],
    tool_message="Before using the search engine, carefully think through and clarify the query."
        "Then, conduct a single search that addresses all aspects of the query in one go",
    # custom_notice=(
    #     "Notice:\n"
    #     "Only gather and organize information. Do not generate code or give final conclusions, leave that for other assistants."
    #
    # )
)

research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

chart_agent = create_agent(
    chart_llm,
    [python_repl],
    tool_message="Create clear and user-friendly charts based on the provided data."
)

chart_node = functools.partial(agent_node, agent=chart_agent, name="Chart_Generator")

#
# def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
#     messages = state["messages"]
#     last_message = messages[-1]
#     if last_message.tool_calls:
#         return "tools"
#     else:
#         return "__end__"
#
# def call_model(state: MessagesState):
#     messages = state["messages"]
#     response = model_with_tools.invoke(messages)

tools = [python_repl, tavily_tool]

tool_node = ToolNode(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]



workflow = StateGraph(MessagesState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Chart_Generator", chart_node)
workflow.add_node("call_tool", tool_node)


# 路由器函数，用于决定下一步是执行工具还是结束任务
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    messages = state["messages"]  # 获取当前状态中的消息列表
    last_message = messages[-1]  # 获取最新的一条消息

    # 如果最新消息包含工具调用，则返回 "call_tool"，指示执行工具
    if last_message.tool_calls:
        return "call_tool"

    # 如果最新消息中包含 "FINAL ANSWER"，表示任务已完成，返回 "__end__" 结束工作流
    if "FINAL ANSWER" in last_message.content:
        return "__end__"

    # 如果既没有工具调用也没有完成任务，继续流程，返回 "continue"
    return "continue"


workflow.add_conditional_edges(
    "Researcher",
    router,
    {
        "continue": "Chart_Generator",
        "call_tool": "call_tool",
        "__end__": END,
    }
)

workflow.add_conditional_edges(
    "Chart_Generator",
    router,
    {
        "continue": "Researcher",
        "call_tool": "call_tool",
        "__end__": END,
    }
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {
        "Researcher": "Researcher",  # 如果 sender 是 Researcher，则返回给 Researcher
        "Chart_Generator": "Chart_Generator",  # 如果 sender 是 Chart_Generator，则返回给 Chart_Generator
    }
)


workflow.add_edge(START, "Researcher")
graph = workflow.compile()







