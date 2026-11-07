"""
start -> NodeA --> NodeB --> End
NodeA 和 NodeB 操作同一份数据
"""
from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages


# 自定义Reducer
# def add_message(message_list_left:str,message_list_right:str):
#   print("="*20)
#   print("正在执行add_message")
#   print('左边的', message_list_left)
#   print('右边的', message_list_right)
#   print("="*20)
#   return message_list_left + '-' + message_list_right

class MyState(TypedDict):
    # name: Annotated[str, lambda old, new: old + '-' + new]
    name: Annotated[str, add_messages]


graph = StateGraph(MyState)
def node_a(dict: MyState):
    print(dict)
    return {'name': 'hello'}

def node_b(dict: MyState):
    print(dict)
    return {'name': 'world'}


graph.add_node(node_a)
graph.add_node(node_b)

graph.add_edge(START, 'node_a')
graph.add_edge('node_a', 'node_b')
graph.add_edge('node_b',  END)


graph = graph.compile()

print(graph.invoke({'name': '哈哈'}))