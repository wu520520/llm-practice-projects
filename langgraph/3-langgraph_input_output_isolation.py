
from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph

# 定义状态字典
class MyState(TypedDict):
    prompt: str
    web_search_result: str
    rag_search_result: str
    final_answer_result: str
    other_key: str

class InputState(TypedDict):
    prompt: str

class OuputState(TypedDict):
    final_answer_result: str

# 获取状态图
graph = StateGraph(state_schema=MyState, input_schema=InputState, output_schema=OuputState)


# 定义节点
def web_search_node(dict:MyState):
    print("WEB --> ", dict)
    # 获取用户问题
    prompt = dict['prompt']
    # 根据用户的问题查询结果
    web_search_result = f"根据用户问题 {prompt} 查询得到的Web结果是: 明天会放假"
    # 将结果存放到状态字典中
    return {"web_search_result": web_search_result}


# 定义节点
def rag_search_node(dict:MyState):
    print("RAG --> ", dict)
    # 获取用户问题
    prompt = dict['prompt']
    # 根据用户的问题查询结果
    rag_search_result = f"根据用户问题 {prompt} 查询得到的Rag结果是: 明天确定会放假"
    # 将结果存放到状态字典中
    return {"rag_search_result": rag_search_result}

# 定义节点
def final_answer_node(dict:MyState):
    print("LLM --> ", dict)
    # 获取用户问题
    prompt = dict['prompt']
    # 获取web_search_result
    web_search_result = dict['web_search_result']
    # 获取rag_search_result
    rag_search_result = dict['rag_search_result']
    # 根据用户的问题查询结果
    final_answer_result = f"根据用户问题 {prompt} 和 搜索结果: {web_search_result} -- {rag_search_result} 得到的最终结果是: 明天一定以及确定会放假"
    # 将结果存放到状态字典中
    return {"final_answer_result": final_answer_result}

# 添加节点
graph.add_node("web", web_search_node)
graph.add_node("rag", rag_search_node)
graph.add_node("llm", final_answer_node)
# graph.add_node(web_search_node)  # graph.add_node("web_search_node", web_search_node)
# graph.add_node(rag_search_node)
# graph.add_node(final_answer_node)

# 添加边
graph.add_edge(START, 'web')
graph.add_edge(START, 'rag')
graph.add_edge('web', 'llm')
graph.add_edge('rag', 'llm')
graph.add_edge('llm', END)


# 编译图
compiled_graph = graph.compile()

graph_structure = compiled_graph.get_graph()
res = graph_structure.draw_ascii()
print(res)

# 运行状态图
print(compiled_graph.invoke({"prompt": "明天会放假吗", "other_key": "哈哈哈哈哈哈哈"}))








