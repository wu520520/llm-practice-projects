import asyncio
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
load_dotenv()
# 1. 定义工具
@tool
def get_weather(city: str) -> str:
      """查询天气"""
      return f"{city}的天气晴朗，气温25度。"

@tool
def transfer_money(amount: int, to_account: str) -> str:
      """转账工具 (敏感操作)"""
      print(f"!!! 正在执行转账: {amount} -> {to_account} !!!")
      return f"成功转账 {amount} 元给 {to_account}。"

# 2. 初始化模型
llm = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=os.getenv("CLOSEAI_API_KEY"),
    base_url=os.getenv("CLOSEAI_BASE_URL")
)

# 3. 配置 HumanInTheLoopMiddleware
# 我们希望在调用 transfer_money 时暂停，让用户审核
# True 表示允许所有操作 (approve, edit, reject)
interrupt_config = {
      "transfer_money": True,
      "get_weather": False   # False 表示自动批准，不中断
}
# 创建人工审核中间件
hitl_middleware = HumanInTheLoopMiddleware(interrupt_on=interrupt_config)

# 4. 创建 Agent
# 注意：使用中断功能必须配置 checkpointer，因为中断需要保存状态
checkpointer = InMemorySaver()

agent = create_agent(
      model=llm,
      tools=[get_weather, transfer_money],
      middleware=[hitl_middleware],
      checkpointer=checkpointer,
)

async def run_demo():
      print("=== HumanInTheLoopMiddleware 演示 ===")
      print("场景：用户让 Agent 转账，Agent 在执行前会暂停等待批准。")
      
      thread_id = "thread-1"
      config = {"configurable": {"thread_id": thread_id}}
      
      # 第一步：用户发出指令,可以调整成查询天气
      print("\n[User]: 请帮我转账 100 元给 Alice")
      
      # 使用 ainvoke 或 stream 运行
      # 如果遇到中断，LangGraph 会暂停并保存状态
      # 我们需要在一个循环中处理这种情况，但为了演示清晰，我们分步执行
      
      # 第一次运行：Agent 思考 -> 决定调用 transfer_money -> Middleware 拦截 -> 中断
      # 注意：create_agent 返回的是一个 CompiledGraph，它的行为和标准 LangGraph 一致
      
      # 我们用一个循环来模拟持续交互，并处理潜在的中断
      current_input = {"messages": [{"role": "user", "content": "请帮我转账 100 元给 Alice"}]}
      # 人工指令
      current_command = None

      while True:
            try:
                  # 如果有 resume command，就用它；否则用 input
                  if current_command:
                        result = await agent.ainvoke(current_command, config=config)
                        current_command = None # 重置
                  else:
                        if not current_input:
                              break
                        result = await agent.ainvoke(current_input, config=config)
                        current_input = None # 处理完了

                  # 打印结果消息
                  if "messages" in result:
                        for msg in result["messages"]:
                              if hasattr(msg, "tool_calls") and msg.tool_calls:
                                      print(f"[Agent]: 我想调用工具: {msg.tool_calls}")
                              if msg.type == "tool":
                                      print(f"[Tool Output]: {msg.content}")
                              if msg.type == "ai" and not msg.tool_calls:
                                      print(f"[Agent]: {msg.content}")
            except Exception as e:
                  print(f"发生错误: {e}")
                  pass
            # 检查是否中断，当中断时，result当中有__interrupt__键
            if "__interrupt__" in result:
                  interrupt_value = result["__interrupt__"][0].value
                  print(f"\n!!! 检测到中断 (Middleware 拦截) !!!")
                  print(f"中断详情: {interrupt_value}")
                  
                  # 这里模拟用户决策
                  print("\n[System]: 请审核上述操作 (approve/reject/edit):")
                  # 模拟用户输入 "approve"
                  decision_type = "approve" 
                  print(f"[User]: {decision_type}")
                  
                  # 构建回复
                  # Middleware 期望的格式是 {"decisions": [{"type": "approve", ...}]}
                  decisions = []
                  action_requests = interrupt_value.get("action_requests", [])
                  
                  for req in action_requests:
                        print(f"   - 批准操作: {req['name']}")
                        decisions.append({"type": "approve"})
                  
                  # 人工审核结果：通过构造 resume command，并传入的方式传入给agent
                  current_command = Command(resume={"decisions": decisions})
                  print("[System]: 恢复执行...")
                  continue
            
            # 如果没有中断，说明任务完成
            break

if __name__ == "__main__":
      asyncio.run(run_demo())