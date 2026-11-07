from mcp.server import FastMCP

# 创建 MCP 实例
mcp = FastMCP('demo')

# 定义工具
@mcp.tool()
def add(a:int, b:int):
    return a + b


# 定义静态资源
@mcp.resource("greeting://default")
def get_greeting() -> str:
    return "Hello from static resource!"

# 定义提示词资源
@mcp.prompt()
def get_prompt(kw: str):
    return f'请给我将一个关于{kw}的笑话'


if __name__ == '__main__':
    mcp.settings.port=9999
    mcp.run(transport='streamable-http')