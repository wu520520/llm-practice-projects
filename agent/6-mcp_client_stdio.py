import asyncio

from mcp import StdioServerParameters, stdio_client, ClientSession


async def stdio_run():
    server_params = StdioServerParameters(
        command=r'D:\DevTools\miniconda3\envs\langchain-2603\python.exe',
        args=['./5-mcp_server_stdio.py']
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()
            print('和服务器建立了连接')

            # 查看所有工具
            tools = await session.list_tools()
            # print(tools)
            # 调用工具
            tool_res = await session.call_tool("add", {"a": 1, "b": 2})
            print(tool_res)

            # 查看资源
            resoures = await session.list_resources()
            # print(resoures)
            # 获取资源
            resource_res = await session.read_resource("greeting://default")
            print(resource_res)

            # 查看提示词
            prompts = await session.list_prompts()
            # print(prompts)
            # 获取提示词
            prompt_res = await session.get_prompt('get_prompt', {'kw': '程序员'})
            print(prompt_res)


if __name__ == '__main__':
    asyncio.run(stdio_run())