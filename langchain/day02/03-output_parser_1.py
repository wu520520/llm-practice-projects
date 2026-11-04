
"""
需求: 让大模型找出0 - 100 之间的所有素数

数据必须的json格式，格式如下:{prime:[1,3,5,7,13,.....], count:10}

"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# 1. 定义json schema
# class Prime(BaseModel):
#     prime:list[int] = Field(description='0-100之间的所有素数')
#     count:int = Field(description='0-100之间的所有素数的个数')

# 2.获取json解析器
# json_parser = JsonOutputParser(pydantic_object=Prime)

# 3.获取json约束的提示词
# json_schema_prompt = json_parser.get_format_instructions()
# print(json_schema_prompt)


# 加载环境变量
load_dotenv()

# 创建客户端
client = ChatOpenAI(
    model="gpt-4o-mini",
    # openai系列的sdk，包括langchain，都需要这个/v1的后缀
    base_url=os.getenv("CLOSEAI_BASE_URL"),
    api_key=os.getenv("CLOSEAI_API_KEY"),
)

# 调用模型
result = client.invoke(
    input=[
        {
            "role": "system",
            "content": """
            STRICT OUTPUT FORMAT:
- Return only the JSON value that conforms to the schema. Do not include any additional text, explanations, headings, or separators.
- Do not wrap the JSON in Markdown or code fences (no ``` or ```json).
- Do not prepend or append any text (e.g., do not write "Here is the JSON:").
- The response must be a single top-level JSON value exactly as required by the schema (object/array/etc.), with no trailing commas or comments.

The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]} the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema (shown in a code block for readability only — do not include any backticks or Markdown in your output):
```
{"properties": {"prime": {"description": "0-100之间的所有素数", "items": {"type": "integer"}, "title": "Prime", "type": "array"}, "count": {"description": "0-100之间的所有素数的个数", "title": "Count", "type": "integer"}}, "required": ["prime", "count"]}
```
            """,
        },
        {
            "role": "user",
            "content": "生成1到100之间素数，并返回素数个数",
        }
    ]
)

print(result.content)