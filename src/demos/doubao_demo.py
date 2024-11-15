import os
from volcenginesdkarkruntime import Ark

client = Ark(api_key=os.environ.get("ARK_API_KEY"))

print("----- standard request -----")
completion = client.chat.completions.create(
    model="ep-20241115095025-cw5x8",
    messages = [
        {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
        {"role": "user", "content": "常见的十字花科植物有哪些？"},
    ],
)
print(completion.choices[0].message.content)