import json
import os
import dashscope
from dashscope import MultiModalConversation
import requests
from PIL import Image
from io import BytesIO

# Set the base HTTP API URL for DashScope
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

# Input image path
img1 = "./test.jpeg"

messages = [
    {
        "role": "user",
        "content": [
            {"image": img1},
            {"text": "Add some coconuts trees on the road"}
        ]
    }
]

# API Key
api_key = ""

response = MultiModalConversation.call(
    api_key=api_key,
    model="qwen-image-edit-plus",
    messages=messages,
    stream=False,
    n=1,
    watermark=False,
    negative_prompt=" ",
    prompt_extend=True,
    size="1919*1280"
)

print(response)

if response.status_code == 200:
    for i, content in enumerate(response.output.choices[0].message.content):
        print(f"URL of output image {i+1}: {content['image']}")
else:
    print(f"HTTP status code: {response.status_code}")
    print(f"Error code: {response.code}")
    print(f"Error message: {response.message}")

url = content['image']
img_bytes = requests.get(url).content
img = Image.open(BytesIO(img_bytes))
img.save(f"output_{i+1}.png")
