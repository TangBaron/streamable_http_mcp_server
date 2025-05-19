手搓Streamable HTTP MCP Server项目

环境准备:

```
conda create -n mcp python=3.12
pip install uv
uv add openai fastapi 
```


代码准备：

1. 将server.py中的fetch_weather函数的key修改为你注册的心知天气api key
2. 将client.py中的Configuration类中的api key修改为你注册的DeepSeek-V3-0324 api key
