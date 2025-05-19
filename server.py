import argparse
import asyncio
import json
from typing import Any, AsyncIterator

import requests
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import StreamingResponse
import uvicorn

SERVER_NAME = "WeatherServer" # 定义服务器名称
SERVER_VERSION = "1.0.0" #定义服务器版本
PROTOCOL_VERSION = "2025-05-16" #定义协议版本号

# 编写请求天气函数
async def fetch_weather(city: str):
    try:
        url="https://api.seniverse.com/v3/weather/now.json"
        params={
            "key": "你注册的心知天气api key",
            "location": city,
            "language": "zh-Hans",
            "unit": "c"
        }
        response = requests.get(url, params=params)
        temperature = response.json()['results'][0]['now']
    except Exception:
        return "error"
    return json.dumps(temperature)

# 构造天气流式响应, 使用生成器yield流式返回天气内容（符合MCP协议的逐行JSON）
async def stream_weather(city: str, req_id: int | str):
    yield json.dumps({"jsonrpc": "2.0", "id": req_id, "stream": f"查询 {city} 天气中…"}).encode() + b"\n"

    await asyncio.sleep(0.3)
    data = await fetch_weather(city)

    if data == "error":
        yield json.dumps({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": data["error"]}}).encode() + b"\n"
        return
    
    yield json.dumps({
        "jsonrpc": "2.0", "id": req_id,
        "result": {
            "content": [
                {"type": "text", "text": data}
            ],
            "isError": False
        }
    }).encode() + b"\n"

TOOLS_REGISTRY = {
    "tools": [
        {
            "name": "get_weather",
            "description": "用于进行天气信息查询的函数，输入城市英文名称，即可获得当前城市天气信息。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Hangzhou'"
                    }
                },
                "required": ["city"]
            }
        }
    ],
    "nextCursor": None
}

app = FastAPI(title="Weather HTTP-Streamble MCP SERVER")

@app.get("/mcp")
async def mcp_initialize_via_get():
    #  GET 请求也执行了 initialize 方法
    return {
        "jsonrpc": "2.0",
        "id": 0,
        "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "streaming": True,
                "tools": {"listChanged": True}
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION
            },
            "instructions": "Use the get_weather tool to fetch weather by city name."
        }
    }

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()
        # 打印客户端的请求内容
        print("收到请求：", json.dumps(body, ensure_ascii=False, indent=2))
    except Exception:
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
    req_id = body.get("id", 1)
    method = body.get("method")
    
    #打印当前方法类型
    print(f"方法: {method}")

    if method == "notifications/initialized":
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    
    if method is None:
        return {"jsonrpc": "2.0", "id": req_id, "result": {"status": "MCP server online."}}
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0", 
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "streaming": True,
                    "tools": {"listChanged": True}
                },
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": "Use the get_weather tool to fetch weather by city name."
            }
        }
    
    if method == "tools/list":
        print(json.dumps(TOOLS_REGISTRY, indent=2, ensure_ascii=False))
        return {"jsonrpc": "2.0", "id": req_id, "result": TOOLS_REGISTRY}
    
    if method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name != "get_weather":
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Unknown tool"}}

        city = args.get("city")
        if not city:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Missing city"}}

        return StreamingResponse(stream_weather(city, req_id), media_type="application/json")

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main() -> None:
    parser = argparse.ArgumentParser(description="Weather MCP HTTP-Stream")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()