#!/usr/bin/env python3
from flask import Flask, request, jsonify
import time
import random

app = Flask(__name__)

@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "nvidia/OpenMath-Nemotron-7B",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nvidia"
            }
        ]
    })

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 256)
    
    prompt = messages[-1]['content'] if messages else ""
    
    response_text = f"Solution: x = 4 (mock response for: {prompt[:50]})"
    
    latency = random.uniform(800, 1200)
    time.sleep(latency / 1000)
    
    return jsonify({
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "nvidia/OpenMath-Nemotron-7B",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(prompt.split()) + len(response_text.split())
        }
    })

if __name__ == '__main__':
    print("Mock Student Server Starting...")
    print("Model: nvidia/OpenMath-Nemotron-7B (DEMO MODE)")
    print("Port: 8009")
    print("Simulated latency: 800-1200ms")
    app.run(host='0.0.0.0', port=8009, debug=False)
