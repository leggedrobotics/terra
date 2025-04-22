from openai import OpenAI

file = open("OPENAI_API_KEY.txt", "r")
api_key = file.read()

client = OpenAI(api_key=api_key)

def get_weather(location):
    # Mock implementation of the function
    return {"temperature": "20°C", "condition": "Sunny"}


tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            }
        },
        "required": [
            "location"
        ],
        "additionalProperties": False
    }
}]

# response = client.responses.create(
#     model="gpt-4o",
#     input=[{"role": "system", "content": "What is the weather like in Paris today?"}],
#     tools=tools
# )

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What is the weather like in Paris today?"}
    ],
    functions=tools,
    function_call={"name": "get_weather"},
)

# response2 = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "What is the weather like in Paris today?"}
#     ],
#     tools=tools,
#     function_call={"name": "get_weather"},
# )
print(response.choices[0].message)
# print(response2.choices[0].message)