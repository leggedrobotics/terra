from google.genai import types
import time

# Define a function that the model can call to control smart lights
set_light_values_declaration = {
    "name": "set_light_values",
    "description": "Sets the brightness and color temperature of a light.",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "integer",
                "description": "Light level from 0 to 100. Zero is off and 100 is full brightness",
            },
            "color_temp": {
                "type": "string",
                "enum": ["daylight", "cool", "warm"],
                "description": "Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.",
            },
        },
        "required": ["brightness", "color_temp"],
    },
}

# This is the actual function that would be called based on the model's suggestion
def set_light_values(brightness: int, color_temp: str) -> dict[str, int | str]:
    """Set the brightness and color temperature of a room light. (mock API).

    Args:
        brightness: Light level from 0 to 100. Zero is off and 100 is full brightness
        color_temp: Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.

    Returns:
        A dictionary containing the set brightness and color temperature.
    """
    return {"brightness": brightness, "colorTemperature": color_temp}

from google import genai

# Generation Config with Function Declaration
tools = types.Tool(function_declarations=[set_light_values_declaration])
config = types.GenerateContentConfig(tools=[tools])

# Configure the client
file = open("GOOGLE_API_KEY_FREE.txt", "r")
api_key = file.read()
client = genai.Client(api_key=api_key)

# Define user prompt
contents = [
    types.Content(
        role="user", parts=[types.Part(text="Turn the lights down to a romantic level")]
    )
]

model_0 = "gemini-2.0-flash"
#model_1 = "gemini-2.5-pro-exp-03-25"
model_1 = "gemini-2.0-flash-lite"
model_2 = "gemini-2.5-pro-preview-03-25"

list_0 = []
for i in range(10):
    response_0 = client.models.generate_content(
        model=model_0, config=config, contents=contents
    )
    list_0.append(response_0.candidates[0].content.parts[0].function_call)
    time.sleep(0.1)


print(f"Response for {model_0}:")
print(list_0, sep="\n")


list_1 = []
for i in range(10):
    response_1 = client.models.generate_content(
        model=model_1, config=config, contents=contents
    )
    list_1.append(response_1.candidates[0].content.parts[0].function_call)
    time.sleep(0.1)

print("_________________________________________________________________________________________________________________________________________")
print(f"Response for {model_1}:")
print(list_1, sep="\n")

list_2 = []
for i in range(10):
    response_2 = client.models.generate_content(
        model=model_2, config=config, contents=contents
    )
    list_2.append(response_2.candidates[0].content.parts[0].function_call)
    time.sleep(0.1)

print("_________________________________________________________________________________________________________________________________________")
print(f"Response for {model_2}:")
print(list_2, sep="\n")