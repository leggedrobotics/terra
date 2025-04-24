from google import genai
from google.genai import types
import time

file = open("GOOGLE_API_KEY_FREE.txt", "r")
api_key = file.read()
client = genai.Client(api_key=api_key)

start = time.time()
response = client.models.generate_content(
    model='gemini-2.5-pro-preview-03-25',
    contents="Do you like pizza?",
)
end = time.time()
print("Time taken:", end - start)


print(response.text)
#print(response.executable_code)
print(response.code_execution_result)