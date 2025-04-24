from google import genai
from google.genai import types
import time

file = open("GOOGLE_API_KEY_FREE.txt", "r")
api_key = file.read()
client = genai.Client(api_key=api_key)
myfile = client.files.upload(file="screenshot.png")
orientation = "right"

start = time.time()
response = client.models.generate_content(
    model='gemini-2.5-pro-preview-03-25',
    contents=[
        myfile,
        """
Analyze the provided image, treating it as a grid.
1. Assume that the start position (where the excavator is centered) is (58, 46) (y,x) and the target position (marked in purple (136, 0, 255)) is (48, 38) (y,x) (use top-down coordinates (X-axis: left to right, Y-axis: top to bottom). The dimension of the image is 64x64 (without the white border) pixels. The initial orientation is right.
2. Assume that any black areas in the image represent obstacles that must be avoided when moving from the start to the target.
3. Your task is to find the shortest path on the grid from the start coordinate to the target coordinate, moving only horizontally or vertically (no diagonal movement), and staying out of the black (0,0,0) obstacle areas.
4. Generate and execute Python code using a suitable grid-based pathfinding algorithm (such as A* or Breadth-First Search) to calculate this shortest path.
5. After executing the code, output the calculated shortest path as an ordered list of (y, x) coordinates, starting from the start coordinate and ending at the target coordinate.
6. Assuming that the excavator can move forward and backward and can rotate 90 degrees to change direction, provide the sequence of moves (forward/backward and rotation) needed to follow the calculated path. When moving forward or backward consider that you actually move by 6 pixel (and not 1). The moves should be encoded in the following format: '-1': DO_NOTHING, '0': FORWARD, '1': BACKWARD, '2': CLOCK, '3': ANTICLOCK.
7. The output should be a JSON object with two keys: "path" and "moves". The value of "path" should be the list of (y, x) coordinates, and the value of "moves" should be the list of moves in the specified format.
"""
    ],
    config=types.GenerateContentConfig(
        tools=[types.Tool(
            code_execution=types.ToolCodeExecution
        )]
    )
)
end = time.time()
print("Time taken:", end - start)


print(response.text)
#print(response.executable_code)
print(response.code_execution_result)