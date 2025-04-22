
from terra.viz.llms_utils import extract_positions

def get_excavator_position(current_state):
    """Gets the current (y, x) position of the excavator base."""
    # This function needs access to the game state.
    # Ensure 'extract_positions' is available in this scope.
    # You might need to import it or make this function a method
    # of a class that has access to state processing functions.
    try:
        # Assuming extract_positions is available in the scope
        start_pos, _ = extract_positions(current_state)
        # Convert JAX array elements to standard Python integers for JSON serialization
        return {"position_y": int(start_pos[0]), "position_x": int(start_pos[1])}
    except NameError:
        print("Error: extract_positions function not found.")
        return {"error": "extract_positions function not available"}
    except Exception as e:
        print(f"Error getting excavator position: {e}")
        return {"error": str(e)}


# --- Tool Schema Definition ---
# Define this globally or within the class __init__

excavator_tools = [{
        "type": "function",
        "name": "get_excavator_position",
        "description": "Get the current Y and X coordinates of the excavator's base.",
        "parameters": {
                "type": "object",
                "properties": {
                    "current_state": {
                        "type": "object",
                        "description": "The current state of the excavator game.",
                        # Define the expected structure of the current_state object here
                    }
                }, 
                "required": [
                    "current_state"
                ],
                "additionalProperties": False
        }
}]