import base64
import cv2
import numpy as np
import jax
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import jax.numpy as jnp
from terra.viz.llms_adk import *
from terra.viz.a_star import compute_path, simplify_path
from terra.viz.llms_utils import *
import csv

from utils.models import get_model_ready
import json

def print_stats(
    stats,
):
    episode_done_once = stats["episode_done_once"]
    episode_length = stats["episode_length"]
    path_efficiency = stats["path_efficiency"]
    workspaces_efficiency = stats["workspaces_efficiency"]
    coverage = stats["coverage"]

    completion_rate = 100 * episode_done_once.sum() / len(episode_done_once)

    print("\nStats:\n")
    print(f"Completion: {completion_rate:.2f}%")
    # print(f"First episode length average: {episode_length.mean()}")
    # print(f"First episode length min: {episode_length.min()}")
    # print(f"First episode length max: {episode_length.max()}")
    print(
        f"Path efficiency: {path_efficiency['mean']:.2f} ({path_efficiency['std']:.2f})"
    )
    print(
        f"Workspaces efficiency: {workspaces_efficiency['mean']:.2f} ({workspaces_efficiency['std']:.2f})"
    )
    print(f"Coverage: {coverage['mean']:.2f} ({coverage['std']:.2f})")
def encode_image(cv_image):
    _, buffer = cv2.imencode(".jpg", cv_image)
    return base64.b64encode(buffer).decode("utf-8")

async def call_agent_async_master(query: str, image, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    #print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    #content = types.Content(role='user', parts=[types.Part(text=query)])
    text = types.Part.from_text(text=query)
    parts = [text]
    if image is not None:
        # Convert the image to a format suitable for ADK
        image_data = encode_image(image)

        content_image = types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
        parts.append(content_image)

    user_content = types.Content(role='user', parts=parts)
    
    final_response_text = "Agent did not produce a final response." # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_content):
        # You can uncomment the line below to see *all* events during execution
        print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate: # Handle potential errors/escalations
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            # Add more checks here if needed (e.g., specific error codes)
            break # Stop processing events once the final response is found

    #print(f"<<< Agent Response: {final_response_text}")
    return final_response_text


def load_neural_network(config, env):
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config, env)
    return model

def init_llms(llm_model_key, llm_model_name, USE_PATH, config, env, n_envs, APP_NAME, USER_ID, SESSION_ID):
    if llm_model_key == "gpt":
        llm_model_name_extended = "openai/{}".format(llm_model_name)
    elif llm_model_key == "claude":
        llm_model_name_extended = "anthropic/{}".format(llm_model_name)
    else:
        llm_model_name_extended =  llm_model_name
    
    print("Using model: ", llm_model_name_extended)

    description_master = "You are a master agent controlling an excavator. Observe the state. " \
    "Decide if you should delegate digging tasks to a " \
    "specialized RL agent (respond with 'delegate_to_rl') or to delegate the task to a" \
    "specialized LLM agent (respond with 'delegate_to_llm')."

    system_message_master = "You are a master agent controlling an excavator. Observe the state. " \
    "Decide if you should delegate digging tasks to a " \
    "specialized RL agent (respond with 'delegate_to_rl') or to delegate the task to a" \
    "specialized LLM agent (respond with 'delegate_to_llm')."

    description_excavator = "You are an excavator agent. You can control the excavator to dig and move."

    if USE_PATH:
        # Load the JSON configuration file
        with open("envs19.json", "r") as file:
            game_instructions = json.load(file)
    else:
        # Load the JSON configuration file
        with open("envs18.json", "r") as file:
            game_instructions = json.load(file)

    # Define the environment name for the Autonomous Excavator Game
    environment_name = "AutonomousExcavatorGame"

    # Retrieve the system message for the environment
    system_message_excavator = game_instructions.get(
        environment_name,
        "You are a game playing assistant. Provide the best action for the current game state."
    )

    if llm_model_key == "gemini":
        llm_excavator_agent = Agent(
            name="ExcavatorAgent",
            model=llm_model_name_extended,
            description=description_excavator,
            instruction=system_message_excavator,
        )

        llm_master_agent = Agent(
            name="MasterAgent",
            model=llm_model_name_extended,
            description=description_master,
            instruction=system_message_master,
            #sub_agents=[llm_excavator_agent],
        )
    else:
        llm_excavator_agent = Agent(
            name="ExcavatorAgent",
            model=LiteLlm(model=llm_model_name_extended),
            description=description_excavator,
            instruction=system_message_excavator,
        )

        llm_master_agent = Agent(
            name="MasterAgent",
            model=LiteLlm(model=llm_model_name_extended),
            description=description_master,
            instruction=system_message_master,
            #sub_agents=[llm_excavator_agent],
        )
    
    
    print("Master Agent initialized.")
    print("Excavator Agent initialized.")



    session_service = InMemorySessionService()



    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    print("Session created. App: ", APP_NAME, " User ID: ", USER_ID, " Session ID: ", SESSION_ID)
    
    runner = Runner(
        agent=llm_master_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    print(f"Runner initialized for agent {runner.agent.name}.")
    

    APP_NAME_2 = APP_NAME + "_excavator"
    SESSION_ID_2 = SESSION_ID + "_excavator"
    USER_ID_2 = USER_ID + "_excavator"     
        
    session_service_2 = InMemorySessionService()
    session_2 = session_service_2.create_session(
        app_name=APP_NAME_2,
        user_id=USER_ID_2,
        session_id=SESSION_ID_2,
    )
    runner_2 = Runner(
        agent=llm_excavator_agent,
        app_name=APP_NAME_2,
        session_service=session_service_2,
    )

    llm_query = LLM_query(
        model_name=llm_model_name_extended,
        model=llm_model_key,
        system_message=system_message_excavator,
        env=env,
        session_id=SESSION_ID_2,
        runner=runner_2,
        user_id=USER_ID_2,
    )

    prev_actions = None
    if config:
        prev_actions = jnp.zeros(
            (n_envs, config.num_prev_actions),
            dtype=jnp.int32
        )
    else:
        print("Warning: rl_config is None, prev_actions will not be initialized.")
    
    return llm_query, runner, prev_actions, system_message_master

def compute_action_list(timestep,env):
        # Compute the path
        start, target_positions = extract_positions(timestep.state)
        target = find_nearest_target(start, target_positions)
        path, path2, _ = compute_path(timestep.state, start, target)


        initial_orientation = extract_base_orientation(timestep.state)
        initial_direction = initial_orientation["direction"]

        actions = path_to_actions(path, initial_direction, 6)
        print("Action list", actions)

        if path:
            game = env.terra_env.rendering_engine
            game.path = path
        else:
            print("No path found.")

        return actions

def save_csv(output_file, action_list, cumulative_rewards):
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["actions", "cumulative_rewards"]) # Header updated
        # Iterate through actions and the calculated cumulative rewards
        for action, cum_reward in zip(action_list, cumulative_rewards):
            # Assuming action is array-like (e.g., JAX array) with one element
            action_value = action[0] if hasattr(action, '__getitem__') and len(action) > 0 else action
            # cum_reward from np.cumsum is already a scalar number
            reward_value = cum_reward
            writer.writerow([action_value, reward_value])

    print(f"Results saved to {output_file}")
