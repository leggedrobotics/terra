from google import genai
from google.genai import types

import base64 
import json 
import cv2 
import re 
import os 
import logging
from typing import Optional, Dict, Any, List, Tuple
# Set up logging

import asyncio
logger = logging.getLogger("AutonomousExcavatorADK.llms")


class LLM_query: 
    def __init__(self, model_name=None, model=None, system_message=None, env=None, runner=None, user_id=None, session_id=None): 
        """
        Initialize the Agent with the specified model and environment.
        
        Args:
            model_name: The specific model name to use
            model: The model provider key ('gpt4', 'gpt4o', 'claude', 'gemini')
            system_message: The system prompt to use
            env: The Gymnasium environment
        """
        self.model_key = model 
        logger.info(f'Model Key: {self.model_key}') 

        self.model_name = model_name 
        logger.info(f'Model Name: {self.model_name}') 

        self.messages = [] 
        self.system_message = system_message 
        self.env = env 
        #self.action_space = self.env.action_space.n
        self.action_space = self.env.actions_size  # Updated line
        self.reset_count = 0 
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id

    def encode_image(self, cv_image):
        _, buffer = cv2.imencode(".jpg", cv_image)
        return base64.b64encode(buffer).decode("utf-8")
        
    async def query_LLM(self):
        #TODO: Optimize the lenght of the messages to be sent to the model

        response_text = ""

        #for i, message in enumerate(self.messages):
        message = self.messages[-1]
        async for event in self.runner.run_async(user_id=self.user_id, session_id=self.session_id, new_message=message):
            print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}")
            if event.is_final_response():
                if event.content and event.content.parts:
                    response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    response_text = f"Agent escalated: {event.error_message or 'No specific message'}"
                break
            else:
                if event.content and event.content.parts and event.content.parts[0].text:
                    response_text += event.content.parts[0].text


        self.response = response_text
        self.reset_count = 0

        # return the output of the model
        return self.response
    
    def reset_model(self):

        if self.reset_count >= 3:
            return
        
        self.messages = []
        
        self.reset_count += 1

        print('Model is re-initiated...')

    def clean_model_output(self, output):
        """
        Clean the model output to ensure it's valid JSON.
        
        Args:
            output: The raw output from the model
            
        Returns:
            Cleaned output string
        """
        if not output:
            logger.warning("Received empty output from model")
            return ""
            
        # Remove any unescaped newline characters within the JSON string values
        cleaned_output = re.sub(r'(?<!\\)\n', ' ', output)
        
        # Replace curly quotes with straight quotes if necessary
        cleaned_output = cleaned_output.replace('"', '"').replace('"', '"')
        
        return cleaned_output

    def clean_response(self, response, path):
        """
        Extract and clean the response from the model.
        
        Args:
            response: The raw response object from the model
            path: Path to save the response for debugging
            
        Returns:
            Parsed JSON response or None if parsing fails
        """
        try:
            response_text = response
            
            if response_text is None:
                logger.warning("Received None response, attempting to get response again")
                response_text = self.get_response()

            response_text = self.clean_model_output(response_text)

            # Save the raw response for debugging
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path+'all_responses.txt', "a") as file:
                file.write(str(response_text) + '\n\n')

            # This regular expression finds the first { to the last }
            pattern = r'\{.*\}'
            # Search for the pattern
            match = re.search(pattern, response_text, flags=re.DOTALL)
            # Return the matched group which should be a valid JSON string
            if match:
                response_text = match.group(0)

            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                logger.info("Reprompting the model for valid JSON")

                # Create error message to reprompt the model
                error_message = 'Your output should be in a valid JSON format with a comma after every key-value pair except the last one. Please provide a response with the format: {"reasoning": "your reasoning here", "action": numeric_action_value}'
                
                # Add the error message to the context
                self.add_user_message(user_msg=error_message)

                logger.info('Generating new response...')
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    try:
                        response = asyncio.run(self.query_LLM())
                        logger.info('Proper response was generated')
                        return self.clean_response(response, path)  # Recursively process the new response
                    except Exception as e:
                        logger.error(f"Error generating response (attempt {retry_count+1}): {str(e)}")
                        self.reset_model()
                        retry_count += 1
                
                logger.error("Failed to get valid JSON after multiple attempts")
                return None
                
        except Exception as e:
            logger.error(f"Error cleaning response: {str(e)}")
            return None
    
    def check_action(self, response_text):
        """
        Validate the action from the model response.
        
        Args:
            response_text: The parsed JSON response from the model
            
        Returns:
            Valid action integer or None if invalid
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            # Check if the response is a dictionary
            if isinstance(response_text, dict):
                # Check if the key action exists 
                if "action" in response_text:
                    try:
                        action = int(response_text["action"])
                        # Check if the action is valid for the environment
                        #print("action: ", action, self.action_space)
                        if -1 <= action < self.action_space:
                            return action
                        else:
                            logger.warning(f"Invalid action value: {action}. Must be between -1 and {self.env.actions_size-1}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error converting action to integer: {str(e)}")
                else:
                    logger.warning("Response missing 'action' key")
            else:
                logger.warning(f"Response is not a dictionary: {type(response_text)}")
            
            # If we get here, the action was invalid
            error_message = f'Your action value is invalid. Please provide a valid action between -1 and {self.env.actions_size-1}.'
            self.add_user_message(user_msg=error_message)
            
            try:
                response = asyncio.run(self.query_LLM())
                response_text = self.clean_response(response, "./")
                retry_count += 1
            except Exception as e:
                logger.error(f"Error getting new response: {str(e)}")
                retry_count += 1
        
        # If we've exhausted retries, return a default action (0)
        logger.error("Failed to get valid action after multiple attempts, using default action -1")
        return -1

    def get_response(self):
        # Check to see if you get a response from the model
        try: 
            response = asyncio.run(self.query_LLM())

        # If there is an error with generating a response (internal error)
        # Reset the model and try again
        except:
            print('\n\nReceived Error when generating response reseting model\n\n')
            
            # Reset model
            self.reset_model()

            while True:

                # See if you get the correct output
                try:
                    response = asyncio.run(self.query_LLM())
                    print('\n\nReceived correct output continuing experiment.')
                    break
                # If it doesn't then reset the model
                except:
                    # Create error message to reprompt the model
                    error_message = 'Please provide a proper output'
                    
                    # Add the error message to the context
                    self.add_user_message(user_msg=error_message)

                    print('Re-initiating model...')
                    self.reset_model() 

                    # This means that more than likely you ran out of credits so break the code to not spend money
                    if self.reset_count >= 3:
                        return None
                    
        if response == 'idchoicescreatedmodelobjectsystem_fingerprintusage':
            response = self.get_response()
        
        return response


    def generate_response(self, path) -> str:   
        response = self.get_response()
        # Check if it is just reasoning or actual action output
        self.path = path

        response_text = self.clean_response(response, path)

        action_output = self.check_action(response_text)

        return action_output, response_text

    def add_user_message(self, frame=None, user_msg=None, local_map=None, traversability_map=None):
        
        if self.model_key == 'gemini' or self.model_key == 'claude' or self.model_key == 'gpt':
            if frame is not None and user_msg is not None and traversability_map is not None and local_map is not None:
                image_data = self.encode_image(frame)
                image_data_traversability = self.encode_image(traversability_map)
                image_data_local_map = self.encode_image(local_map)

                # Create the list of Part objects
                parts = [
                    types.Part.from_bytes(data=image_data, mime_type="image/jpeg"), # Create image Part from bytes/base64
                    types.Part.from_bytes(data=image_data_local_map, mime_type="image/jpeg"),
                    types.Part.from_bytes(data=image_data_traversability, mime_type="image/jpeg"),
                    types.Part.from_text(text=user_msg) # Create text Part from string
                ]

                # Create a Content object with the role and the list of parts
                user_content = types.Content(role="user", parts=parts)

                # Append the Content object to your messages list
                self.messages.append(user_content)

            elif frame is not None and user_msg is not None and traversability_map is None:
                image_data = self.encode_image(frame)

                # Create the list of Part objects
                parts = [
                    types.Part.from_bytes(data=image_data, mime_type="image/jpeg"), # Create image Part from bytes/base64
                    types.Part.from_text(text=user_msg) # Create text Part from string
                ]

                # Create a Content object with the role and the list of parts
                user_content = types.Content(role="user", parts=parts)

                # Append the Content object to your messages list
                self.messages.append(user_content)

            elif frame is not None and user_msg is None:
                image_data = self.encode_image(frame)
                
                # Create the list of Part objects
                parts = [
                    types.Part.from_bytes(data=image_data, mime_type="image/jpeg"), # Create image Part from bytes/base64
                ]

                # Create a Content object with the role and the list of parts
                user_content = types.Content(role="user", parts=parts)

                # Append the Content object to your messages list
                self.messages.append(user_content)

            elif frame is None and user_msg is not None:
                self.messages.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": types.Part.from_text(text=user_msg)
                            }
                        ]
                    }
                )
            else:
                pass

            # Ensure self.messages only contains types.Content objects
            self.messages = [msg for msg in self.messages if isinstance(msg, types.Content)]

    def add_assistant_message(self, demo_str=None):

        if self.model_key =='gemini' or self.model_key == 'claude' or self.model_key == 'gpt':
            if demo_str is not None:
                self.messages.append(
                    {
                        "role": "model",
                        "parts": types.Part.from_text(text=demo_str),
                    }
                )
                demo_str = None
                return

            if self.response is not None:
                #assistant_msg = self.response.text
                assistant_msg = self.response
                self.messages.append(
                    {
                        "role": "model",
                        "parts": types.Part.from_text(text=assistant_msg),
                    }
                )

        else:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ' '},
                    ]
                }
            )

    def delete_messages(self):
        print('Deleting Set of Messages...')

        if self.model_key == 'gemini' or self.model_key == 'claude' or self.model_key == 'gpt':
                # Check if the first message is a system message (using .role attribute)
                if self.messages and isinstance(self.messages[0], types.Content) and self.messages[0].role == 'system':
                    print("Removing oldest user/model pair after system message.")
                    # Delete the oldest user message (at index 1 after system)
                    # and the oldest model message (at index 2 after system)
                    # Remove higher index first
                    if len(self.messages) > 2: # Ensure there are at least 3 messages (system, user, model)
                        self.messages.pop(2) # Remove model
                        self.messages.pop(1) # Remove user
                    elif len(self.messages) > 1: # If only system and user
                         self.messages.pop(1) # Remove user
                    # If only system message, do nothing as we need a pair to remove

                else:
                    print("Removing oldest user/model pair.")
                    # Delete the oldest user message (at index 0)
                    # and the oldest model message (at index 1)
                    # Remove higher index first
                    if len(self.messages) > 1: # Ensure there is at least a user and model message
                        self.messages.pop(1) # Remove model
                        self.messages.pop(0) # Remove user
                    elif len(self.messages) > 0: # If only user message
                         self.messages.pop(0) # Remove user
                    # If list is empty, do nothing

        else:
            print("Message history length below threshold. No messages deleted.")