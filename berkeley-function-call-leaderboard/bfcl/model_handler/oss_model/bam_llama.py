import json
import os
import time
from datetime import date
import copy
import requests

from bfcl.model_handler.base_handler import BaseHandler
from bfcl.model_handler.constant import GORILLA_TO_OPENAPI
from bfcl.model_handler.model_style import ModelStyle
from bfcl.model_handler.utils import (
    combine_consecutive_user_prompts,
    convert_system_prompt_into_user_prompt,
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
)
# from openai import OpenAI, RateLimitError
from tenacity import retry, wait_random_exponential, stop_after_attempt

def process_system_message(message):
    message = '\n\nCutting Knowledge Date: December 2023\n' + f'Today Date: {date.today().strftime("%d %b %Y")}\n\n' + 'You are a helpful assistant with tool calling capabilities.\n\n' + message
    #When you receive a tool call response, use the output to format an answer to the orginal user question.'
    return message


def process_user_message_with_functions(message, functions, max_examples=50):
    # prompt = '\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\n\n'
    prompt = "Given the following functions, please respond with a JSON for a function call "
    prompt += "with its proper arguments that best answers the given prompt.\n\n"
    prompt += 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.'
    prompt += "Do not use variables.\n\n"
    # prompt = '\nGiven the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\nRespond in the format: your reasoning thoughts\n{"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables in the JSON output.\n\n'
    examples = []
    for function in functions:
        function_new = copy.deepcopy(function)
        if 'examples' in function_new['function'] and len(function_new['function']['examples']) > 0:
            if len(examples) < max_examples:
                examples.append(function_new['function']['examples'])
            del function_new['function']['examples']
        prompt += json.dumps(function_new) + '\n'
        # prompt += json.dumps(function) + '\n'

    if len(examples) > 0:
        prompt += '\nThe following are examples that demonstrate how to use these functions: \n'
        for example in examples:
            prompt += example + '\n'

    # prompt += '\n\nRespond in the format: your reasoning thoughts\n{"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables in the JSON output.\n\nTask:'
    return prompt + message

class BamLlamaHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI
        # self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def decode_ast(self, result, language="Python"):
        if "FC" in self.model_name or self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        else:
            return default_decode_ast_prompting(result, language)

    def decode_execute(self, result):
        if "FC" in self.model_name or self.is_fc_model:
            return convert_to_function_call(result)
        else:
            return default_decode_execute_prompting(result)

    def prepare_prompt(self, messages, tools, shorten=False):
        roles = {"system": "system", "user": "user", "tool": "ipython", "assistant": "assistant"}
        prompt = '<|begin_of_text|>'
        for i, message in enumerate(messages):
            role = message['role']
            if role == "system":
                content = process_system_message(message['content'])
                # content = process_system_message("")
            elif role == "assistant":
                content = message['content'] + '\n' + json.dumps(message['tool_calls'][0]['function'])
            elif role == 'tool':
                content = message['content'] if not shorten else message['content'][:1024] + '...'
            elif role == 'user':
                # content = ""
                # if conversation_history[i-1]['role'] == 'system':
                    # content += conversation_history[i-1]['content'] + '\n'
                # content += process_user_message_with_functions(message['content'], tools)
                content = process_user_message_with_functions(message['content'], tools)
            else:
                raise NotImplementedError

            prompt += f"<|start_header_id|>{roles[role]}<|end_header_id|>\n{content}<|eot_id|>"
        prompt += f"<|start_header_id|>{roles['assistant']}<|end_header_id|>"
        return prompt

    @retry(wait=wait_random_exponential(min=0.5, max=5), stop=stop_after_attempt(5))
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        prompt = self.prepare_prompt(kwargs['messages'], kwargs['tools'])
        model_id = kwargs['model'].lower()
        llm_api_key = os.environ['BAM_API_KEY']

        # api_response = self.client.chat.completions.create(**kwargs)
        endpoint = "https://bam-api.res.ibm.com/v2/text/generation?version=2024-03-19"
        request_body = {
            'model_id': model_id,
            'input': prompt,
            'parameters': {
                'decoding_method': 'sample',
                'temperature': kwargs['temperature'],
                'max_new_tokens': 1024,
                'include_stop_sequence': False,
                'stop_sequences': ['<|eot_id|>', '<|end_of_text|>', '<|eom_id|>'], 
            }
        }
        # for k, v in kwargs.items():
            # request_body['parameters'][k] = v
        
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {llm_api_key}'}
        json_data = requests.post(endpoint, json=request_body, headers=headers).json()
        end_time = time.time()

        if 'status_code' in json_data and 'error' in json_data:
            # service_err = True
            print(json_data)
            # print(json_data['extensions']['state']['message'])
            raise RuntimeError

        predictions = json_data['results'][0]['generated_text']
        gen_token_len = json_data['results'][0]['generated_token_count']
        input_token_len = json_data['results'][0]['input_token_count']

        # react format prediction
        json_idx = predictions.find('{"name":')
        if json_idx == -1:
            json_idx = predictions.find('{\n"name":')
        if json_idx == -1:
            json_idx = predictions.find('{\n')
        if json_idx == -1:
            json_idx = predictions.find('{')
        # print('pred', predictions, json_idx)
        json_end_idx = predictions.rfind('}')
        json_end_idx = json_end_idx + 1 if json_end_idx != -1 else -1

        thoughts = predictions[:json_idx].strip()
        output = predictions[json_idx:json_end_idx].strip()
        # output = json.loads(predictions.strip())
        try:
            output = json.loads(output)
            action = output['name']
            action_input = {}
            if 'parameters' in output:
                action_input = output['parameters']
            if 'arguments' in output:
                action_input = output['arguments']
            if 'properties' in output:
                action_input = output['properties']
            action_input = {k: v for k, v in action_input.items() if len(k) > 0}
        except Exception as e:
            print(e, predictions)

        message = {
            "role": "assistant",
            "content": thoughts,
            "tool_calls": [
                {
                    "function": {
                        "name": action,
                        "arguments": action_input
                    },
                    "id": 0
                }
            ]
        }
        api_response = {
            'usage': {
                'prompt_tokens': input_token_len,
                'completion_tokens': gen_token_len,
            },
            'choices': [
                {
                    'message': message,
                },
            ]
        }


        return api_response, end_time - start_time

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data["message"]
        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}

        if len(tools) > 0:
            return self.generate_with_backoff(
                messages=message,
                model=self.model_name.replace("-FC", ""),
                temperature=self.temperature,
                tools=tools,
            )
        else:
            return self.generate_with_backoff(
                messages=message,
                model=self.model_name.replace("-FC", ""),
                temperature=self.temperature,
            )

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: any) -> dict:
        try:
            model_responses = [
                {func_call['function']['name']: func_call['function']['arguments']}
                for func_call in api_response['choices'][0]['message']['tool_calls']
            ]
            tool_call_ids = [
                func_call['id'] for func_call in api_response['choices'][0]['message']['tool_calls']
            ]
        except:
            model_responses = api_response['choices'][0]['message']['content']
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response['choices'][0]['message']

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "tool_call_ids": tool_call_ids,
            "input_token": api_response['usage']['prompt_tokens'],
            "output_token": api_response['usage']['completion_tokens'],
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # Add the execution results to the current round result, one at a time
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)

        return inference_data

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}

        # These two models have temperature fixed to 1
        # Beta limitation: https://platform.openai.com/docs/guides/reasoning/beta-limitations
        if "o1-preview" in self.model_name or "o1-mini" in self.model_name:
            return self.generate_with_backoff(
                messages=inference_data["message"],
                model=self.model_name,
                temperature=1,
            )
        else:
            return self.generate_with_backoff(
                messages=inference_data["message"],
                model=self.model_name,
                temperature=self.temperature,
            )

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )
        # Special handling for o1-preview and o1-mini as they don't support system prompts yet
        if "o1-preview" in self.model_name or "o1-mini" in self.model_name:
            for round_idx in range(len(test_entry["question"])):
                test_entry["question"][round_idx] = convert_system_prompt_into_user_prompt(
                    test_entry["question"][round_idx]
                )
                test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                    test_entry["question"][round_idx]
                )

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: any) -> dict:
        return {
            "model_responses": api_response.choices[0].message.content,
            "model_responses_message_for_chat_history": api_response.choices[0].message,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )

        return inference_data
