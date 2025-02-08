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


class RitsLlamaFCHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI
        # self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # def prepare_prompt(self, messages, tools, shorten=False):
        # roles = {"system": "system", "user": "user", "tool": "ipython", "assistant": "assistant"}
        # prompt = '<|begin_of_text|>'
        # for i, message in enumerate(messages):
            # role = message['role']
            # if role == "system":
                # content = process_system_message(message['content'])
                # # content = process_system_message("")
            # elif role == "assistant":
                # content = message['content'] + '\n' + json.dumps(message['tool_calls'][0]['function'])
            # elif role == 'tool':
                # content = message['content'] if not shorten else message['content'][:1024] + '...'
            # elif role == 'user':
                # # content = ""
                # # if conversation_history[i-1]['role'] == 'system':
                    # # content += conversation_history[i-1]['content'] + '\n'
                # # content += process_user_message_with_functions(message['content'], tools)
                # content = process_user_message_with_functions(message['content'], tools)
            # else:
                # raise NotImplementedError

            # prompt += f"<|start_header_id|>{roles[role]}<|end_header_id|>\n{content}<|eot_id|>"
        # prompt += f"<|start_header_id|>{roles['assistant']}<|end_header_id|>"
        # return prompt

    @retry(wait=wait_random_exponential(min=0.5, max=5), stop=stop_after_attempt(5))
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        # prompt = self.prepare_prompt(kwargs['messages'], kwargs['tools'])
        prompt = self._format_prompt(kwargs['messages'], kwargs['tools'])
        model_id = kwargs['model']
        llm_api_key = os.environ['RITS_API_KEY']

        model_name_full = model_id.replace('-FC', '')
        model_name = model_name_full.split('/')[1].lower().replace('.', '-')
        endpoint = f"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{model_name}/v1/completions"
        request_body = {
            'model': model_name_full,
            'prompt': prompt,
            'max_tokens': 2048,
        }
        for k, v in kwargs.items():
            request_body[k] = v
        
        headers = {'Content-Type': 'application/json', 'RITS_API_KEY': f'{llm_api_key}'}
        api_json = requests.post(endpoint, json=request_body, headers=headers).json()
        end_time = time.time()

        if api_json['object'] == 'error':
            # service_err = True
            print(api_json)
            # print(json_data['extensions']['state']['message'])
            raise RuntimeError

        predictions = api_json['choices'][0]['text']
        gen_token_len = api_json['usage']['completion_tokens']
        input_token_len = api_json['usage']['prompt_tokens']
        api_response = {
            'usage': {
                'prompt_tokens': input_token_len,
                'completion_tokens': gen_token_len,
            },
            'choices': [
                {
                    'message': predictions,
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
        # insert in-context examples here
        if 'examples' in test_entry and len(test_entry['examples']) > 0:
            examples_prompt = 'The following are example usage for the given functions:\n\n'
            for ex in test_entry['examples']:
                examples_prompt += f"Example question:{ex['question']}\nGround truth answer:{json.dumps(ex['answer'])}\n\n"
            examples_prompt += 'Now you should answer the following user question:'
            test_entry['question'][0][0]['content'] = examples_prompt + test_entry['question'][0][0]['content']

        inference_data["message"] = copy.deepcopy(test_entry['question'][0])
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: any, inference_data: dict = None) -> dict:
        # try:
            # model_responses = [
                # {func_call['function']['name']: func_call['function']['arguments']}
                # for func_call in api_response['choices'][0]['message']['tool_calls']
            # ]
            # tool_call_ids = [
                # func_call['id'] for func_call in api_response['choices'][0]['message']['tool_calls']
            # ]
        # except:
        model_responses = []
        tool_call_ids = []
        result = api_response['choices'][0]['message']
        # print(repr(result))
        try:
            func_param_map = {}
            for func in inference_data['tools']:
                func_name = func['function']['name']
                for k, v in func['function']['parameters']['properties'].items():
                    pname = f'{func_name}_{k}'
                    func_param_map[pname] = v['type']

            result = result.replace("<|python_tag|>", "")
            # Llama sometimes separates the function calls with `;` and sometimes with `,`
            if ";" in result:
                function_calls = result.split(";")
                function_calls = [json.loads(func_call) for func_call in function_calls]
            else:
                # function_calls = eval(result)
                function_calls = json.loads(result)
                if type(function_calls) == dict:
                    function_calls = [function_calls]
                else:
                    function_calls = [json.loads(func_call) if isinstance(func_call, str) else func_call for func_call in function_calls]

            for i, func_call in enumerate(function_calls):
                func_name = func_call['name']
                params = {}
                for k, v in func_call['parameters'].items():
                    if f'{func_name}_{k}' in func_param_map and func_param_map[f'{func_name}_{k}'] != 'string' and isinstance(v, str):
                        if func_param_map[f'{func_name}_{k}'] == 'boolean':
                            v = v.replace('true', 'True').replace('false', 'False')
                        params[k] = eval(v)
                    else:
                        params[k] = v
                model_responses.append({func_name: params})
                tool_call_ids.append(i)
        except Exception as e:
            print(e, f'model response = {result}')
            model_responses.append(result)


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
        # inference_data["message"].extend(first_turn_message)
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

    def _format_prompt(self, messages, function):
        """
        "bos_token": "<|begin_of_text|>",
        "chat_template":
        {{- bos_token }}
        {%- if custom_tools is defined %}
            {%- set tools = custom_tools %}
        {%- endif %}
        {%- if not tools_in_user_message is defined %}
            {%- set tools_in_user_message = true %}
        {%- endif %}
        {%- if not date_string is defined %}
            {%- set date_string = "26 Jul 2024" %}
        {%- endif %}
        {%- if not tools is defined %}
            {%- set tools = none %}
        {%- endif %}

        {#- This block extracts the system message, so we can slot it into the right place. #}
        {%- if messages[0]['role'] == 'system' %}
            {%- set system_message = messages[0]['content']|trim %}
            {%- set messages = messages[1:] %}
        {%- else %}
            {%- set system_message = "" %}
        {%- endif %}

        {#- System message + builtin tools #}
        {{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
        {%- if builtin_tools is defined or tools is not none %}
            {{- "Environment: ipython\n" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
        {%- endif %}
        {{- "Cutting Knowledge Date: December 2023\n" }}
        {{- "Today Date: " + date_string + "\n\n" }}
        {%- if tools is not none and not tools_in_user_message %}
            {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
            {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
            {{- "Do not use variables.\n\n" }}
            {%- for t in tools %}
                {{- t | tojson(indent=4) }}
                {{- "\n\n" }}
            {%- endfor %}
        {%- endif %}
        {{- system_message }}
        {{- "<|eot_id|>" }}

        {#- Custom tools are passed in a user message with some extra guidance #}
        {%- if tools_in_user_message and not tools is none %}
            {#- Extract the first user message so we can plug it in here #}
            {%- if messages | length != 0 %}
                {%- set first_user_message = messages[0]['content']|trim %}
                {%- set messages = messages[1:] %}
            {%- else %}
                {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
        {%- endif %}
            {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
            {{- "Given the following functions, please respond with a JSON for a function call " }}
            {{- "with its proper arguments that best answers the given prompt.\n\n" }}
            {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
            {{- "Do not use variables.\n\n" }}
            {%- for t in tools %}
                {{- t | tojson(indent=4) }}
                {{- "\n\n" }}
            {%- endfor %}
            {{- first_user_message + "<|eot_id|>"}}
        {%- endif %}

        {%- for message in messages %}
            {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
                {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
            {%- elif 'tool_calls' in message %}
                {%- if not message.tool_calls|length == 1 %}
                    {{- raise_exception("This model only supports single tool-calls at once!") }}
                {%- endif %}
                {%- set tool_call = message.tool_calls[0].function %}
                {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
                    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
                    {{- "<|python_tag|>" + tool_call.name + ".call(" }}
                    {%- for arg_name, arg_val in tool_call.arguments | items %}
                        {{- arg_name + '="' + arg_val + '"' }}
                        {%- if not loop.last %}
                            {{- ", " }}
                        {%- endif %}
                        {%- endfor %}
                    {{- ")" }}
                {%- else  %}
                    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
                    {{- '{"name": "' + tool_call.name + '", ' }}
                    {{- '"parameters": ' }}
                    {{- tool_call.arguments | tojson }}
                    {{- "}" }}
                {%- endif %}
                {%- if builtin_tools is defined %}
                    {#- This means we're in ipython mode #}
                    {{- "<|eom_id|>" }}
                    {#- This means we're in ipython mode #}
                    {{- "<|eom_id|>" }}
                    {{- "<|eom_id|>" }}
                {%- else %}
                    {{- "<|eot_id|>" }}
                {%- endif %}
            {%- elif message.role == "tool" or message.role == "ipython" %}
                {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
                {%- if message.content is mapping or message.content is iterable %}
                    {{- message.content | tojson }}
                {%- else %}
                    {{- message.content }}
                {%- endif %}
                {{- "<|eot_id|>" }}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
        {%- endif %}
        """
        formatted_prompt = "<|begin_of_text|>"

        system_message = ""
        remaining_messages = messages
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"].strip()
            remaining_messages = messages[1:]

        formatted_prompt += "<|start_header_id|>system<|end_header_id|>\n\n"
        formatted_prompt += "Environment: ipython\n"
        formatted_prompt += "Cutting Knowledge Date: December 2023\n"
        formatted_prompt += "Today Date: 26 Jul 2024\n\n"
        formatted_prompt += system_message + "<|eot_id|>"

        # Llama pass in custom tools in first user message
        is_first_user_message = True
        for message in remaining_messages:
            if message["role"] == "user" and is_first_user_message:
                is_first_user_message = False
                formatted_prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
                formatted_prompt += "Given the following functions, please respond with a JSON for a function call "
                formatted_prompt += (
                    "with its proper arguments that best answers the given prompt.\n\n"
                )
                formatted_prompt += 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.'
                formatted_prompt += "Do not use variables.\n\n"
                for func in function:
                    formatted_prompt += json.dumps(func, indent=4) + "\n\n"
                formatted_prompt += f"{message['content'].strip()}<|eot_id|>"

            elif message["role"] == "tool":
                formatted_prompt += "<|start_header_id|>ipython<|end_header_id|>\n\n"
                if isinstance(message["content"], (dict, list)):
                    formatted_prompt += json.dumps(message["content"])
                else:
                    formatted_prompt += message["content"]
                formatted_prompt += "<|eot_id|>"

            else:
                formatted_prompt += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>"

        formatted_prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return formatted_prompt

    def decode_ast(self, result, language="Python"):
        result = result.replace("<|python_tag|>", "")
        # Llama sometimes separates the function calls with `;` and sometimes with `,`
        if ";" in result:
            """
            "<|python_tag|>{\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"10\", \"k\": \"3\", \"p\": \"0\"}}; {\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"15\", \"k\": \"5\", \"p\": \"0\"}}; {\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"20\", \"k\": \"7\", \"p\": \"0\"}}"
            """
            function_calls = result.split(";")
            function_calls = [json.loads(func_call) for func_call in function_calls]
        else:
            """
            "[\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"20\", \"k\": \"5\"}},\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"12\", \"k\": \"5\"}},\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"10\", \"k\": \"3\"}}\n]"
            """
            function_calls = eval(result)
            if type(function_calls) == dict:
                function_calls = [function_calls]

        decoded_output = []
        for func_call in function_calls:
            name = func_call["name"]
            params = func_call["parameters"]
            decoded_output.append({name: params})

        return decoded_output

    def decode_execute(self, result):
        return convert_to_function_call(result)
        result = result.replace("<|python_tag|>", "")
        # Llama sometimes separates the function calls with `;` and sometimes with `,`
        if ";" in result:
            function_calls = result.split(";")
            function_calls = [json.loads(func_call) for func_call in function_calls]
        else:
            function_calls = eval(result)
            if type(function_calls) == dict:
                function_calls = [function_calls]

        execution_list = []
        for func_call in function_calls:
            name = func_call["name"]
            params = func_call["parameters"]
            # params_true = []
            # for k, v in params.items():
                # try:
                    # val = eval(v)
                    # params_true.append(f'{k}={val}')
                # except:
                    # params_true.append(f'{k}={repr(v)}')

            execution_list.append(
                f"{name}({','.join([f'{k}={repr(v)}' for k,v in params.items()])})"
                # f"{name}({','.join([p for p in params_true])})"
            )

        return execution_list
