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
    ast_parse,
    decoded_output_to_execution_list,
    format_execution_results_prompting,
    func_doc_language_specific_pre_processing,
    system_prompt_pre_processing_chat_model,
)
# from openai import OpenAI, RateLimitError
from tenacity import retry, wait_random_exponential, stop_after_attempt
import numpy as np
np.random.seed(52)

class RitsLlamaHandler(BaseHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OpenAI

    @retry(wait=wait_random_exponential(min=0.5, max=5), stop=stop_after_attempt(5))
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        # prompt = self.prepare_prompt(kwargs['messages'], kwargs['tools'])
        prompt = self._format_prompt(kwargs['messages'], function=None)
        model_id = kwargs['model']
        llm_api_key = os.environ['RITS_API_KEY']
        # print(prompt)

        model_name_full = model_id
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
        api_json = requests.post(endpoint, json=request_body, headers=headers, timeout=100).json()
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

    #### Prompting methods ####
    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}

        return self.generate_with_backoff(
            messages=inference_data["message"],
            model=self.model_name,
            temperature=self.temperature,
        )


    def _parse_query_response_prompting(self, api_response: any) -> dict:
        return {
            "model_responses": api_response['choices'][0]['message'],
            "model_responses_message_for_chat_history": api_response['choices'][0]['message'],
            "input_token": api_response['usage']['prompt_tokens'],
            "output_token": api_response['usage']['completion_tokens'],
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

    def _format_prompt(self, messages, function):
        formatted_prompt = "<|begin_of_text|>"

        for message in messages:
            formatted_prompt += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>"

        formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"

        return formatted_prompt

    def decode_ast(self, result, language="Python"):
        return default_decode_ast_prompting(result, language)

    def decode_execute(self, result):
        return default_decode_execute_prompting(result)

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        # insert in-context examples here
        if 'examples' in test_entry and len(test_entry['examples']) > 0:
            examples_prompt = 'The following are example usages for the given functions:\n\n'

            # for ex in test_entry['examples']:
                # examples_prompt += f"Example question:{ex['question']}\nGround truth:{ex['answer']}\n\n"

            # n = len(test_entry['examples'])
            # rand_idx = list(np.random.permutation(np.arange(n)))
            # n = n // 2
            # single_exs = [test_entry['examples'][i] for i in rand_idx[:n]]
            # double_exs = [test_entry['examples'][i] for i in rand_idx[n:n*2]]

            # for ex in single_exs:
                # examples_prompt += f"Example question:{ex['question']}\nGround truth:[{ex['answer']}]\n\n"
            # for i in range(0, len(double_exs), 2):
                # ques = ' '.join(ex['question'] for ex in double_exs[i:i+2])
                # gt = ', '.join(ex['answer'] for ex in double_exs[i:i+2])
                # examples_prompt += f"Example question:{ques}\nGround truth:[{gt}]\n\n"

            n = len(test_entry['examples'])
            rand_idx = list(np.random.permutation(np.arange(n)))
            n = n // 3
            single_exs = [test_entry['examples'][i] for i in rand_idx[:n]]
            double_exs = [test_entry['examples'][i] for i in rand_idx[n:n*2]]
            triple_exs = [test_entry['examples'][i] for i in rand_idx[n*2:]]

            for ex in single_exs:
                examples_prompt += f"Example question:{ex['question']}\nGround truth:[{ex['answer']}]\n\n"
            for i in range(0, len(double_exs), 2):
                ques = ' '.join(ex['question'] for ex in double_exs[i:i+2])
                gt = ', '.join(ex['answer'] for ex in double_exs[i:i+2])
                examples_prompt += f"Example question:{ques}\nGround truth:[{gt}]\n\n"
            for i in range(0, len(triple_exs), 3):
                ques = ' '.join(ex['question'] for ex in triple_exs[i:i+3])
                gt = ', '.join(ex['answer'] for ex in triple_exs[i:i+3])
                examples_prompt += f"Example question:{ques}\nGround truth:[{gt}]\n\n"

            test_entry['question'][0].insert(
                0,
                {"role": "system", "content": examples_prompt},
            )

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )

        return {"message": [], "function": functions}
