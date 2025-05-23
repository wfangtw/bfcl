from bfcl.model_handler.handler_map import local_inference_handler_map

MODEL_METADATA_MAPPING = {
    "gorilla-openfunctions-v2": [
        "Gorilla-OpenFunctions-v2 (FC)",
        "https://gorilla.cs.berkeley.edu/blogs/7_open_functions_v2.html",
        "Gorilla LLM",
        "Apache 2.0",
    ],
    "o1-preview-2024-09-12": [
        "o1-preview-2024-09-12 (Prompt)",
        "https://openai.com/index/introducing-openai-o1-preview/",
        "OpenAI",
        "Proprietary",
    ],
    "o1-mini-2024-09-12": [
        "o1-mini-2024-09-12 (Prompt)",
        "https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-2024-11-20": [
        "GPT-4o-2024-11-20 (Prompt)",
        "https://openai.com/index/hello-gpt-4o/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-2024-11-20-FC": [
        "GPT-4o-2024-11-20 (FC)",
        "https://openai.com/index/hello-gpt-4o/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-2024-08-06": [
        "GPT-4o-2024-08-06 (Prompt)",
        "https://openai.com/index/hello-gpt-4o/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-2024-08-06-FC": [
        "GPT-4o-2024-08-06 (FC)",
        "https://openai.com/index/hello-gpt-4o/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-2024-05-13-FC": [
        "GPT-4o-2024-05-13 (FC)",
        "https://openai.com/index/hello-gpt-4o/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-2024-05-13": [
        "GPT-4o-2024-05-13 (Prompt)",
        "https://openai.com/index/hello-gpt-4o/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-mini-2024-07-18": [
        "GPT-4o-mini-2024-07-18 (Prompt)",
        "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-mini-2024-07-18-FC": [
        "GPT-4o-mini-2024-07-18 (FC)",
        "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-1106-preview-FC": [
        "GPT-4-1106-Preview (FC)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-1106-preview": [
        "GPT-4-1106-Preview (Prompt)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-0125-preview-FC": [
        "GPT-4-0125-Preview (FC)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-0125-preview": [
        "GPT-4-0125-Preview (Prompt)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-turbo-2024-04-09-FC": [
        "GPT-4-turbo-2024-04-09 (FC)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-turbo-2024-04-09": [
        "GPT-4-turbo-2024-04-09 (Prompt)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "claude-3-opus-20240229-FC": [
        "Claude-3-Opus-20240229 (FC)",
        "https://www.anthropic.com/news/claude-3-family",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-opus-20240229": [
        "Claude-3-Opus-20240229 (Prompt)",
        "https://www.anthropic.com/news/claude-3-family",
        "Anthropic",
        "Proprietary",
    ],
    "open-mistral-nemo-2407": [
        "Open-Mistral-Nemo-2407 (Prompt)",
        "https://mistral.ai/news/mistral-nemo/",
        "Mistral AI",
        "Proprietary",
    ],
    "open-mistral-nemo-2407-FC": [
        "Open-Mistral-Nemo-2407 (FC)",
        "https://mistral.ai/news/mistral-nemo/",
        "Mistral AI",
        "Proprietary",
    ],
    "open-mixtral-8x22b": [
        "Open-Mixtral-8x22b (Prompt)",
        "https://mistral.ai/news/mixtral-8x22b/",
        "Mistral AI",
        "Proprietary",
    ],
    "open-mixtral-8x22b-FC": [
        "Open-Mixtral-8x22b (FC)",
        "https://mistral.ai/news/mixtral-8x22b/",
        "Mistral AI",
        "Proprietary",
    ],
    "open-mixtral-8x7b": [
        "Open-Mixtral-8x7b (Prompt)",
        "https://mistral.ai/news/mixtral-of-experts/",
        "Mistral AI",
        "Proprietary",
    ],
    "mistral-medium-2312": [
        "Mistral-Medium-2312 (Prompt)",
        "https://docs.mistral.ai/guides/model-selection/",
        "Mistral AI",
        "Proprietary",
    ],
    "mistral-small-2402": [
        "Mistral-Small-2402 (Prompt)",
        "https://docs.mistral.ai/guides/model-selection/",
        "Mistral AI",
        "Proprietary",
    ],
    "mistral-large-2407": [
        "mistral-large-2407 (Prompt)",
        "https://mistral.ai/news/mistral-large-2407/",
        "Mistral AI",
        "Proprietary",
    ],
    "claude-3-sonnet-20240229-FC": [
        "Claude-3-Sonnet-20240229 (FC)",
        "https://www.anthropic.com/news/claude-3-family",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-sonnet-20240229": [
        "Claude-3-Sonnet-20240229 (Prompt)",
        "https://www.anthropic.com/news/claude-3-family",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-haiku-20240307-FC": [
        "Claude-3-Haiku-20240307 (FC)",
        "https://www.anthropic.com/news/claude-3-family",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-haiku-20240307": [
        "Claude-3-Haiku-20240307 (Prompt)",
        "https://www.anthropic.com/news/claude-3-family",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-5-haiku-20241022-FC": [
        "claude-3.5-haiku-20241022 (FC)",
        "https://www.anthropic.com/news/3-5-models-and-computer-use",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-5-haiku-20241022": [
        "claude-3.5-haiku-20241022 (Prompt)",
        "https://www.anthropic.com/news/3-5-models-and-computer-use",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-5-sonnet-20240620-FC": [
        "Claude-3.5-Sonnet-20240620 (FC)",
        "https://www.anthropic.com/news/claude-3-5-sonnet",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-5-sonnet-20240620": [
        "Claude-3.5-Sonnet-20240620 (Prompt)",
        "https://www.anthropic.com/news/claude-3-5-sonnet",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-5-sonnet-20241022-FC": [
        "Claude-3.5-Sonnet-20241022 (FC)",
        "https://www.anthropic.com/news/3-5-models-and-computer-use",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-5-sonnet-20241022": [
        "Claude-3.5-Sonnet-20241022 (Prompt)",
        "https://www.anthropic.com/news/3-5-models-and-computer-use",
        "Anthropic",
        "Proprietary",
    ],
    "gpt-3.5-turbo-0125-FC": [
        "GPT-3.5-Turbo-0125 (FC)",
        "https://platform.openai.com/docs/models/gpt-3-5-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-3.5-turbo-0125": [
        "GPT-3.5-Turbo-0125 (Prompt)",
        "https://platform.openai.com/docs/models/gpt-3-5-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "meetkai/functionary-small-v3.1-FC": [
        "Functionary-Small-v3.1 (FC)",
        "https://huggingface.co/meetkai/functionary-small-v3.1",
        "MeetKai",
        "MIT",
    ],
    "meetkai/functionary-medium-v3.1-FC": [
        "Functionary-Medium-v3.1 (FC)",
        "https://huggingface.co/meetkai/functionary-medium-v3.1",
        "MeetKai",
        "MIT",
    ],
    "claude-2.1": [
        "Claude-2.1 (Prompt)",
        "https://www.anthropic.com/news/claude-2-1",
        "Anthropic",
        "Proprietary",
    ],
    "mistral-tiny-2312": [
        "Mistral-tiny-2312 (Prompt)",
        "https://docs.mistral.ai/guides/model-selection/",
        "Mistral AI",
        "Proprietary",
    ],
    "claude-instant-1.2": [
        "Claude-instant-1.2 (Prompt)",
        "https://www.anthropic.com/news/releasing-claude-instant-1-2",
        "Anthropic",
        "Proprietary",
    ],
    "mistral-small-2402-FC": [
        "Mistral-small-2402 (FC)",
        "https://docs.mistral.ai/guides/model-selection/",
        "Mistral AI",
        "Proprietary",
    ],
    "mistral-large-2407-FC": [
        "mistral-large-2407 (FC)",
        "https://mistral.ai/news/mistral-large-2407/",
        "Mistral AI",
        "Proprietary",
    ],
    "Nexusflow-Raven-v2": [
        "Nexusflow-Raven-v2 (FC)",
        "https://huggingface.co/Nexusflow/NexusRaven-V2-13B",
        "Nexusflow",
        "Apache 2.0",
    ],
    "firefunction-v1-FC": [
        "FireFunction-v1 (FC)",
        "https://huggingface.co/fireworks-ai/firefunction-v1",
        "Fireworks",
        "Apache 2.0",
    ],
    "firefunction-v2-FC": [
        "FireFunction-v2 (FC)",
        "https://huggingface.co/fireworks-ai/firefunction-v2",
        "Fireworks",
        "Apache 2.0",
    ],
    "gemini-1.5-pro-002": [
        "Gemini-1.5-Pro-002 (Prompt)",
        "https://deepmind.google/technologies/gemini/pro/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-pro-002-FC": [
        "Gemini-1.5-Pro-002 (FC)",
        "https://deepmind.google/technologies/gemini/pro/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-pro-001": [
        "Gemini-1.5-Pro-001 (Prompt)",
        "https://deepmind.google/technologies/gemini/pro/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-pro-001-FC": [
        "Gemini-1.5-Pro-001 (FC)",
        "https://deepmind.google/technologies/gemini/pro/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-flash-002": [
        "Gemini-1.5-Flash-002 (Prompt)",
        "https://deepmind.google/technologies/gemini/flash/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-flash-002-FC": [
        "Gemini-1.5-Flash-002 (FC)",
        "https://deepmind.google/technologies/gemini/flash/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-flash-001": [
        "Gemini-1.5-Flash-001 (Prompt)",
        "https://deepmind.google/technologies/gemini/flash/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-flash-001-FC": [
        "Gemini-1.5-Flash-001 (FC)",
        "https://deepmind.google/technologies/gemini/flash/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.0-pro-002": [
        "Gemini-1.0-Pro-002 (Prompt)",
        "https://deepmind.google/technologies/gemini/pro/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.0-pro-002-FC": [
        "Gemini-1.0-Pro-002 (FC)",
        "https://deepmind.google/technologies/gemini/pro/",
        "Google",
        "Proprietary",
    ],
    "gpt-4-0613-FC": [
        "GPT-4-0613 (FC)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-0613": [
        "GPT-4-0613 (Prompt)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "deepseek-ai/DeepSeek-V2.5": [
        "DeepSeek-V2.5 (FC)",
        "https://huggingface.co/deepseek-ai/DeepSeek-V2.5",
        "DeepSeek",
        "DeepSeek License"
    ],
    "deepseek-ai/DeepSeek-Coder-V2-Instruct-0724": [
        "DeepSeek-Coder-V2 (FC)",
        "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct-0724",
        "DeepSeek",
        "DeepSeek License"
    ],
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": [
        "DeepSeek-Coder-V2-Lite-Instruct (FC)",
        "https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "DeepSeek",
        "DeepSeek License"
    ],
    "deepseek-ai/DeepSeek-V2-Chat-0628": [
        "DeepSeek-V2 (Prompt)",
        "https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628",
        "DeepSeek",
        "DeepSeek License",
    ],
    "deepseek-ai/DeepSeek-V2-Lite-Chat": [
        "DeepSeek-V2-Lite (Prompt)",
        "https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat",
        "DeepSeek",
        "DeepSeek License",
    ],
    "google/gemma-7b-it": [
        "Gemma-7b-it (Prompt)",
        "https://blog.google/technology/developers/gemma-open-models/",
        "Google",
        "gemma-terms-of-use",
    ],
    "google/gemma-2-2b-it": [
        "Gemma-2-2b-it (Prompt)",
        "https://blog.google/technology/developers/gemma-open-models/",
        "Google",
        "gemma-terms-of-use",
    ],
    "google/gemma-2-9b-it": [
        "Gemma-2-9b-it (Prompt)",
        "https://blog.google/technology/developers/gemma-open-models/",
        "Google",
        "gemma-terms-of-use",
    ],
    "google/gemma-2-27b-it": [
        "Gemma-2-27b-it (Prompt)",
        "https://blog.google/technology/developers/gemma-open-models/",
        "Google",
        "gemma-terms-of-use",
    ],
    "glaiveai/glaive-function-calling-v1": [
        "Glaive-v1 (FC)",
        "https://huggingface.co/glaiveai/glaive-function-calling-v1",
        "Glaive",
        "cc-by-sa-4.0",
    ],
    "databricks-dbrx-instruct": [
        "DBRX-Instruct (Prompt)",
        "https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm",
        "Databricks",
        "Databricks Open Model",
    ],
    "NousResearch/Hermes-2-Pro-Llama-3-8B": [
        "Hermes-2-Pro-Llama-3-8B (FC)",
        "https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B",
        "NousResearch",
        "apache-2.0",
    ],
    "NousResearch/Hermes-2-Pro-Llama-3-70B": [
        "Hermes-2-Pro-Llama-3-70B (FC)",
        "https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B",
        "NousResearch",
        "apache-2.0",
    ],
    "NousResearch/Hermes-2-Pro-Mistral-7B": [
        "Hermes-2-Pro-Mistral-7B (FC)",
        "https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B",
        "NousResearch",
        "apache-2.0",
    ],
    "NousResearch/Hermes-2-Theta-Llama-3-8B": [
        "Hermes-2-Theta-Llama-3-8B (FC)",
        "https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B",
        "NousResearch",
        "apache-2.0",
    ],
    "NousResearch/Hermes-2-Theta-Llama-3-70B": [
        "Hermes-2-Theta-Llama-3-70B (FC)",
        "https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B",
        "NousResearch",
        "apache-2.0",
    ],
    "meta-llama/Meta-Llama-3-8B-Instruct": [
        "Meta-Llama-3-8B-Instruct (Prompt)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Meta-Llama-3-70B-Instruct": [
        "Meta-Llama-3-70B-Instruct (Prompt)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3.1-8B-Instruct": [
        "Llama-3.1-8B-Instruct (Prompt)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/llama-3-1-70b-instruct": [
        "Llama-3.1-70B-Instruct (Prompt)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3.2-1B-Instruct": [
        "Llama-3.2-1B-Instruct (Prompt)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3.2-3B-Instruct": [
        "Llama-3.2-3B-Instruct (Prompt)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3-8B-Instruct-FC": [
        "Meta-Llama-3-8B-Instruct (FC)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3.1-8B-Instruct-FC": [
        "Llama-3.1-8B-Instruct (FC)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/llama-3-1-70B-instruct-FC": [
        "Llama-3.1-70B-Instruct (FC)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/llama-3-3-70B-instruct-FC": [
        "Llama-3.3-70B-Instruct (FC)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "command-r-plus-FC": [
        "Command-R-Plus (FC) (Original)",
        "https://txt.cohere.com/command-r-plus-microsoft-azure",
        "Cohere For AI",
        "cc-by-nc-4.0",
    ],
    "command-r-plus": [
        "Command-R-Plus (Prompt) (Original)",
        "https://txt.cohere.com/command-r-plus-microsoft-azure",
        "Cohere For AI",
        "cc-by-nc-4.0",
    ],
    "command-r-plus-FC-optimized": [
        "Command-R-Plus (FC) (Optimized)",
        "https://txt.cohere.com/command-r-plus-microsoft-azure",
        "Cohere For AI",
        "cc-by-nc-4.0",
    ],
    "command-r-plus-optimized": [
        "Command-R-Plus (Prompt) (Optimized)",
        "https://txt.cohere.com/command-r-plus-microsoft-azure",
        "Cohere For AI",
        "cc-by-nc-4.0",
    ],
    "snowflake/arctic": [
        "Snowflake/snowflake-arctic-instruct (Prompt)",
        "https://huggingface.co/Snowflake/snowflake-arctic-instruct",
        "Snowflake",
        "apache-2.0",
    ],
    "nvidia/nemotron-4-340b-instruct": [
        "Nemotron-4-340b-instruct (Prompt)",
        "https://huggingface.co/nvidia/nemotron-4-340b-instruct",
        "NVIDIA",
        "nvidia-open-model-license",
    ],
    "ibm-granite/granite-20b-functioncalling": [
        "Granite-20b-FunctionCalling (FC)",
        "https://huggingface.co/ibm-granite/granite-20b-functioncalling",
        "IBM",
        "Apache-2.0",
    ],
    "THUDM/glm-4-9b-chat": [
        "GLM-4-9b-Chat (FC)",
        "https://huggingface.co/THUDM/glm-4-9b-chat",
        "THUDM",
        "glm-4",
    ],
    "yi-large-fc": [
        "yi-large (FC)",
        "https://platform.01.ai/",
        "01.AI",
        "Proprietary",
    ],
    "Salesforce/xLAM-1b-fc-r": [
        "xLAM-1b-fc-r (FC)",
        "https://huggingface.co/Salesforce/xLAM-1b-fc-r",
        "Salesforce",
        "cc-by-nc-4.0",
    ],
    "Salesforce/xLAM-7b-fc-r": [
        "xLAM-7b-fc-r (FC)",
        "https://huggingface.co/Salesforce/xLAM-7b-fc-r",
        "Salesforce",
        "cc-by-nc-4.0",
    ],
    "Salesforce/xLAM-7b-r": [
        "xLAM-7b-r (FC)",
        "https://huggingface.co/Salesforce/xLAM-7b-r",
        "Salesforce",
        "cc-by-nc-4.0",
    ],
    "Salesforce/xLAM-8x7b-r": [
        "xLAM-8x7b-r (FC)",
        "https://huggingface.co/Salesforce/xLAM-8x7b-r",
        "Salesforce",
        "cc-by-nc-4.0",
    ],
    "Salesforce/xLAM-8x22b-r": [
        "xLAM-8x22b-r (FC)",
        "https://huggingface.co/Salesforce/xLAM-8x22b-r",
        "Salesforce",
        "cc-by-nc-4.0",
    ],
    "MadeAgents/Hammer2.0-7b": [
        "Hammer2.0-7b (FC)",
        "https://huggingface.co/MadeAgents/Hammer2.0-7b",
        "MadeAgents",
        "cc-by-nc-4.0",
    ],
    "MadeAgents/Hammer2.0-3b": [
        "Hammer2.0-3b (FC)",
        "https://huggingface.co/MadeAgents/Hammer2.0-3b",
        "MadeAgents",
        "cc-by-nc-4.0",
    ],
    "MadeAgents/Hammer2.0-1.5b": [
        "Hammer2.0-1.5b (FC)",
        "https://huggingface.co/MadeAgents/Hammer2.0-1.5b",
        "MadeAgents",
        "cc-by-nc-4.0",
    ],
    "MadeAgents/Hammer2.0-0.5b": [
        "Hammer2.0-0.5b (FC)",
        "https://huggingface.co/MadeAgents/Hammer2.0-0.5b",
        "MadeAgents",
        "cc-by-nc-4.0",
    ],
    "microsoft/Phi-3-mini-4k-instruct": [
        "Phi-3-mini-4k-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
        "Microsoft",
        "MIT",
    ],
    "microsoft/Phi-3-mini-128k-instruct": [
        "Phi-3-mini-128k-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3-mini-128k-instruct",
        "Microsoft",
        "MIT",
    ],
    "microsoft/Phi-3-small-8k-instruct": [
        "Phi-3-small-8k-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3-small-8k-instruct",
        "Microsoft",
        "MIT",
    ],
    "microsoft/Phi-3-small-128k-instruct": [
        "Phi-3-small-128k-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3-small-128k-instruct",
        "Microsoft",
        "MIT",
    ],
    "microsoft/Phi-3-medium-4k-instruct": [
        "Phi-3-medium-4k-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3-medium-4k-instruct",
        "Microsoft",
        "MIT",
    ],
    "microsoft/Phi-3-medium-128k-instruct": [
        "Phi-3-medium-128k-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3-medium-128k-instruct",
        "Microsoft",
        "MIT",
    ],
    "microsoft/Phi-3.5-mini-instruct": [
        "Phi-3.5-mini-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
        "Microsoft",
        "MIT",
    ],
    "Qwen/Qwen2-1.5B-Instruct": [
        "Qwen2-1.5B-Instruct (Prompt)",
        "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct",
        "Qwen",
        "apache-2.0",
    ],
    "Qwen/Qwen2-7B-Instruct": [
        "Qwen2-7B-Instruct (Prompt)",
        "https://huggingface.co/Qwen/Qwen2-7B-Instruct",
        "Qwen",
        "apache-2.0",
    ],
    "Qwen/Qwen2.5-1.5B-Instruct": [
        "Qwen2.5-1.5B-Instruct (Prompt)",
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen",
        "apache-2.0",
    ],
    "Qwen/Qwen2.5-7B-Instruct": [
        "Qwen2.5-7B-Instruct (Prompt)",
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        "Qwen",
        "apache-2.0",
    ],
    "Qwen/Qwen2.5-72B-Instruct": [
        "Qwen2.5-72B-Instruct (Prompt)",
        "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct",
        "Qwen",
        "apache-2.0",
    ],
    "Team-ACE/ToolACE-8B": [
        "ToolACE-8B (FC)",
        "https://huggingface.co/Team-ACE/ToolACE-8B",
        "Huawei Noah & USTC",
        "Apache-2.0",
    ],
    "openbmb/MiniCPM3-4B": [
        "MiniCPM3-4B (Prompt)",
        "https://huggingface.co/openbmb/MiniCPM3-4B",
        "openbmb",
        "Apache-2.0",
    ],
    "openbmb/MiniCPM3-4B-FC": [
        "MiniCPM3-4B-FC (FC)",
        "https://huggingface.co/openbmb/MiniCPM3-4B",
        "openbmb",
        "Apache-2.0",
    ],
    "BitAgent/GoGoAgent": [
        "GoGoAgent",
        "https://gogoagent.ai",
        "BitAgent",
        "Proprietary",
    ],
    "palmyra-x-004": [
        "palmyra-x-004 (FC)",
        "https://writer.com/engineering/actions-with-palmyra-x-004/",
        "Writer",
        "Proprietary",
    ],
    "grok-beta": [
        "Grok-beta (FC)",
        "https://x.ai/",
        "xAI",
        "Proprietary",
    ],
}

INPUT_PRICE_PER_MILLION_TOKEN = {
    "claude-3-opus-20240229-FC": 15,
    "claude-3-opus-20240229": 15,
    "claude-3-sonnet-20240229-FC": 3,
    "claude-3-sonnet-20240229": 3,
    "claude-3-5-sonnet-20240620-FC": 3,
    "claude-3-5-sonnet-20240620": 3,
    "claude-3-5-sonnet-20241022-FC": 3,
    "claude-3-5-sonnet-20241022": 3,
    "claude-3-haiku-20240307-FC": 0.25,
    "claude-3-haiku-20240307": 0.25,
    "claude-3-5-haiku-20241022-FC": 1,
    "claude-3-5-haiku-20241022": 1,
    "claude-2.1": 8,
    "claude-instant-1.2": 0.8,
    "open-mistral-nemo-2407": 0.3,
    "open-mistral-nemo-2407-FC": 0.3,
    "open-mixtral-8x22b": 2,
    "open-mixtral-8x22b-FC": 2,
    "open-mixtral-8x7b": 0.7,
    "mistral-large-2407": 3,
    "mistral-large-2407-FC": 3,
    "mistral-medium-2312": 2.7,
    "mistral-small-2402-FC": 1,
    "mistral-small-2402": 1,
    "mistral-tiny-2312": 0.25,
    "o1-preview-2024-09-12": 15,
    "o1-mini-2024-09-12": 3,
    "gpt-4o-2024-05-13-FC": 5,
    "gpt-4o-2024-05-13": 5,
    "gpt-4o-2024-08-06-FC": 2.5,
    "gpt-4o-2024-08-06": 2.5,
    "gpt-4o-2024-11-20-FC": 2.5,
    "gpt-4o-2024-11-20": 2.5,
    "gpt-4o-mini-2024-07-18": 0.15,
    "gpt-4o-mini-2024-07-18-FC": 0.15,
    "gpt-4-1106-preview-FC": 10,
    "gpt-4-1106-preview": 10,
    "gpt-4-0125-preview": 10,
    "gpt-4-0125-preview-FC": 10,
    "gpt-4-turbo-2024-04-09-FC": 10,
    "gpt-4-turbo-2024-04-09": 10,
    "gpt-4-0613": 30,
    "gpt-4-0613-FC": 30,
    "gpt-3.5-turbo-0125": 0.5,
    "gpt-3.5-turbo-0125-FC": 0.5,
    "gemini-1.5-pro-002": 1.25,
    "gemini-1.5-pro-002-FC": 1.25,
    "gemini-1.5-pro-001": 1.25,
    "gemini-1.5-pro-001-FC": 1.25,
    "gemini-1.5-flash-002": 0.075,
    "gemini-1.5-flash-002-FC": 0.075,
    "gemini-1.5-flash-001": 0.075,
    "gemini-1.5-flash-001-FC": 0.075,
    "gemini-1.0-pro-002": 0.5,
    "gemini-1.0-pro-002-FC": 0.5,
    "databricks-dbrx-instruct": 2.25,
    "command-r-plus-FC": 3,
    "command-r-plus": 3,
    "command-r-plus-FC-optimized": 3,
    "command-r-plus-optimized": 3,
    "yi-large-fc": 3,
    "palmyra-x-004": 5,
    "grok-beta": 5,
}

OUTPUT_PRICE_PER_MILLION_TOKEN = {
    "claude-3-opus-20240229-FC": 75,
    "claude-3-opus-20240229": 75,
    "claude-3-sonnet-20240229-FC": 15,
    "claude-3-sonnet-20240229": 15,
    "claude-3-5-sonnet-20240620-FC": 15,
    "claude-3-5-sonnet-20240620": 15,
    "claude-3-5-sonnet-20241022-FC": 15,
    "claude-3-5-sonnet-20241022": 15,
    "claude-3-haiku-20240307-FC": 1.25,
    "claude-3-haiku-20240307": 1.25,
    "claude-3-5-haiku-20241022-FC": 5,
    "claude-3-5-haiku-20241022": 5,
    "claude-2.1": 24,
    "claude-instant-1.2": 2.4,
    "open-mistral-nemo-2407": 0.3,
    "open-mistral-nemo-2407-FC": 0.3,
    "open-mixtral-8x22b": 6,
    "open-mixtral-8x22b-FC": 6,
    "open-mixtral-8x7b": 0.7,
    "mistral-large-2407": 9,
    "mistral-large-2407-FC": 9,
    "mistral-small-2402": 3,
    "mistral-medium-2312": 8.1,
    "mistral-small-2402-FC": 3,
    "mistral-tiny-2312": 0.25,
    "o1-preview-2024-09-12": 60,
    "o1-mini-2024-09-12": 12,
    "gpt-4o-2024-05-13-FC": 15,
    "gpt-4o-2024-05-13": 15,
    "gpt-4o-2024-08-06-FC": 10,
    "gpt-4o-2024-08-06": 10,
    "gpt-4o-2024-11-20-FC": 10,
    "gpt-4o-2024-11-20": 10,
    "gpt-4o-mini-2024-07-18": 0.6,
    "gpt-4o-mini-2024-07-18-FC": 0.6,
    "gpt-4-turbo-2024-04-09-FC": 30,
    "gpt-4-turbo-2024-04-09": 30,
    "gpt-4-1106-preview": 30,
    "gpt-4-1106-preview-FC": 30,
    "gpt-4-0125-preview-FC": 30,
    "gpt-4-0125-preview": 30,
    "gpt-4-0613": 60,
    "gpt-4-0613-FC": 60,
    "gpt-3.5-turbo-0125": 1.5,
    "gpt-3.5-turbo-0125-FC": 1.5,
    "gemini-1.5-pro-002": 5,
    "gemini-1.5-pro-002-FC": 5,
    "gemini-1.5-pro-001": 5,
    "gemini-1.5-pro-001-FC": 5,
    "gemini-1.5-flash-002": 0.30,
    "gemini-1.5-flash-002-FC": 0.30,
    "gemini-1.5-flash-001": 0.30,
    "gemini-1.5-flash-001-FC": 0.30,
    "gemini-1.0-pro-002": 1.5,
    "gemini-1.0-pro-002-FC": 1.5,
    "databricks-dbrx-instruct": 6.75,
    "command-r-plus-FC": 15,
    "command-r-plus": 15,
    "command-r-plus-FC-optimized": 15,
    "command-r-plus-optimized": 15,
    "yi-large-fc": 3,
    "palmyra-x-004": 12,
    "grok-beta": 15,
}

# The latency of the open-source models are hardcoded here.
# Because we do batching when generating the data, so the latency is not accurate from the result data.
# This is the latency for the whole batch of data, when using 8 V100 GPUs.
OSS_LATENCY = {}

# All OSS models will have no cost shown on the leaderboard.
NO_COST_MODELS = list(local_inference_handler_map.keys())
# The following models will also have no cost, even though they are queries through the API.
NO_COST_MODELS += [
    "Nexusflow-Raven-v2",
    "firefunction-v1-FC",
    "firefunction-v2-FC",
    "meetkai/functionary-small-v3.1-FC",
    "meetkai/functionary-medium-v3.1-FC",
    "snowflake/arctic",
    "nvidia/nemotron-4-340b-instruct",
]
