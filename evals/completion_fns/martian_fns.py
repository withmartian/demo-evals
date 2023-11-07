import os
from typing import Optional

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import CompletionPrompt
from evals.record import record_sampling

import requests
import json
import time
import random
from tenacity import retry, stop_after_attempt, wait_random

import openai


class LangChainLLMCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class RouterCompletionFn(CompletionFn):
    def __init__(self, llm_kwargs: Optional[dict] = None, **kwargs) -> None:
        self.router_url = 'https://route.withmartian.com/router-backend2/api/router/'
        self.headers = {
            'Authorization': os.environ.get('MARTIAN_BEARER_TOKEN'),
            'Content-Type': 'application/json'
        }

    def call_router(self, prompt: str) -> str:
        data = {'conversation': [{'role': 'user', 'content': prompt}]}
        response = self._post_request(data)
        if 'status_code' in response and response['status_code'] != 200:
            raise Exception('Router error: ' + response['detail'])
        return response['response']['content']

    @retry(stop=stop_after_attempt(3))
    def _post_request(self, data):
        response = requests.post(self.router_url, headers=self.headers, json=data)
        response = response.json()
        return response


    def __call__(self, prompt, **kwargs) -> LangChainLLMCompletionResult:
        prompt = CompletionPrompt(prompt).to_formatted_prompt()
        response = self.call_router(prompt)
        record_sampling(prompt=prompt, sampled=response)
        return LangChainLLMCompletionResult(response)


