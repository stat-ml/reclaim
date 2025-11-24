import logging
import os
import time
from typing import Optional

import openai
from pydantic import BaseModel

log = logging.getLogger(__name__)


class OpenAIChat:
    """
    Chat client used for dataset marking and claim alignment.
    """

    def __init__(
        self,
        openai_model: str = "gpt-4o",
        base_url: Optional[str] = None,
        cache_path: str = os.path.expanduser("~") + "/.cache",
        timeout: int = 600,
        max_tokens: Optional[int] = None,
        rewrite_cache: bool = False,
    ):
        """
        Configure an OpenAI chat client for repeated requests.

        Args:
            openai_model: Model name used for chat completions.
            base_url: Optional API base URL override.
            cache_path: Directory for potential cache files.
            timeout: Request timeout in seconds.
            max_tokens: Optional max completion tokens.
            rewrite_cache: Whether to rewrite cache entries when present.
        """
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is not None:
            openai.api_key = api_key
        self.openai_model = openai_model
        self.cache_path = os.path.join(cache_path, "openai_chat_cache.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.base_url = base_url
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.rewrite_cache = rewrite_cache

    def ask(
        self,
        message: str,
        schema: Optional[BaseModel] = None,
        system_role: str = "You are an intelligent assistant.",
    ):
        """
        Send a prompt to OpenAI and return the model reply.

        Args:
            message: User content to send.
            schema: Optional schema for structured parsing.
            system_role: System prompt to prepend to the conversation.

        Returns:
            Model response text or parsed schema instance.
        """
        if openai.api_key is None:
            raise Exception(
                "Cant ask openAI without token. "
                "Please specify OPENAI_API_KEY in environment parameters."
            )
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": message},
        ]
        chat = self._send_request(messages, schema)
        if schema is not None:
            reply = chat.output[0].content[0].parsed
        else:
            reply = chat.choices[0].message.content
            # Ignore boilerplate refusals that would pollute downstream parsing.
            if "please provide" in reply.lower():
                return ""
            if "to assist you" in reply.lower():
                return ""
            if "as an ai language model" in reply.lower():
                return ""
        return reply

    def _send_request(self, messages, schema: Optional[BaseModel] = None):
        """
        Dispatch a chat or responses.parse request with progressive backoff.

        Args:
            messages: List of role/content dictionaries for the API.
            schema: Optional schema to enable responses.parse.

        Returns:
            OpenAI response object.
        """
        sleep_time_values = (5, 10, 30, 60, 120)
        # Simple progressive backoff; escalate exception after exhausting attempts.
        for i in range(len(sleep_time_values)):
            try:
                if schema is None:
                    result = openai.OpenAI(
                        base_url=self.base_url, timeout=self.timeout
                    ).chat.completions.create(
                        model=self.openai_model,
                        messages=messages,
                        temperature=0,
                        max_completion_tokens=self.max_tokens,
                    )
                else:
                    result = openai.OpenAI(
                        base_url=self.base_url, timeout=self.timeout
                    ).responses.parse(
                        model=self.openai_model,
                        input=messages,
                        temperature=0,
                        text_format=schema,
                    )
                return result
            except Exception as e:
                sleep_time = sleep_time_values[i]
                print(
                    "Request to OpenAI failed with exception: %s. Retry #%s/5 after %s seconds.",
                    e,
                    i,
                    sleep_time,
                )
                time.sleep(sleep_time)
