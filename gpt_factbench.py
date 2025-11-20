import time
import openai
from openai import OpenAI

def gpt_single_try(user_input, model = "gpt-4o", system_role = "You are a helpful assistant.", schema=None):
    if schema is not None:
        response = OpenAI().responses.parse(
            model=model,
            input=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_input},
            ],
            text_format=schema
        )

        return response.output[0].content[0].parsed

    else:
        response = OpenAI().chat.completions.create(
            model=model,
            messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_input},
            ]
        )

        result = ''

        for choice in response.choices:
            result += choice.message.content

        return result


def gpt(user_input, model = "gpt-4o", 
        system_role="You are a helpful assistant.", 
        num_retries=3, waiting_time = 1, schema=None):
    r = ''
    for _ in range(num_retries):
        try:
            r = gpt_single_try(user_input, model, system_role, schema)
            break
        except openai.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(waiting_time)
    return r

