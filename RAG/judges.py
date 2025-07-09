from openai import OpenAI
import yaml
api_key= yaml.safe_load(open("/root/llm_NaiveFinetune/config/api.yaml","r"))['api_key']

class Judges:
    def __init__(self):
        self.rag_model = None
        self.system_setting_prompt = "你是一名问答评测员，请根据提示打分。"
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.gpt_model = "gpt-4o-mini"
        self.response_temperature = 0.1

    def gpt_judge(self, prompt):
        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": self.system_setting_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.response_temperature,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()
        # return response['choices'][0]['message']['content'].strip()
