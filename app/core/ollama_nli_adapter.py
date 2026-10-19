import json


class OllamaNLIAdapter:
    def __init__(self, model):
        self.model = model

    def chat(self, system_prompt, user_prompt):
        result = self.model.generate_json(
            system_prompt,
            user_prompt,
            {
                "type": "object",
                "additionalProperties": True,
            },
        )
        if not isinstance(result, dict):
            raise ValueError("Expected JSON object")
        return json.dumps(result, ensure_ascii=False)
