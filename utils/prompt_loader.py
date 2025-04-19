import os

def load_prompt(template_name: str, replacements: dict):
    prompt_path = os.path.join("prompts", template_name)
    with open(prompt_path, "r") as f:
        prompt = f.read()
    for key, val in replacements.items():
        prompt = prompt.replace(f"{{{{ {key} }}}}", val)
    return prompt
