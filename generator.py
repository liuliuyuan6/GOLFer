import transformers
import torch

class Generator:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def generate(self):
        return ""


class LlamaGenerator():   
    def __init__(self, model_name, n=8, max_tokens=256, temperature=0.6, top_p=0.9):
        self.model_name=model_name
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p


    def generate(self, prompt_text):
        hypothesis_documents=[]
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )
        messages = [
            {"role": "user", "content": prompt_text},
        ]
        prompt = pipeline.tokenizer.apply_chat_template(
		    messages, 
		    tokenize=False, 
		    add_generation_prompt=True
        )
        terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        for j in range(n):
            outputs = pipeline(
                prompt,
                max_new_tokens=self.max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            hypothesis_documents=outputs[0]["generated_text"][len(prompt):].replace('\n','')
        return hypothesis_documents


