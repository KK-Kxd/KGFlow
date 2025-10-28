from transformers import AutoModelForCausalLM, AutoTokenizer

class ChatModel:
    def __init__(self, model_path: str, tokenizer_path: str, max_token=2048 ,device: str = "auto"):

        self.model_identifier = model_path.lower()
        self.max_token = max_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model.generation_config.do_sample = None



    def generate_response(self, input_text: str, temperature: float = 1.0) -> str:

        messages = [{"role": "user", "content": input_text}]


        try:

            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                formatted_input = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:

                formatted_input = f"User: {input_text}\nAssistant:"
        except Exception:

            formatted_input = f"User: {input_text}\nAssistant:"

        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)
        input_token_length = inputs.input_ids.shape[1]


        if temperature == 0:

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_token,
                do_sample=False
            )
        else:

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_token,
                do_sample=True,
                temperature=temperature
            )
        

        generated_token_ids = outputs[0][input_token_length:]
        response = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()


        if 'qwen3' in self.model_identifier:
            marker = "</think>"
            if marker in response:
                response = response.rsplit(marker, 1)[-1].strip()
        if 'huatuo' in self.model_identifier:
            marker = "## Final Response"
            if marker in response:
                response = response.rsplit(marker, 1)[-1].strip()
        
        return response