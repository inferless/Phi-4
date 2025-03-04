from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer

class InferlessPythonModel:
    def initialize(self):
        model_id = "microsoft/phi-4"
        self.llm = LLM(model=model_id,enforce_eager=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def infer(self, inputs):
        prompt = inputs["prompt"]
        system_prompt = inputs.get("system_prompt","You are a friendly bot.")
        temperature = inputs.get("temperature",0.7)
        top_p = inputs.get("top_p",0.1)
        repetition_penalty = inputs.get("repetition_penalty",1.18)
        top_k = int(inputs.get("top_k",40))
        max_tokens = inputs.get("max_tokens",256)

        chat_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        sampling_params = SamplingParams(temperature=temperature,top_p=top_p,
                                         repetition_penalty=repetition_penalty,
                                         top_k=top_k,max_tokens=max_tokens
                                        )
        chat_format = self.tokenizer.apply_chat_template(chat_message,tokenize=False,add_generation_prompt=True)
        result = self.llm.generate(chat_format, sampling_params)
        result_output = [output.outputs[0].text for output in result]
        
        return {"generated_text":result_output[0]}
        
    def finalize(self):
        self.llm = None
