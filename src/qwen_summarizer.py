import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

class QwenSummarizer:
    def __init__(self, model_id="Qwen/Qwen3.5-2B", device=None, use_lora=False, lora_weights_path=None, use_4bit=True):
        self.model_id = model_id
        self.use_4bit = use_4bit
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # Fallback for some environments
            if self.device == "cuda":
                try:
                    torch.isin(torch.tensor([1], device="cuda"), torch.tensor([1], device="cuda"))
                except Exception:
                    self.device = "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        self.dtype = torch.float32
        if self.device == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        print(f"Loading tokenizer {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        quantization_config = None
        if self.use_4bit and self.device == "cuda":
            print("Configuring 4-bit quantization via bitsandbytes...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
        print(f"Loading model {model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        
        if use_lora:
            self.model = self._setup_lora(lora_weights_path)
            
        if self.device != "cuda" or not self.use_4bit:
            self.model.to(self.device)
            
        self.model.eval()
        
    def _setup_lora(self, lora_weights_path=None):
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
            
        if lora_weights_path:
            print(f"Loading LoRA weights from {lora_weights_path}")
            model = PeftModel.from_pretrained(self.model, lora_weights_path)
        else:
            print("Initializing new LoRA config for fine-tuning")
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            model = get_peft_model(self.model, lora_config)
            model.print_trainable_parameters()
        return model

    def judge_paper(self, title, abstract):
        prompt = f"""Judge if this AI paper is genuinely worth reading. Be very strict.

Reject (NO) if the paper:
- Is an incremental improvement over existing work
- Only achieves SOTA by minor tweaks (architecture, hyperparams, more data)
- Proposes a method that only works on narrow benchmarks
- Lacks surprising or counter-intuitive findings
- Is a survey, benchmark, or dataset paper without novel insights

Accept (YES) only if the paper:
- Introduces a fundamentally new idea or paradigm
- Solves a long-standing open problem
- Has findings that would surprise experts in the field

Default to NO. Less than 10% of papers should pass.

Title: {title}
Abstract: {abstract}

Decision (YES/NO):"""
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,  # Use greedy search for yes/no
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        input_length = inputs["input_ids"].shape[1]
        summary_tokens = output_tokens[0][input_length:]
        decision = self.tokenizer.decode(summary_tokens, skip_special_tokens=True).strip().upper()
        return "YES" in decision

    def summarize(self, abstract, max_new_tokens=150, custom_prompt=None):
        if custom_prompt:
            prompt = custom_prompt.format(abstract=abstract)
        else:
            prompt = f"Summarize the following AI research paper abstract in a concise and easily understandable way:\n\nAbstract: {abstract}\n\nSummary:"
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        summary_tokens = output_tokens[0][input_length:]
        summary = self.tokenizer.decode(summary_tokens, skip_special_tokens=True)
        return summary
