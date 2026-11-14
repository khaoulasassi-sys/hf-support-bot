# Questions et réponses du chatbot (profil offline)

Réponses générées par le chatbot en mode extractif (sans LLM). Page cible = page de la documentation Hugging Face qui contient la réponse.


## Question couverte par la documentation (62 questions)

### q001 — I'm on a MacBook with an M2 chip and an operator isn't implemented for the Metal backend. Is there an environment variable to avoid the crash?

**Page cible :** apple_silicon  

**Réponse du bot** (décision : no_answer, confiance 13%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q002 — How can I limit how many CPU threads PyTorch uses when I run inference?

**Page cible :** cpu  

**Réponse du bot** (décision : confident, confiance 79%) :

**Answer (extracted from the documentation)**

- PyTorch is not async aware, so computation will block the main thread from running. [2]
- For this reason, it's better to run PyTorch on its own separate thread or process. [2]
- Some PyTorch operations don't have an MPS implementation yet. [3]

**Sources :**
- CPU > Next steps — https://huggingface.co/docs/transformers/cpu#next-steps
- Web server inference > Block the main thread — https://huggingface.co/docs/transformers/pipeline_webserver#block-the-main-thread
- Apple Silicon > Limitations — https://huggingface.co/docs/transformers/apple_silicon#limitations
- CPU > Train with Trainer — https://huggingface.co/docs/transformers/cpu#train-with-trainer

### q003 — What is the difference between WordPiece and Unigram tokenization?

**Page cible :** tokenizer_summary  

**Réponse du bot** (décision : confident, confiance 68%) :

**Answer (extracted from the documentation)**

- Chapter 6 of the LLM course teaches you how to train a tokenizer from scratch and explains the differences between the BPE, Unigram, and WordPiece algorithms. [1]
- Transformers support three subword tokenization algorithms: Byte pair encoding (BPE), Unigram, and WordPiece. [2]
- WordPiece is the tokenization algorithm for BERT-family models like DistilBERT and Electra. [3]

**Sources :**
- Tokenization algorithms > Resources — https://huggingface.co/docs/transformers/tokenizer_summary#resources
- Tokenization algorithms >  — https://huggingface.co/docs/transformers/tokenizer_summary
- Tokenization algorithms > WordPiece — https://huggingface.co/docs/transformers/tokenizer_summary#wordpiece
- Tokenization algorithms > SentencePiece — https://huggingface.co/docs/transformers/tokenizer_summary#sentencepiece

### q004 — Which subword algorithm does SentencePiece rely on?

**Page cible :** tokenizer_summary, fast_tokenizers  

**Réponse du bot** (décision : confident, confiance 75%) :

**Answer (extracted from the documentation)**

- SentencePiece is a tokenization library that applies BPE or Unigram directly on raw text. [1]
- SentencePiece treats the input text as a raw byte or character stream and includes the space character, represented as `"▁"`, in the vocabulary. [1]
- SentencePiece then applies BPE or Unigram to the text. [1]

**Sources :**
- Tokenization algorithms > SentencePiece — https://huggingface.co/docs/transformers/tokenizer_summary#sentencepiece
- Tokenizers > Backends — https://huggingface.co/docs/transformers/fast_tokenizers#backends
- Tokenization algorithms >  — https://huggingface.co/docs/transformers/tokenizer_summary
- Tokenization algorithms > WordPiece — https://huggingface.co/docs/transformers/tokenizer_summary#wordpiece

### q005 — Why do ALiBi and rotary position embeddings matter for long prompts?

**Page cible :** llm_tutorial_optimization  

**Réponse du bot** (décision : uncertain, confiance 47%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Recently, relative positional embeddings that can tackle the above mentioned problems have become more popular, most notably: Rotary Position Embedding (RoPE) ALiBi Both *RoPE* and *ALiBi* argue that it's best to cue the LLM about sentence order directly in the self-attention algorithm as it's there that word tokens are put into relation with each other. [1]
-  As shown in the ALiBi paper, this simple relative positional encoding allows the model to retain a high performance even at very long text input sequences. [2]
- Both RoPE and ALiBi are relative positional embeddings that are *not* learned during training, but instead are based on the following intuitions: Positional cues about the text inputs should be given directly to the $\mathbf{QK}^T$ matrix of the self-attention layer. [3]

**Sources :**
- Optimizing LLMs for Speed and Memory > 3. Architectural Innovations > 3.1 Improving positional embeddings of LLMs — https://huggingface.co/docs/transformers/llm_tutorial_optimization#31-improving-positional-embeddings-of-llms
- Optimizing LLMs for Speed and Memory > 3. Architectural Innovations > 3.1 Improving positional embeddings of LLMs — https://huggingface.co/docs/transformers/llm_tutorial_optimization#31-improving-positional-embeddings-of-llms
- Optimizing LLMs for Speed and Memory > 3. Architectural Innovations > 3.1 Improving positional embeddings of LLMs — https://huggingface.co/docs/transformers/llm_tutorial_optimization#31-improving-positional-embeddings-of-llms
- Optimizing LLMs for Speed and Memory > 3. Architectural Innovations > 3.1 Improving positional embeddings of LLMs — https://huggingface.co/docs/transformers/llm_tutorial_optimization#31-improving-positional-embeddings-of-llms

### q006 — Is there an optimizer that saves memory by projecting gradients into a low-rank space?

**Page cible :** optimizers  

**Réponse du bot** (décision : uncertain, confiance 43%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Tip: Set `scale` to `n/r`, where `n` is the original space dimension and `r` is the low-rank space dimension. [1]
- Low-Memory Optimization (LOMO) includes two optimizers for low-memory full-parameter finetuning, LOMO and AdaLomo. [1]
- Gradient Low-Rank Projection (GaLore) reduces memory for training LLMs. [3]

**Code example** [1]
```py
args = TrainingArguments(
    optim="apollo_adamw",
    optim_target_modules=[r".*.attn.*", r".*.mlp.*"],
    optim_args="proj=random,rank=1,scale=128.0,scale_type=tensor,update_proj_gap=200",
    ...  # remaining args from the TrainingArguments intro config
)
```

**Sources :**
- Optimizers and schedulers > Optimizer integrations — https://huggingface.co/docs/transformers/optimizers#optimizer-integrations
- Optimizers and schedulers > Optimizer integrations — https://huggingface.co/docs/transformers/optimizers#optimizer-integrations
- Optimizers and schedulers > Optimizer integrations — https://huggingface.co/docs/transformers/optimizers#optimizer-integrations
- Bitsandbytes >  — https://huggingface.co/docs/transformers/quantization/bitsandbytes

### q007 — Can I train without having to tune a learning rate schedule?

**Page cible :** optimizers  

**Réponse du bot** (décision : uncertain, confiance 39%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Some schedulers adapt to training dynamics instead of following a fixed schedule. [1]
- The scheduler wraps the optimizer and adjusts the learning rate each training step. [2]
- Schedule Free optimizer (SFO) replaces momentum with a combination of averaging and interpolation, completely removing the need to anneal the learning rate. [3]

**Code example** [2]
```py
from transformers import TrainingArguments

args = TrainingArguments(
    ...,
    # Optimizer
    optim="adamw_torch",          # or "adamw_torch_fused", "adafactor", "sgd", etc.
    learning_rate=2e-5,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    # Scheduler
    lr_scheduler_type="cosine",   # "linear", "cosine", "constant_with_warmup", etc.
    warmup_steps=500,
```

**Sources :**
- Optimizers and schedulers > Metric-based schedulers — https://huggingface.co/docs/transformers/optimizers#metric-based-schedulers
- Optimizers and schedulers >  — https://huggingface.co/docs/transformers/optimizers
- Optimizers and schedulers > Optimizer integrations — https://huggingface.co/docs/transformers/optimizers#optimizer-integrations
- Hyperparameter search > Define the search space — https://huggingface.co/docs/transformers/hpo_train#define-the-search-space

### q008 — What is LOMO and why would I use it for a big model?

**Page cible :** optimizers  

**Réponse du bot** (décision : uncertain, confiance 54%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- `PreTrainedModel.from_pretrained` is supercharged with Accelerate's Big Model Inference feature. [1]
- Big Model Inference creates a *model skeleton* on the PyTorch meta device. [1]
- Big Model Inference's second feature relates to how weights are loaded and dispatched in the model skeleton. [1]

**Code example** [1]
```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto")
```

**Sources :**
- Loading models > Large models > Big Model Inference — https://huggingface.co/docs/transformers/models#big-model-inference
- Loading models > Large models > Big Model Inference — https://huggingface.co/docs/transformers/models#big-model-inference
- Loading models > Large models — https://huggingface.co/docs/transformers/models#large-models
- Optimizers and schedulers > Optimizer integrations — https://huggingface.co/docs/transformers/optimizers#optimizer-integrations

### q009 — Can the Trainer run its hyperparameter search with Optuna?

**Page cible :** hpo_train  

**Réponse du bot** (décision : confident, confiance 77%) :

**Answer (extracted from the documentation)**

- `Trainer.hyperparameter_search` returns a `trainer.utils.BestRun` containing the objective value and best hyperparameter combination. [1]
- `Trainer.hyperparameter_search` finds the best combination by running multiple trials, each with a different set of values, and returning the best one. [3]
- Each trial initializes a fresh model with `model_init`, samples new hyperparameters, runs a full training loop, and reports an objective to the search backend. [3]

**Code example** [1]
```py
def compute_objective(metrics):
    return metrics["eval_loss"]

best_run = trainer.hyperparameter_search(
    hp_space=hp_space,
    compute_objective=compute_objective,
    n_trials=30,               # how many trials to run
    direction="minimize",      # or "maximize" for metrics like accuracy/F1
    backend="optuna",          # "optuna", "ray", or "wandb"
)
```

**Sources :**
- Hyperparameter search > Run the search — https://huggingface.co/docs/transformers/hpo_train#run-the-search
- Hyperparameter search > Define the search space — https://huggingface.co/docs/transformers/hpo_train#define-the-search-space
- Hyperparameter search >  — https://huggingface.co/docs/transformers/hpo_train
- Hyperparameter search > Initializing a model — https://huggingface.co/docs/transformers/hpo_train#initializing-a-model

### q010 — How do I attach a trained LoRA adapter to a base model I already loaded?

**Page cible :** peft  

**Réponse du bot** (décision : uncertain, confiance 34%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- For QLoRA (quantized base model + LoRA), `Trainer` also adjusts the mixed precision policy to match the quantization storage dtype. [1]
- Pass `hotswap=True` when loading a LoRA adapter to swap its weights into an existing adapter slot. [2]
- To fine-tune a model that doesn't fit in memory, load it in 4-bit with bitsandbytes and train LoRA adapters with PEFT. [3]

**Code example** [2]
```py
model = AutoModel.from_pretrained(...)
# Load the first adapter normally
model.load_adapter(adapter_path_1)
# Generate outputs with adapter 1
...
# Hotswap the second adapter in-place
model.load_adapter(adapter_path_2, hotswap=True, adapter_name="default")
# Generate outputs with adapter 2
```

**Sources :**
- Parameter-efficient fine-tuning > Training > Distributed training — https://huggingface.co/docs/transformers/peft#distributed-training
- Parameter-efficient fine-tuning > Hotswapping adapters — https://huggingface.co/docs/transformers/peft#hotswapping-adapters
- Apple Silicon > Quantize models — https://huggingface.co/docs/transformers/apple_silicon#quantize-models
- NVIDIA GPU > Quantize models — https://huggingface.co/docs/transformers/nvidia_gpu#quantize-models

### q011 — What do the r and lora_alpha settings control in a LoRA configuration?

**Page cible :** peft  

**Réponse du bot** (décision : uncertain, confiance 23%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Create a PEFT config, like `peft.LoraConfig` for example, and attach it to a model with `integrations.PeftAdapterMixin.add_adapter`. [1]
- For FSDP, `Trainer` updates the FSDP auto-wrap policy to correctly handle LoRA layers. [3]
- For QLoRA (quantized base model + LoRA), `Trainer` also adjusts the mixed precision policy to match the quantization storage dtype. [3]

**Code example** [1]
```py
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model.add_adapter(lora_config, adapter_name="my_adapter")
```

**Sources :**
- Parameter-efficient fine-tuning > Add an adapter — https://huggingface.co/docs/transformers/peft#add-an-adapter
- Parameter-efficient fine-tuning > Managing multiple adapters — https://huggingface.co/docs/transformers/peft#managing-multiple-adapters
- Parameter-efficient fine-tuning > Training > Distributed training — https://huggingface.co/docs/transformers/peft#distributed-training
- Parameter-efficient fine-tuning > Hotswapping adapters — https://huggingface.co/docs/transformers/peft#hotswapping-adapters

### q012 — My training crashed after 6 hours. How do I pick it up again from the last saved checkpoint?

**Page cible :** trainer_recipes  

**Réponse du bot** (décision : uncertain, confiance 39%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- With periodic checkpointing (save_strategy="steps" or "epoch"), you lose any training progress between the last saved checkpoint and an interruption. [1]
- When the trainer receives a SIGTERM signal, it saves a checkpoint at the exact point training was interrupted, so you resume with minimal loss of progress. [1]
- When SIGTERM is received, `Trainer` waits for the current training step to finish, saves a checkpoint, and stops training gracefully. [1]

**Code example** [3]
```py
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="out",
        eval_strategy="epoch",
        save_strategy="best",
        metric_for_best_model="perplexity",   # save when eval perplexity improves
        greater_is_better=False,              # lower perplexity is better
        load_best_model_at_end=True,          # load the best weights after training finishes
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
```

**Sources :**
- Trainer features > Checkpointing > JIT checkpointing — https://huggingface.co/docs/transformers/trainer_recipes#jit-checkpointing
- Save and share a model >  — https://huggingface.co/docs/transformers/quantization/save
- Trainer features > Checkpointing — https://huggingface.co/docs/transformers/trainer_recipes#checkpointing
- Choose a method >  — https://huggingface.co/docs/transformers/quantization/selecting

### q013 — Checkpoints are filling my disk, how can I keep only the most recent ones?

**Page cible :** trainer_recipes  

**Réponse du bot** (décision : uncertain, confiance 26%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- `Trainer` saves a checkpoint every `TrainingArguments.save_steps` optimizer update and keeps all of them (or the most recent `TrainingArguments.save_total_limit`). [1]
- `save_strategy="best"` keeps only the single best checkpoint according to a metric. [1]
- A new checkpoint is saved only when the tracked metric improves, which saves disk space and avoids accumulating stale checkpoints. [1]

**Code example** [2]
```py
every_n_layers=2

Forward:   input  -> [L1] -> [L2] -> [L3] -> [L4] -> [L5] -> [L6]
                      CP     keep     CP     keep     CP     keep

Backward:  output <- [L6] <- [L5] <- [L4] <- [L3] <- [L2] <- [L1]
                     keep   rerun    keep   rerun    keep   rerun
```

**Sources :**
- Trainer features > Checkpointing — https://huggingface.co/docs/transformers/trainer_recipes#checkpointing
- Gradient checkpointing > Partial checkpointing — https://huggingface.co/docs/transformers/grad_checkpointing#partial-checkpointing
- Loading models > Large models > Disk offloading — https://huggingface.co/docs/transformers/models#disk-offloading
- Apple Silicon > Limitations — https://huggingface.co/docs/transformers/apple_silicon#limitations

### q014 — Is there a trick that adds noise to embeddings during instruction fine-tuning?

**Page cible :** trainer_recipes  

**Réponse du bot** (décision : uncertain, confiance 49%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- NEFTune adds random noise to token embeddings during the forward pass. [1]
- The noise acts as regularization and can improve performance for instruction fine-tuning. [1]
- Enable NEFTune by setting `neftune_noise_alpha` in `TrainingArguments`. [1]

**Code example** [1]
```py
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="out",
        num_train_epochs=3,
        neftune_noise_alpha=5,
    ),
    train_dataset=train_dataset,
)
trainer.train()
```

**Sources :**
- Trainer features > NEFTune — https://huggingface.co/docs/transformers/trainer_recipes#neftune
- Parameter-efficient fine-tuning > Add an adapter > Fully fine-tuning specific layers — https://huggingface.co/docs/transformers/peft#fully-fine-tuning-specific-layers
- Parameter-efficient fine-tuning >  — https://huggingface.co/docs/transformers/peft
- Prompt engineering > Fine-tuning — https://huggingface.co/docs/transformers/tasks/prompting#fine-tuning

### q015 — My GPU waits for data between steps, how do I load batches with several workers?

**Page cible :** trainer_recipes  

**Réponse du bot** (décision : uncertain, confiance 30%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Data is loaded in the main process while the GPU idles, which shows up as low GPU utilization between batches. [1]
- Instead of waiting for each GPU to finish processing a batch of data, pipeline parallelism creates *micro-batches* of data. [2]
- This way, each GPU can concurrently process part of the data without waiting for the other GPU to completely finish processing a mini batch of data. [2]

**Code example** [1]
```py
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="out",
    dataloader_num_workers=4,            # spawn 4 worker subprocesses
    dataloader_persistent_workers=True,  # keep them alive between epochs
    dataloader_prefetch_factor=2,        # each worker preloads 2 batches ahead
)
```

**Sources :**
- Trainer features > Dataloader performance — https://huggingface.co/docs/transformers/trainer_recipes#dataloader-performance
- Parallelism methods > Pipeline parallelism — https://huggingface.co/docs/transformers/perf_train_gpu_many#pipeline-parallelism
- Parallelism methods > Model parallelism — https://huggingface.co/docs/transformers/perf_train_gpu_many#model-parallelism
- Parallelism methods > Data parallelism > DistributedDataParallel — https://huggingface.co/docs/transformers/perf_train_gpu_many#distributeddataparallel

### q016 — Which attention implementation relies on PyTorch's scaled dot product attention?

**Page cible :** nvidia_gpu, apple_silicon, amd_gpu  

**Réponse du bot** (décision : uncertain, confiance 20%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- For causal attention, the mask prevents the model from attending to future tokens. [1]
- At each step, the new key and value vectors are stored in the cache and appended to the past keys and values. $$ K_{\text{cache}} \leftarrow \text{concat}(K_{\text{past}}, k_t), \quad V_{\text{cache}} \leftarrow \text{concat}(V_{\text{past}}, v_t) $$ Attention is calculated independently in each layer of the model, and caching is done on a per-layer basis. [1]
- Check the Metal column in the quantization overview to find every quantization method that runs on MPS. [3]

**Sources :**
- How caching works > Attention matrices — https://huggingface.co/docs/transformers/cache_explanation#attention-matrices
- NVIDIA GPU > Next steps — https://huggingface.co/docs/transformers/nvidia_gpu#next-steps
- Apple Silicon > Next steps — https://huggingface.co/docs/transformers/apple_silicon#next-steps
- AMD GPU > Choose an attention backend — https://huggingface.co/docs/transformers/amd_gpu#choose-an-attention-backend

### q017 — How can I keep memory constant for very long generations using a sliding window?

**Page cible :** kv_cache  

**Réponse du bot** (décision : uncertain, confiance 21%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Sliding-window and linear-attention layers discard past states as they go, so there's nothing to roll back to by default. [1]
- Without it, `crop` raises a `RuntimeError` once a sliding-window layer fills its window, and immediately for linear-attention layers. [1]
- It's a no-op for full-attention layers, but for sliding-window and linear-attention layers it still discards recorded states the next forward pass won't read, which frees memory without rolling anything back. [2]

**Code example** [1]
```py
cached positions
                    ┌───┬───┬───┬───┬───┬───┐
before crop(-3)     │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │
                    └───┴───┴───┴───┴───┴───┘
                                ╰─────┬─────╯
                                 rolled back

after crop(-3)
                    ┌───┬───┬───┐
full attention      │ 0 │ 1 │ 2 │    keeps the whole prefix
                    └───┴───┴───┘
                            ┌───┐
sliding_window=2            │ 2 │    keeps sliding_window - 1
                            └───┘
```

**Sources :**
- Cache strategies > Crop a cache — https://huggingface.co/docs/transformers/kv_cache#crop-a-cache
- Cache strategies > Crop a cache — https://huggingface.co/docs/transformers/kv_cache#crop-a-cache
- Cache strategies > Fixed-size cache — https://huggingface.co/docs/transformers/kv_cache#fixed-size-cache
- How caching works > Cache storage implementation — https://huggingface.co/docs/transformers/cache_explanation#cache-storage-implementation

### q018 — When is a static cache worth using?

**Page cible :** kv_cache  

**Réponse du bot** (décision : uncertain, confiance 62%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Offloading is available for both `DynamicCache` and `StaticCache`. [1]
- You can enable it by configuring `cache_implementation="offloaded"` for the dynamic version, or `cache_implementation="offloaded_static"` for the static version, in either `GenerationConfig` or `GenerationMixin.generate`. [1]
- Additionally, you can also instantiate your own `DynamicCache` or `StaticCache` with the `offloading=True` option, and pass this cache in `generate` or your model's `forward` (for example, `past_key_values=DynamicCache(config=model.config, offloading=True)` for a dynamic cache). [1]

**Code example** [2]
```py
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

model_id = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": 0})
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Init StaticCache with big enough max-length (1024 tokens for the below example)
# You can also init a DynamicCache, if that suits you better
prompt_cache = StaticCache(config=model.config, max_cache_len=1024)

INITIAL_PROMPT = "You are a helpful assistant. "
inputs_initial_prompt = tokenizer(INITIAL_PROMPT, return_tensors="pt").to(model.device.type)
```

**Sources :**
- Cache strategies > Cache offloading — https://huggingface.co/docs/transformers/kv_cache#cache-offloading
- Cache strategies > Prefill a cache (prefix caching) — https://huggingface.co/docs/transformers/kv_cache#prefill-a-cache-prefix-caching
- Cache strategies > Fixed-size cache — https://huggingface.co/docs/transformers/kv_cache#fixed-size-cache
- How caching works > Cache storage implementation — https://huggingface.co/docs/transformers/cache_explanation#cache-storage-implementation

### q019 — How can I make generation stop with a custom rule instead of only the end-of-sequence token?

**Page cible :** llm_tutorial  

**Réponse du bot** (décision : uncertain, confiance 26%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- If you're adding a new decoding loop, you might want to preserve the input preparation present in `generate` (batch expansion, attention masks, logits processors, stopping criteria, etc.). [1]
- `GenerationMixin.generate` can also be extended with external libraries or custom code: 1. the `logits_processor` parameter accepts custom `LogitsProcessor` instances for manipulating the next token probability distribution; 2. the `stopping_criteria` parameters supports custom `StoppingCriteria` to stop text generation; 3. other custom generation methods can be loaded through the `custom_generate` flag (docs). [2]
- It removes end of sequence tokens so that the model continues generation from the final message. [3]

**Code example** [1]
```py
def custom_loop(model, input_ids, attention_mask, logits_processor, stopping_criteria, generation_config, **model_kwargs):
    next_tokens = input_ids
    while input_ids.shape[1] < stopping_criteria[0].max_length:
        logits = model(next_tokens, attention_mask=attention_mask, **model_kwargs).logits
        next_token_logits = logits_processor(input_ids, logits[:, -1, :])
        next_tokens = torch.argmax(next_token_logits, dim=-1)[:, None]
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_tokens)), dim=-1)
    return input_ids

output = model.generate(
    **inputs,
    custom_generate=custom_loop,
    max_new_tokens=10,
```

**Sources :**
- Generation strategies > Custom generation methods > Reusing `generate`'s input preparation — https://huggingface.co/docs/transformers/generation_strategies#reusing-generates-input-preparation
- Text generation > Generation configuration — https://huggingface.co/docs/transformers/llm_tutorial#generation-configuration
- Chat templates > Using `apply_chat_template` > continue_final_message — https://huggingface.co/docs/transformers/chat_templating#continuefinalmessage
- Chat templates > Using `apply_chat_template` > add_generation_prompt — https://huggingface.co/docs/transformers/chat_templating#addgenerationprompt

### q020 — How do I get several different completions for the same prompt?

**Page cible :** pipeline_tutorial, kv_cache  

**Réponse du bot** (décision : no_answer, confiance 8%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q021 — How can I get timestamps along with the transcription in a speech recognition pipeline?

**Page cible :** pipeline_tutorial  

**Réponse du bot** (décision : uncertain, confiance 34%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Pass the `return_timestamps="word"` parameter to `Pipeline` to return when each word was spoken. [1]
- Instantiate a `pipeline` for automatic speech recognition with your model, and pass your audio file to it: Tip: The transcription is decent, but it could be better! [2]
- Pass the `return_timestamps="word"` parameter to `Pipeline` to return when each word was spoken. [3]

**Code example** [1]
```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline(audio="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac", return_timestamps="word")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.',
 'chunks': [{'text': ' I', 'timestamp': (0.0, 1.1)},
  {'text': ' have', 'timestamp': (1.1, 1.44)},
  {'text': ' a', 'timestamp': (1.44, 1.62)},
  {'text': ' dream', 'timestamp': (1.62, 1.92)},
  {'text': ' that', 'timestamp': (1.92, 3.7)},
  {'text': ' one', 'timestamp': (3.7, 3.88)},
  {'text': ' day', 'timestamp': (3.88, 4.24)},
  {'text': ' this', 'timestamp': (4.24, 5.82)},
  {'text': ' nation', 'timestamp': (5.82, 6.78)},
```

**Sources :**
- Pipeline > Parameters > Task-specific parameters — https://huggingface.co/docs/transformers/pipeline_tutorial#task-specific-parameters
- Automatic speech recognition > Inference — https://huggingface.co/docs/transformers/tasks/asr#inference
- Pipeline > Parameters > Task-specific parameters — https://huggingface.co/docs/transformers/pipeline_tutorial#task-specific-parameters
- Automatic speech recognition > Load MInDS-14 dataset — https://huggingface.co/docs/transformers/tasks/asr#load-minds-14-dataset

### q022 — How do I put a pipeline behind a small web server?

**Page cible :** pipeline_webserver  

**Réponse du bot** (décision : uncertain, confiance 63%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- This means you can use `Pipeline` as an inference engine on a web server, since you can use an iterator (similar to how you would iterate over a dataset) to handle each incoming request. [2]
- Designing a web server with `Pipeline` is unique though because they're fundamentally different. [2]
- This guide shows how to work around this difference by using a web server to handle the lighter load of receiving and sending requests, and having a single thread to handle the heavier load of running `Pipeline`. [2]

**Code example** [1]
```py
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import pipeline
import asyncio

async def homepage(request):
    payload = await request.body()
    string = payload.decode("utf-8")
    response_q = asyncio.Queue()
    await request.app.model_queue.put((string, response_q))
    output = await response_q.get()
    return JSONResponse(output)

```

**Sources :**
- Web server inference > Create a server — https://huggingface.co/docs/transformers/pipeline_webserver#create-a-server
- Web server inference >  — https://huggingface.co/docs/transformers/pipeline_webserver
- Web server inference > Queuing requests — https://huggingface.co/docs/transformers/pipeline_webserver#queuing-requests
- Web server inference > Dynamic batching — https://huggingface.co/docs/transformers/pipeline_webserver#dynamic-batching

### q023 — How can I quickly build a demo interface for a pipeline?

**Page cible :** pipeline_gradio  

**Réponse du bot** (décision : uncertain, confiance 46%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Gradio, a fast and easy library for building and sharing machine learning apps, is integrated with `Pipeline` to quickly create a simple interface for inference. [1]
- Create a pipeline for your task, and then pass it to Gradio's Interface.from_pipeline function to create the interface. [1]
- This guide shows you how to quickly load chat models in Transformers from the command line, how to build and format a conversation, and how to chat using the `TextGenerationPipeline`. [2]

**Code example** [1]
```py
from transformers import pipeline
import gradio as gr

pipeline = pipeline("image-classification", model="google/vit-base-patch16-224")
gr.Interface.from_pipeline(pipeline).launch()
```

**Sources :**
- Machine learning apps >  — https://huggingface.co/docs/transformers/pipeline_gradio
- Chat basics >  — https://huggingface.co/docs/transformers/conversations
- Transformers > Design — https://huggingface.co/docs/transformers/index#design
- Quickstart >  — https://huggingface.co/docs/transformers/quicktour

### q024 — What sampling rate must my audio have before fine-tuning a speech model?

**Page cible :** tasks/asr  

**Réponse du bot** (décision : uncertain, confiance 55%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Great, now that you've fine-tuned a model, you can use it for inference! [1]
- Remember to resample the sampling rate of the audio file to match the sampling rate of the model if you need to! [1]
- The simplest way to try out your fine-tuned model for inference is to use it in a `pipeline`. [1]

**Code example** [1]
```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

**Sources :**
- Automatic speech recognition > Inference — https://huggingface.co/docs/transformers/tasks/asr#inference
- Automatic speech recognition >  — https://huggingface.co/docs/transformers/tasks/asr
- Glossary > S > sampling rate — https://huggingface.co/docs/transformers/glossary#sampling-rate
- Automatic speech recognition > Preprocess — https://huggingface.co/docs/transformers/tasks/asr#preprocess

### q025 — In question answering, how do I map the predicted span back to the original text?

**Page cible :** tasks/question_answering  

**Réponse du bot** (décision : uncertain, confiance 27%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- The next step is to load a DistilBERT tokenizer to process the `question` and `context` fields: There are a few preprocessing steps particular to question answering tasks you should be aware of: 1. [1]
- Next, map the start and end positions of the answer to the original `context` by setting `return_offset_mapping=True`. [1]
- With the mapping in hand, now you can find the start and end tokens of the answer. [1]

**Code example** [2]
```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```

**Sources :**
- Question answering > Preprocess — https://huggingface.co/docs/transformers/tasks/question_answering#preprocess
- Question answering > Inference — https://huggingface.co/docs/transformers/tasks/question_answering#inference
- Serve CLI > Reasoning — https://huggingface.co/docs/transformers/serve-cli/serving#reasoning
- Question answering > Preprocess — https://huggingface.co/docs/transformers/tasks/question_answering#preprocess

### q026 — What does double quantization change when loading a model in 4-bit?

**Page cible :** quantization/bitsandbytes  

**Réponse du bot** (décision : uncertain, confiance 52%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- This feature performs a second quantization of the already quantized weights to save an additional 0.4 bits/parameter. [1]
- For example, with nested quantization, you can finetune a Llama-13b model on a 16GB NVIDIA T4 GPU with a sequence length of 1024, a batch size of 1, and enable gradient accumulation with 4 steps. [1]
- Quantizing a model in 4-bit reduces your memory-usage by 4x, and for large models, set `device_map="auto"` to efficiently distribute the weights across all available GPUs. [2]

**Code example** [1]
```py
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", dtype="auto", quantization_config=double_quant_config)
```

**Sources :**
- Bitsandbytes > QLoRA > Nested quantization — https://huggingface.co/docs/transformers/quantization/bitsandbytes#nested-quantization
- Bitsandbytes > Quantization Examples — https://huggingface.co/docs/transformers/quantization/bitsandbytes#quantization-examples
- Overview > Quantize a model while loading — https://huggingface.co/docs/transformers/quantization/overview#quantize-a-model-while-loading
- Bitsandbytes > Quantization Examples — https://huggingface.co/docs/transformers/quantization/bitsandbytes#quantization-examples

### q027 — Which 4-bit data type is recommended for QLoRA?

**Page cible :** quantization/bitsandbytes  

**Réponse du bot** (décision : confident, confiance 75%) :

**Answer (extracted from the documentation)**

- NF4 is a 4-bit data type from the QLoRA paper, adapted for weights initialized from a normal distribution. [1]
- For inference, the `bnb_4bit_quant_type` does not have a huge impact on performance. [1]
- This section explores some of the specific features of 4-bit quantization, such as changing the compute data type, the Normal Float 4 (NF4) data type, and nested quantization. [3]

**Code example** [1]
```py
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", quantization_config=nf4_config)
```

**Sources :**
- Bitsandbytes > QLoRA > Normal Float 4 (NF4) — https://huggingface.co/docs/transformers/quantization/bitsandbytes#normal-float-4-nf4
- Bitsandbytes > QLoRA > Compute data type — https://huggingface.co/docs/transformers/quantization/bitsandbytes#compute-data-type
- Bitsandbytes > QLoRA — https://huggingface.co/docs/transformers/quantization/bitsandbytes#qlora
- Bitsandbytes > Resources — https://huggingface.co/docs/transformers/quantization/bitsandbytes#resources

### q028 — Can I offload some modules of an 8-bit model to the CPU?

**Page cible :** quantization/bitsandbytes  

**Réponse du bot** (décision : uncertain, confiance 51%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- 8-bit models can offload weights between the CPU and GPU to fit very large models into memory. [1]
- The weights dispatched to the CPU are stored in float32 and aren't converted to 8-bit. [1]
- For some models, like Jukebox, you don't need to quantize every module to 8-bit because it can actually cause instability. [2]

**Code example** [1]
```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
```

**Sources :**
- Bitsandbytes > LLM.int8 > Offloading — https://huggingface.co/docs/transformers/quantization/bitsandbytes#offloading
- Bitsandbytes > LLM.int8 > Skip module conversion — https://huggingface.co/docs/transformers/quantization/bitsandbytes#skip-module-conversion
- Bitsandbytes > LLM.int8 > Outlier threshold — https://huggingface.co/docs/transformers/quantization/bitsandbytes#outlier-threshold
- Bitsandbytes > Quantization Examples — https://huggingface.co/docs/transformers/quantization/bitsandbytes#quantization-examples

### q029 — What does the group size mean for AWQ quantization?

**Page cible :** quantization/awq  

**Réponse du bot** (décision : uncertain, confiance 59%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Activation-aware Weight Quantization (AWQ) preserves a small fraction of the weights that are important for LLM performance to compress a model to 4-bits with minimal performance degradation. [1]
- There are several libraries for quantizing models with the AWQ algorithm, such as llm-awq, autoawq or optimum-intel. [1]
- Transformers supports loading models quantized with the llm-awq and autoawq libraries. [1]

**Code example** [1]
```py
{
  "_name_or_path": "/workspace/process/huggingfaceh4_zephyr-7b-alpha/source",
  "architectures": [
    "MistralForCausalLM"
  ],
  ...
  ...
  ...
  "quantization_config": {
    "quant_method": "awq",
    "zero_point": true,
    "group_size": 128,
    "bits": 4,
    "version": "gemm"
```

**Sources :**
- AWQ >  — https://huggingface.co/docs/transformers/quantization/awq
- AWQ > Fused modules — https://huggingface.co/docs/transformers/quantization/awq#fused-modules
- Quantization concepts > When quantization happens > Post-training quantization — https://huggingface.co/docs/transformers/quantization/concept_guide#post-training-quantization
- Choose a method > Benchmarks — https://huggingface.co/docs/transformers/quantization/selecting#benchmarks

### q030 — Are there faster kernels for GPTQ models?

**Page cible :** quantization/gptq, quantization/selecting  

**Réponse du bot** (décision : confident, confiance 73%) :

**Answer (extracted from the documentation)**

- GPT-QModel is the actively maintained backend for GPTQ in Transformers. [1]
- It was originally forked from AutoGPTQ, but has since diverged with significant improvements such as faster quantization, lower memory usage, and more accurate defaults. [1]
- It is not backward compatible with legacy AutoGPTQ checkpoints, and not all kernels (Marlin) support asymmetric quantization. [1]

**Code example** [2]
```py
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
```

**Sources :**
- GPTQ > GPT-QModel — https://huggingface.co/docs/transformers/quantization/gptq#gpt-qmodel
- GPTQ >  — https://huggingface.co/docs/transformers/quantization/gptq
- GPTQ > Marlin — https://huggingface.co/docs/transformers/quantization/gptq#marlin
- GPTQ >  — https://huggingface.co/docs/transformers/quantization/gptq

### q031 — What does a name like Q4_K_M mean for a GGUF file?

**Page cible :** quantization/gguf, quantization/overview  

**Réponse du bot** (décision : uncertain, confiance 54%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- transformers serve lists each `.gguf` file in a repository as its own model. [1]
- A GGUF model is named `<repo>:<file>.gguf`, since a repository holds several quantizations and the id has to say which one to load. [1]
- A GGUF checkpoint is a single file that holds the weights, config, and tokenizer, and it has no `config.json` for `from_pretrained` to read. [2]

**Code example** [2]
```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3.5-4B-GGUF",
    gguf_file="Qwen3.5-4B-Q4_K_M.gguf",
)
```

**Sources :**
- GGUF > Serve — https://huggingface.co/docs/transformers/quantization/gguf#serve
- Overview > Load a pre-quantized checkpoint > GGUF checkpoints — https://huggingface.co/docs/transformers/quantization/overview#gguf-checkpoints
- GGUF > Load GGUF models — https://huggingface.co/docs/transformers/quantization/gguf#load-gguf-models
- llama.cpp > Transformers integration — https://huggingface.co/docs/transformers/community_integrations/llama_cpp#transformers-integration

### q032 — How do I run Hub models with llama.cpp?

**Page cible :** community_integrations/llama_cpp  

**Réponse du bot** (décision : uncertain, confiance 60%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- llama.cpp is a C/C++ inference engine for deploying large language models locally. [1]
- It's lightweight and doesn't require Python, CUDA, or other heavy server infrastructure. llama.cpp uses the GGUF file format. [1]
- `transformers` is the pivot across frameworks: if a model definition is supported, it will be compatible with the majority of training frameworks (Axolotl, Unsloth, DeepSpeed, FSDP, PyTorch-Lightning, ...), inference engines (vLLM, SGLang, TGI, ...), and adjacent modeling libraries (llama.cpp, mlx, ...) which leverage the model definition from `transformers`. [3]

**Code example** [1]
```py
python3 convert_hf_to_gguf.py ./models/openai/gpt-oss-20b \
  --outfile gpt-oss-20b.gguf \
```

**Sources :**
- llama.cpp >  — https://huggingface.co/docs/transformers/community_integrations/llama_cpp
- llama.cpp > Transformers integration — https://huggingface.co/docs/transformers/community_integrations/llama_cpp#transformers-integration
- Transformers >  — https://huggingface.co/docs/transformers/index
- Save and share a model > Methods that can't save — https://huggingface.co/docs/transformers/quantization/save#methods-that-cant-save

### q033 — How is symmetric quantization different from asymmetric?

**Page cible :** quantization/concept_guide  

**Réponse du bot** (décision : uncertain, confiance 44%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Asymmetric* quantization uses the formulas above. [1]
- Symmetric* quantization assumes the range is `[-a, a]` and fixes `Z` at 0, so only the scale is stored. [1]
- GPT-QModel provides asymmetric quantization which can potentially lower quantization errors compared to symmetric quantization. [2]

**Code example** [1]
```py
┌─────┐  quantize   ┌────┐ dequantize   ┌───────┐
│ 0.5 │ ──────────▶ │ 11 │ ───────────▶ │ 0.494 │
└─────┘             └────┘              └───────┘
```

**Sources :**
- Quantization concepts > Map values to fewer bits — https://huggingface.co/docs/transformers/quantization/concept_guide#map-values-to-fewer-bits
- GPTQ > GPT-QModel — https://huggingface.co/docs/transformers/quantization/gptq#gpt-qmodel
- Quantization concepts >  — https://huggingface.co/docs/transformers/quantization/concept_guide
- Quantization concepts > When quantization happens > Post-training quantization — https://huggingface.co/docs/transformers/quantization/concept_guide#post-training-quantization

### q034 — Can I use my AMD graphics card with the library?

**Page cible :** amd_gpu  

**Réponse du bot** (décision : uncertain, confiance 58%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- AMD Instinct (MI200, MI300, MI350 series) and Radeon GPUs run PyTorch through ROCm, AMD's GPU software stack. [1]
- The ROCm build of PyTorch uses HIP, AMD’s CUDA-like programming interface, to expose AMD GPUs through the `torch.cuda` API. [1]
- Otherwise, loads amd/aiter-flash-attn from the Hub. | | `"flash_attention_3"` | kernels | Loads amd/aiter-flash-attn, a prebuilt FlashAttention kernel from AMD's AITER library. [3]

**Code example** [3]
```py
You do not have `flash_attn` installed, using `kernels-community/aiter-flash-attn` from the `kernels` library instead!
```

**Sources :**
- AMD GPU >  — https://huggingface.co/docs/transformers/amd_gpu
- Sharing > Model card — https://huggingface.co/docs/transformers/model_sharing#model-card
- AMD GPU > Choose an attention backend — https://huggingface.co/docs/transformers/amd_gpu#choose-an-attention-backend
- AMD GPU > Next steps — https://huggingface.co/docs/transformers/amd_gpu#next-steps

### q035 — How can I restrict which GPUs my script can see?

**Page cible :** troubleshooting, nvidia_gpu, debugging  

**Réponse du bot** (décision : uncertain, confiance 39%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- PyTorch sees only GPUs 0 and 2, which are mapped to `cuda:0` and `cuda:1`. [1]
- The script prints `OK` if both GPUs communicate and allocate memory successfully. [2]
- For example, to select GPUs 0 and 2 out of four: PyTorch sees only GPUs 0 and 2, which are mapped to `cuda:0` and `cuda:1`. [3]

**Code example** [2]
```py
wget https://raw.githubusercontent.com/huggingface/transformers/main/scripts/distributed/torch-distributed-gpu-test.py
python -m torch.distributed.run --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

**Sources :**
- AMD GPU > Select and order GPUs — https://huggingface.co/docs/transformers/amd_gpu#select-and-order-gpus
- Debugging > Communication — https://huggingface.co/docs/transformers/debugging#communication
- NVIDIA GPU > Select and order GPUs — https://huggingface.co/docs/transformers/nvidia_gpu#select-and-order-gpus
- Bitsandbytes > Hardware Compatibility > NVIDIA GPUs (CUDA) — https://huggingface.co/docs/transformers/quantization/bitsandbytes#nvidia-gpus-cuda

### q036 — My multi-GPU job hangs, how do I investigate communication problems?

**Page cible :** debugging  

**Réponse du bot** (décision : no_answer, confiance 13%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q037 — How can I move optimizer states to the CPU with DeepSpeed?

**Page cible :** deepspeed  

**Réponse du bot** (décision : uncertain, confiance 63%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- DeepSpeed ZeRO (Zero Redundancy Optimizer) eliminates memory redundancy across distributed training by sharding optimizer states, gradients, and parameters across GPUs. [1]
- DeepSpeed also supports offloading to CPU or NVMe memory for further savings. [1]
- If you're offloading the optimizer, set `zero_force_ds_cpu_optimizer` to `false` to use DeepSpeed's CPU Adam optimizer. [2]

**Code example** [2]
```py
{
        "zero_force_ds_cpu_optimizer": false
    }
```

**Sources :**
- DeepSpeed ZeRO >  — https://huggingface.co/docs/transformers/deepspeed
- DeepSpeed ZeRO > ZeRO stages — https://huggingface.co/docs/transformers/deepspeed#zero-stages
- Debugging > DeepSpeed > Process killed at startup — https://huggingface.co/docs/transformers/debugging#process-killed-at-startup
- DeepSpeed ZeRO > ZeRO stages — https://huggingface.co/docs/transformers/deepspeed#zero-stages

### q038 — Should I choose fp16 or bf16 for mixed precision?

**Page cible :** mixed_precision_training  

**Réponse du bot** (décision : uncertain, confiance 52%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Mixed precision uses fp16 or bf16 for the compute-intensive forward and backward passes, while keeping an fp32 copy of the weights for the optimizer update. [1]
- Set `TrainingArguments.bf16` or `TrainingArguments.fp16` to `True` to enable mixed precision training. [1]
- If your model is numerically stable in bf16/fp16, you can skip mixed precision entirely and load and train directly in bf16/fp16. [2]

**Code example** [1]
```py
┌─────────────────────────────────────────────────────┐
│           MIXED PRECISION TRAINING LOOP             │
│                                                     │
│  fp32 master weights ──cast──▶ fp16/bf16            │
│         ▲                          │                │
│         │                    FORWARD (autocast)     │
│         │                    matmuls in fp16/bf16   │
│         │                    reductions stay fp32   │
│         │                          │ loss           │
│         │                    LOSS SCALE ×S  ──fp16  │
│         │                          │                │
│         │                    BACKWARD               │
│         │                    grads in fp16/bf16     │
│         │                          │                │
```

**Sources :**
- Mixed precision training >  — https://huggingface.co/docs/transformers/mixed_precision_training
- Mixed precision training >  — https://huggingface.co/docs/transformers/mixed_precision_training
- Mixed precision training > tf32 — https://huggingface.co/docs/transformers/mixed_precision_training#tf32
- NVIDIA GPU > Train with Trainer — https://huggingface.co/docs/transformers/nvidia_gpu#train-with-trainer

### q039 — How do I train a masked language model?

**Page cible :** tasks/language_modeling, training  

**Réponse du bot** (décision : confident, confiance 76%) :

**Answer (extracted from the documentation)**

- Pretraining methods involve a self-supervised objective, which can be reading the text and trying to predict the next word (see causal language modeling) or masking some words and trying to predict them (see masked language modeling). [1]
- On the other hand, BEiT is a vision model pretrained on a masked image modeling task which masks some of the image patches and requires the model to predict the masked patches (similar to the masked language modeling objective). [1]
- Oftentimes, encoder models are pretrained using techniques like masked language modeling, which masks parts of the input sequence and forces the model to create more meaningful representations. [2]

**Code example** [3]
```py
>>> padded_sequences["attention_mask"]
[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

**Sources :**
- Glossary > P > pretrained model — https://huggingface.co/docs/transformers/glossary#pretrained-model
- Glossary > E > encoder models — https://huggingface.co/docs/transformers/glossary#encoder-models
- Glossary > A > attention mask — https://huggingface.co/docs/transformers/glossary#attention-mask
- Glossary > M > masked language modeling (MLM) — https://huggingface.co/docs/transformers/glossary#masked-language-modeling-mlm

### q040 — How can I send my training metrics to Weights & Biases?

**Page cible :** hpo_train, trainer_recipes  

**Réponse du bot** (décision : uncertain, confiance 22%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- If you don't define a `hp_space` function, the default search covers `learning_rate`, `num_train_epochs`, and `per_device_train_batch_size`. [1]
- Weights & Biases is an experiment tracking platform with built-in hyperparameter search. [1]
- Pass the training arguments to `Trainer` along with the model, dataset, tokenizer, data collator, and `compute_metrics` function. [3]

**Code example** [2]
```py
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="out",
        eval_strategy="epoch",
        save_strategy="best",
        metric_for_best_model="perplexity",   # save when eval perplexity improves
        greater_is_better=False,              # lower perplexity is better
        load_best_model_at_end=True,          # load the best weights after training finishes
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
```

**Sources :**
- Hyperparameter search > Define the search space — https://huggingface.co/docs/transformers/hpo_train#define-the-search-space
- Trainer features > Checkpointing — https://huggingface.co/docs/transformers/trainer_recipes#checkpointing
- Token classification > Train — https://huggingface.co/docs/transformers/tasks/token_classification#train
- Text classification > Train — https://huggingface.co/docs/transformers/tasks/sequence_classification#train

### q041 — What do I need to do in the terminal before pushing a model to the Hub?

**Page cible :** model_sharing, custom_models, quantization/save  

**Réponse du bot** (décision : confident, confiance 67%) :

**Answer (extracted from the documentation)**

- The `utils.PushToHubMixin` provides functionality for pushing a model or tokenizer to the Hub. [1]
- Call `utils.PushToHubMixin.push_to_hub` directly on a model to upload it to the Hub. [1]
- It creates a repository under your namespace with the model name specified in `utils.PushToHubMixin.push_to_hub`. [1]

**Code example** [1]
```py
model.push_to_hub("my-awesome-model")
```

**Sources :**
- Sharing > Uploading a model > PushToHubMixin — https://huggingface.co/docs/transformers/model_sharing#pushtohubmixin
- Sharing > Uploading a model > Trainer — https://huggingface.co/docs/transformers/model_sharing#trainer
- Save and share a model > Save the model — https://huggingface.co/docs/transformers/quantization/save#save-the-model
- Sharing > Uploading a model — https://huggingface.co/docs/transformers/model_sharing#uploading-a-model

### q042 — In which file format are weights saved by default?

**Page cible :** models  

**Réponse du bot** (décision : uncertain, confiance 22%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Some methods can't save their quantized weights, and `PreTrainedModel.save_pretrained` raises a `ValueError` for them. | Method | Why | |---|---| | GGUF | Weights load from a GGUF file. [1]
- Save in GGUF format with llama.cpp tools instead. | | NVFP4 | Saving isn't supported. [1]
- `PreTrainedModel.save_pretrained` saves only the adapter weights, so it works with any method. [1]

**Code example** [2]
```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Qwen3.5-4B-GGUF",
    gguf_file="Qwen3.5-4B-Q4_K_M.gguf",
)
```

**Sources :**
- Save and share a model > Methods that can't save — https://huggingface.co/docs/transformers/quantization/save#methods-that-cant-save
- Overview > Load a pre-quantized checkpoint > GGUF checkpoints — https://huggingface.co/docs/transformers/quantization/overview#gguf-checkpoints
- FSDP2 > Configure FSDP — https://huggingface.co/docs/transformers/fsdp#configure-fsdp
- GGUF >  — https://huggingface.co/docs/transformers/quantization/gguf

### q043 — How can I load a model that ships its own code on the Hub?

**Page cible :** custom_models, models, generation_strategies  

**Réponse du bot** (décision : confident, confiance 70%) :

**Answer (extracted from the documentation)**

- Set `trust_remote_code=True` in `PreTrainedModel.from_pretrained` to load a custom model. [1]
- As an extra layer of security, load a custom model from a specific revision to avoid loading model code that may have changed. [1]
- Loading is already sequential when a `device_map` offloads to disk or when quantizing on the fly, so this only affects otherwise parallel loads. | | `DISABLE_SAFETENSORS_CONVERSION` | boolean | Stops Transformers from asking the Hub conversion bot for a safetensors version of a repository that ships only PyTorch `.bin` weights. [2]

**Code example** [1]
```py
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("sgugger/custom-resnet50d", trust_remote_code=True)
```

**Sources :**
- Loading models > Custom models — https://huggingface.co/docs/transformers/models#custom-models
- Environment variables > Model loading — https://huggingface.co/docs/transformers/reference/environment_variables#model-loading
- Customizing models > Upload — https://huggingface.co/docs/transformers/custom_models#upload
- Loading models >  — https://huggingface.co/docs/transformers/models

### q044 — Can I export a model to run on a phone with ExecuTorch?

**Page cible :** serialization  

**Réponse du bot** (décision : confident, confiance 77%) :

**Answer (extracted from the documentation)**

- ExecuTorch runs PyTorch models on mobile and edge devices. [1]
- It exports a model into a graph of standardized operators, compiles the graph into an ExecuTorch program, and executes it on the target device. [1]
- Export a Transformers model to ExecuTorch with the CLI tool. [1]

**Code example** [1]
```py
optimum-cli export executorch \
    --model "Qwen/Qwen3-8B" \
    --task "text-generation" \
    --recipe "xnnpack" \
    --use_custom_sdpa \
    --use_custom_kv_cache \
    --qlinear 8da4w \
    --qembedding 8w \
    --output_dir="hf_smollm2"
```

**Sources :**
- Exporting to production > ExecuTorch — https://huggingface.co/docs/transformers/serialization#executorch
- Exporting to production > ONNX > optimum-cli — https://huggingface.co/docs/transformers/serialization#optimum-cli
- Exporting to production >  — https://huggingface.co/docs/transformers/serialization
- Exporting to production > ONNX > optimum.onnxruntime — https://huggingface.co/docs/transformers/serialization#optimumonnxruntime

### q045 — How do I make the assistant continue a message it already started?

**Page cible :** chat_templating  

**Réponse du bot** (décision : no_answer, confiance 14%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q046 — How does a model return a structured tool call?

**Page cible :** chat_extras  

**Réponse du bot** (décision : uncertain, confiance 57%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- After the model returns a tool call, execute the function locally, then send the result back in a follow-up request to get the model's final answer. [1]
- Warning: Although the key in the assistant message is called `tool_calls`, in most cases, models only emit a single tool call at a time. [2]
- Some older models emit multiple tool calls at the same time, but this is a significantly more complex process, as you need to handle multiple tool responses at once and disambiguate them, often using tool call IDs. [2]

**Code example** [1]
```py
# Model returns a tool call
messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=messages,
    tools=tools,
)
assistant_message = response.choices[0].message

# Execute the tool locally
tool_call = assistant_message.tool_calls[0]
result = {"temperature": 22, "condition": "sunny"}  # your actual function call here

# Send the tool result back
```

**Sources :**
- Serve CLI > Tool calling > Multi-turn tool calling — https://huggingface.co/docs/transformers/serve-cli/serving#multi-turn-tool-calling
- Tool use > Tool-calling Example — https://huggingface.co/docs/transformers/chat_extras#tool-calling-example
- Tool use > Tool-calling Example — https://huggingface.co/docs/transformers/chat_extras#tool-calling-example
- Tool use > Tool-calling Example — https://huggingface.co/docs/transformers/chat_extras#tool-calling-example

### q047 — Can I pass documents to a chat model so it answers from them?

**Page cible :** chat_extras  

**Réponse du bot** (décision : uncertain, confiance 26%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- The `PreTrainedTokenizerBase.apply_chat_template#continue_final_message` parameter controls whether the final message in the chat should be continued or not instead of starting a new one. [1]
- It removes end of sequence tokens so that the model continues generation from the final message. [1]
- Always consult a specific instruction-tuned model's documentation to learn more about the format of their chat template so that you can structure your few-shot prompts accordingly. [3]

**Code example** [1]
```py
chat = [
    {"role": "user", "content": "Can you format the answer in JSON?"},
    {"role": "assistant", "content": '{"name": "'},
]

formatted_chat = tokenizer.apply_chat_template(chat, tokenize=True, return_dict=True, continue_final_message=True)
model.generate(**formatted_chat)
```

**Sources :**
- Chat templates > Using `apply_chat_template` > continue_final_message — https://huggingface.co/docs/transformers/chat_templating#continuefinalmessage
- Pipeline > Tasks — https://huggingface.co/docs/transformers/pipeline_tutorial#tasks
- Prompt engineering > Techniques > Few-shot prompting — https://huggingface.co/docs/transformers/tasks/prompting#few-shot-prompting
- Serve CLI > v1/systemone > Decision configuration — https://huggingface.co/docs/transformers/serve-cli/serving#decision-configuration

### q048 — Which generation setting discourages the model from repeating itself?

**Page cible :** llm_tutorial  

**Réponse du bot** (décision : uncertain, confiance 28%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Check this guide for more information. | | `repetition_penalty` | `float` | Set it to `>1.0` if you're seeing the model repeat itself often. [1]
- However, it breaks down when generating longer sequences because it begins to repeat itself. [2]
- The model updates its weights based on how incorrect its predictions were, and the process is repeated to optimize model performance. [3]

**Code example** [2]
```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.float16).to(device)
# explicitly set to default length because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a suite of tools and services for building, deploying, and maintaining natural language processing'
```

**Sources :**
- Text generation > Common Options — https://huggingface.co/docs/transformers/llm_tutorial#common-options
- Generation strategies > Basic decoding methods > Greedy search — https://huggingface.co/docs/transformers/generation_strategies#greedy-search
- Glossary > S > supervised learning — https://huggingface.co/docs/transformers/glossary#supervised-learning
- Chat basics > TextGenerationPipeline — https://huggingface.co/docs/transformers/conversations#textgenerationpipeline

### q049 — Which parameter controls how random the sampling is?

**Page cible :** chat_extras, serve-cli/serving, assisted_decoding, generation_strategies  

**Réponse du bot** (décision : uncertain, confiance 24%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Validated tokens enter the final output and rejected tokens trigger standard sampling. [2]
- Speculative decoding supports greedy search and sampling but not batched inputs. [2]
- Set `do_sample=True` to resample if token validation fails. [2]

**Code example** [1]
```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", dtype="auto")
assistant_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, assistant_model=assistant_model, do_sample=True, temperature=0.5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that is dedicated to creating a better world through technology.'
```

**Sources :**
- Assisted decoding > Speculative decoding — https://huggingface.co/docs/transformers/assisted_decoding#speculative-decoding
- Assisted decoding > Speculative decoding — https://huggingface.co/docs/transformers/assisted_decoding#speculative-decoding
- Text generation > Common Options — https://huggingface.co/docs/transformers/llm_tutorial#common-options
- Causal language modeling > Inference — https://huggingface.co/docs/transformers/tasks/language_modeling#inference

### q050 — What is prompt lookup decoding?

**Page cible :** assisted_decoding  

**Réponse du bot** (décision : confident, confiance 79%) :

**Answer (extracted from the documentation)**

- Prompt lookup decoding doesn't need an assistant model. [1]
- It finds overlapping n-grams in the prompt to propose candidate tokens. [1]
- This requires candidate logits from the assistant model and is not supported with prompt lookup decoding. [2]

**Code example** [1]
```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-1.7B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-1.7B", dtype="auto")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt")

outputs = model.generate(**inputs, prompt_lookup_num_tokens=5)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a platform for developers to build and deploy machine learning models. It offers a variety of tools'
```

**Sources :**
- Assisted decoding > Prompt lookup decoding — https://huggingface.co/docs/transformers/assisted_decoding#prompt-lookup-decoding
- Assisted decoding > Universal assisted decoding > Static ensemble verification — https://huggingface.co/docs/transformers/assisted_decoding#static-ensemble-verification
- Optimizing LLMs for Speed and Memory > 3. Architectural Innovations > 3.2 The key-value cache > 3.2.1 Multi-round conversation — https://huggingface.co/docs/transformers/llm_tutorial_optimization#321-multi-round-conversation
- Prompt engineering > Best practices — https://huggingface.co/docs/transformers/tasks/prompting#best-practices

### q051 — How do I tokenize a prompt without adding special tokens?

**Page cible :** chat_templating, assisted_decoding  

**Réponse du bot** (décision : uncertain, confiance 61%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Not all models require generation prompts, and some models, like Llama, don't have any special tokens before the `assistant` response. [1]
- First, let's format a chat without `add_generation_prompt`: Now, let's format the same chat with `add_generation_prompt=True`: When `add_generation_prompt=True`, `<|im_start|>assistant` is added at the end to indicate the start of an `assistant` message. [2]
- Chat templates should already include all the necessary special tokens, and adding additional special tokens is often incorrect or duplicated, hurting model performance. [3]

**Code example** [2]
```py
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
tokenized_chat
```

**Sources :**
- Chat templates > Using `apply_chat_template` > add_generation_prompt — https://huggingface.co/docs/transformers/chat_templating#addgenerationprompt
- Chat templates > Using `apply_chat_template` > add_generation_prompt — https://huggingface.co/docs/transformers/chat_templating#addgenerationprompt
- Chat templates > Using `apply_chat_template` — https://huggingface.co/docs/transformers/chat_templating#using-applychattemplate
- Tokenization algorithms > Byte pair encoding (BPE) > Byte-level BPE — https://huggingface.co/docs/transformers/tokenizer_summary#byte-level-bpe

### q052 — My decoder-only model has no pad token and batching fails, what should I do?

**Page cible :** llm_tutorial, llm_tutorial_optimization  

**Réponse du bot** (décision : uncertain, confiance 39%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- But LLMs aren't trained to continue generation from padding tokens, which means the `PreTrainedTokenizer.padding_side` parameter needs to be set to the left of the input. [1]
- Right padding is worse for state-space and hybrid models because their recurrent state keeps updating over the trailing pad tokens, so the cache no longer matches the end of the prompt. [1]
- Speculative decoding uses a smaller assistant model to draft candidate tokens. [3]

**Code example** [1]
```py
model_inputs = tokenizer(
    ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
).to(model.device)
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 33333333333'
```

**Sources :**
- Text generation > Pitfalls > Padding side — https://huggingface.co/docs/transformers/llm_tutorial#padding-side
- Causal language modeling > Preprocess — https://huggingface.co/docs/transformers/tasks/language_modeling#preprocess
- Assisted decoding > Speculative decoding — https://huggingface.co/docs/transformers/assisted_decoding#speculative-decoding
- Fine-tuning > Tokenization — https://huggingface.co/docs/transformers/training#tokenization

### q053 — What is the masked language modeling objective?

**Page cible :** glossary  

**Réponse du bot** (décision : confident, confiance 88%) :

**Answer (extracted from the documentation)**

- Pretraining methods involve a self-supervised objective, which can be reading the text and trying to predict the next word (see causal language modeling) or masking some words and trying to predict them (see masked language modeling). [1]
- On the other hand, BEiT is a vision model pretrained on a masked image modeling task which masks some of the image patches and requires the model to predict the masked patches (similar to the masked language modeling objective). [1]
- One example of self-supervised learning is masked language modeling, where a model is passed sentences with a proportion of its tokens removed and learns to predict the missing tokens. [2]

**Sources :**
- Glossary > P > pretrained model — https://huggingface.co/docs/transformers/glossary#pretrained-model
- Glossary > S > self-supervised learning — https://huggingface.co/docs/transformers/glossary#self-supervised-learning
- Glossary > E > encoder models — https://huggingface.co/docs/transformers/glossary#encoder-models
- Glossary > A > attention mask — https://huggingface.co/docs/transformers/glossary#attention-mask

### q054 — What does an image processor do?

**Page cible :** tasks/image_classification, glossary  

**Réponse du bot** (décision : confident, confiance 75%) :

**Answer (extracted from the documentation)**

- The next step is to load a ViT image processor to process the image into a tensor: Apply some image transformations to the images to make the model more robust against overfitting. [1]
- Here you'll use torchvision's `transforms` module, but you can also use any image library you like. [1]
- The pixel values have a shape of [`batch_size`, `num_channels`, `height`, `width`], and are generated from an image processor. [3]

**Code example** [1]
```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "google/vit-base-patch16-224-in21k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

**Sources :**
- Image classification > Preprocess — https://huggingface.co/docs/transformers/tasks/image_classification#preprocess
- Image classification > Inference — https://huggingface.co/docs/transformers/tasks/image_classification#inference
- Glossary > P > pixel values — https://huggingface.co/docs/transformers/glossary#pixel-values
- Troubleshoot > ImportError — https://huggingface.co/docs/transformers/troubleshooting#importerror

### q055 — How do I start FSDP training with accelerate launch?

**Page cible :** fsdp, accelerate  

**Réponse du bot** (décision : uncertain, confiance 55%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Configure FSDP training with either an Accelerate config file or an FSDP config file passed to `fsdp_config`. [1]
- Run accelerate launch with a `Trainer`-based script, and Accelerate reads the config file to set up training. [2]
- The `TrainingArguments#fsdp_config` and `TrainingArguments#deepspeed` args are unnecessary because the Accelerate config file covers the same settings. [2]

**Code example** [1]
```py
accelerate launch train.py
```

**Sources :**
- FSDP2 > Configure FSDP — https://huggingface.co/docs/transformers/fsdp#configure-fsdp
- Accelerate > Accelerate config file — https://huggingface.co/docs/transformers/accelerate#accelerate-config-file
- NVIDIA GPU > Train with Trainer > Distributed training — https://huggingface.co/docs/transformers/nvidia_gpu#distributed-training
- Accelerate >  — https://huggingface.co/docs/transformers/accelerate

### q056 — Can I speed up CPU training with bf16 mixed precision?

**Page cible :** cpu  

**Réponse du bot** (décision : confident, confiance 66%) :

**Answer (extracted from the documentation)**

- CPUs with bf16 support, such as recent Intel and AMD processors, can use bf16 mixed precision training with PyTorch's AMP, which reduces memory usage and speeds up training. [1]
- If your model is numerically stable in bf16/fp16, you can skip mixed precision entirely and load and train directly in bf16/fp16. [2]
- Set `TrainingArguments.bf16` or `TrainingArguments.fp16` to `True` to enable mixed precision training. [3]

**Code example** [3]
```py
┌─────────────────────────────────────────────────────┐
│           MIXED PRECISION TRAINING LOOP             │
│                                                     │
│  fp32 master weights ──cast──▶ fp16/bf16            │
│         ▲                          │                │
│         │                    FORWARD (autocast)     │
│         │                    matmuls in fp16/bf16   │
│         │                    reductions stay fp32   │
│         │                          │ loss           │
│         │                    LOSS SCALE ×S  ──fp16  │
│         │                          │                │
│         │                    BACKWARD               │
│         │                    grads in fp16/bf16     │
│         │                          │                │
```

**Sources :**
- CPU > Train with Trainer — https://huggingface.co/docs/transformers/cpu#train-with-trainer
- Mixed precision training >  — https://huggingface.co/docs/transformers/mixed_precision_training
- Mixed precision training >  — https://huggingface.co/docs/transformers/mixed_precision_training
- Mixed precision training > tf32 — https://huggingface.co/docs/transformers/mixed_precision_training#tf32

### q057 — Comment limiter le nombre de threads CPU utilisés par PyTorch ?

**Page cible :** cpu  

**Réponse du bot** (décision : uncertain, confiance 34%) :

**Réponse (extraite de la documentation)** — je ne suis pas certain que ces extraits répondent exactement à votre question.

- PyTorch is not async aware, so computation will block the main thread from running. [2]
- For this reason, it's better to run PyTorch on its own separate thread or process. [2]
- CPUs with bf16 support, such as recent Intel and AMD processors, can use bf16 mixed precision training with PyTorch's AMP, which reduces memory usage and speeds up training. [3]

**Exemple de code** [3]
```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
    use_cpu=True,
)
```

**Sources :**
- CPU > Next steps — https://huggingface.co/docs/transformers/cpu#next-steps
- Web server inference > Block the main thread — https://huggingface.co/docs/transformers/pipeline_webserver#block-the-main-thread
- CPU > Train with Trainer — https://huggingface.co/docs/transformers/cpu#train-with-trainer
- Translation > Inference — https://huggingface.co/docs/transformers/tasks/translation#inference

### q058 — Quelle est la différence entre WordPiece et Unigram ?

**Page cible :** tokenizer_summary  

**Réponse du bot** (décision : uncertain, confiance 27%) :

**Réponse (extraite de la documentation)** — je ne suis pas certain que ces extraits répondent exactement à votre question.

- Chapter 6 of the LLM course teaches you how to train a tokenizer from scratch and explains the differences between the BPE, Unigram, and WordPiece algorithms. [1]
- Transformers support three subword tokenization algorithms: Byte pair encoding (BPE), Unigram, and WordPiece. [2]
- WordPiece is the tokenization algorithm for BERT-family models like DistilBERT and Electra. [3]

**Sources :**
- Tokenization algorithms > Resources — https://huggingface.co/docs/transformers/tokenizer_summary#resources
- Tokenization algorithms >  — https://huggingface.co/docs/transformers/tokenizer_summary
- Tokenization algorithms > WordPiece — https://huggingface.co/docs/transformers/tokenizer_summary#wordpiece
- Tokenization algorithms > SentencePiece — https://huggingface.co/docs/transformers/tokenizer_summary#sentencepiece

### q059 — Comment reprendre un entraînement après un plantage ?

**Page cible :** trainer_recipes  

**Réponse du bot** (décision : no_answer, confiance 5%) :

Je n'ai pas trouvé de réponse fiable à cette question dans la documentation de Hugging Face Transformers, et je préfère ne pas deviner. Je peux transmettre votre question à un expert humain.


### q060 — Comment ne garder que les derniers checkpoints sur le disque ?

**Page cible :** trainer_recipes  

**Réponse du bot** (décision : no_answer, confiance 6%) :

Je n'ai pas trouvé de réponse fiable à cette question dans la documentation de Hugging Face Transformers, et je préfère ne pas deviner. Je peux transmettre votre question à un expert humain.


### q061 — Comment charger un adaptateur LoRA sur un modèle de base ?

**Page cible :** peft  

**Réponse du bot** (décision : no_answer, confiance 9%) :

Je n'ai pas trouvé de réponse fiable à cette question dans la documentation de Hugging Face Transformers, et je préfère ne pas deviner. Je peux transmettre votre question à un expert humain.


### q062 — Quelle variable d'environnement permet de choisir les GPU visibles ?

**Page cible :** troubleshooting, nvidia_gpu, debugging  

**Réponse du bot** (décision : no_answer, confiance 12%) :

Je n'ai pas trouvé de réponse fiable à cette question dans la documentation de Hugging Face Transformers, et je préfère ne pas deviner. Je peux transmettre votre question à un expert humain.



## Hors périmètre (le bot doit refuser) (17 questions)

### q063 — How do I make a sourdough starter?

**Réponse du bot** (décision : no_answer, confiance 0%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q064 — What is the exchange rate between the Tunisian dinar and the euro today?

**Réponse du bot** (décision : no_answer, confiance 7%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q065 — Explain how castling works in chess.

**Réponse du bot** (décision : no_answer, confiance 10%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q066 — How do I fix a flat bicycle tire?

**Réponse du bot** (décision : no_answer, confiance 4%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q067 — How do I set up a PostgreSQL read replica?

**Réponse du bot** (décision : no_answer, confiance 5%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q068 — How do I train a YOLOv8 detector on my own images with Ultralytics?

**Réponse du bot** (décision : no_answer, confiance 5%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q069 — How do I write a Dockerfile for a Flask application?

**Réponse du bot** (décision : no_answer, confiance 10%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q070 — How do I use the OpenAI fine-tuning API on my dataset?

**Réponse du bot** (décision : uncertain, confiance 47%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- The Responses API is OpenAI's latest API endpoint for generation. [2]
- OpenAI recommends using the Responses API over the Chat Completions API for new projects. [2]
- The `v1/responses` API supports text, image, audio, and video requests through the `curl` command and OpenAI client. [2]

**Code example** [1]
```py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<random_string>")

response = client.responses.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    input=[
        {"type": "input_text", "text": "What's in this image?"},
        {
            "type": "input_image",
            "image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg",
        },
    ],
    max_output_tokens=256,
```

**Sources :**
- Serve CLI > v1/responses > Image-based responses — https://huggingface.co/docs/transformers/serve-cli/serving#image-based-responses
- Serve CLI > v1/responses — https://huggingface.co/docs/transformers/serve-cli/serving#v1responses
- Serve CLI > v1/responses > Audio-based responses — https://huggingface.co/docs/transformers/serve-cli/serving#audio-based-responses
- Serve CLI > v1/chat/completions > Video-based completions — https://huggingface.co/docs/transformers/serve-cli/serving#video-based-completions

### q071 — How do I tune a model with scikit-learn's GridSearchCV?

**Réponse du bot** (décision : no_answer, confiance 15%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q072 — What are the opening hours of the Hugging Face office in Paris?

**Réponse du bot** (décision : no_answer, confiance 9%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q073 — Can you translate this paragraph into Arabic for me?

**Réponse du bot** (décision : no_answer, confiance 14%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q074 — How do I write a React component with hooks?

**Réponse du bot** (décision : no_answer, confiance 7%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q075 — Which Stable Diffusion sampler is best for portraits?

**Réponse du bot** (décision : no_answer, confiance 5%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q076 — How do I cancel my Hugging Face Enterprise contract?

**Réponse du bot** (décision : no_answer, confiance 15%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q077 — How do I connect to a MySQL database from Java?

**Réponse du bot** (décision : no_answer, confiance 12%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q078 — What is the tallest mountain in Africa?

**Réponse du bot** (décision : no_answer, confiance 5%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q079 — How can I make my Excel macros run faster?

**Réponse du bot** (décision : no_answer, confiance 11%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.



## Ambiguë / complexe (le bot doit escalader) (14 questions)

### q080 — Nothing happens when I press run.

**Réponse du bot** (décision : no_answer, confiance 10%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q081 — The results look weird.

**Réponse du bot** (décision : no_answer, confiance 9%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q082 — How do I fix the bug?

**Réponse du bot** (décision : uncertain, confiance 56%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- It ensures you have the most up-to-date changes in Transformers and it's useful for experimenting with the latest features or fixing a bug that hasn't been officially released in the stable version yet. [1]
- If you encounter any problems, please open a GitHub Issue so we can fix it as soon as possible. [1]
- Try to include as much information describing the bug as possible to help us better figure out what's wrong and how we can fix it. [2]

**Sources :**
- Installation > Python > Source install — https://huggingface.co/docs/transformers/installation#source-install
- Troubleshoot >  — https://huggingface.co/docs/transformers/troubleshooting
- Cache strategies > Fixed-size cache — https://huggingface.co/docs/transformers/kv_cache#fixed-size-cache
- Sharing > Repository features — https://huggingface.co/docs/transformers/model_sharing#repository-features

### q083 — Which GPU should I buy?

**Réponse du bot** (décision : uncertain, confiance 40%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Quantization lets a larger model fit in GPU memory. [2]
- To fine-tune a model that doesn't fit in GPU memory, load it in 4-bit with bitsandbytes and train LoRA adapters with PEFT. [2]
- Multi-GPU training uses the NCCL collective backend. [3]

**Code example** [1]
```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", dtype=torch.bfloat16, device_map="auto")
prompt = """Let's go through this step-by-step:
1. You start with 15 muffins.
2. You eat 2 muffins, leaving you with 13 muffins.
3. You give 5 muffins to your neighbor, leaving you with 8 muffins.
4. Your partner buys 6 more muffins, bringing the total number of muffins to 14.
5. Your partner eats 2 muffins, leaving you with 12 muffins.
If you eat 6 muffins, how many are left?"""

outputs = pipeline(prompt, max_new_tokens=20, do_sample=True, top_k=10)
for output in outputs:
```

**Sources :**
- Prompt engineering > Techniques > Chain-of-thought — https://huggingface.co/docs/transformers/tasks/prompting#chain-of-thought
- NVIDIA GPU > Quantize models — https://huggingface.co/docs/transformers/nvidia_gpu#quantize-models
- NVIDIA GPU > Train with Trainer > Distributed training — https://huggingface.co/docs/transformers/nvidia_gpu#distributed-training
- NVIDIA GPU > Run inference — https://huggingface.co/docs/transformers/nvidia_gpu#run-inference

### q084 — Is my dataset good enough?

**Réponse du bot** (décision : uncertain, confiance 20%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Crafting a good prompt alone, also known as zero-shot prompting, may not be enough to get the results you want. [1]
- In all of these scenarios, ensure that you have a large enough domain-specific dataset to train your model with, have enough time and resources, and the cost of fine-tuning is worth it. [2]
- {'summary': 'Existing law authorizes state agencies to enter into contracts for the acquisition of goods or services upon approval by the Department of General Services. [3]

**Sources :**
- Prompt engineering > Techniques — https://huggingface.co/docs/transformers/tasks/prompting#techniques
- Prompt engineering > Fine-tuning — https://huggingface.co/docs/transformers/tasks/prompting#fine-tuning
- Summarization > Load BillSum dataset — https://huggingface.co/docs/transformers/tasks/summarization#load-billsum-dataset
- Pipeline > Parameters > Task-specific parameters — https://huggingface.co/docs/transformers/pipeline_tutorial#task-specific-parameters

### q085 — What is the best batch size for me?

**Réponse du bot** (décision : uncertain, confiance 59%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Hyperparameters like learning rate, batch size, and number of epochs significantly affect training results. [1]
- `Trainer.hyperparameter_search` finds the best combination by running multiple trials, each with a different set of values, and returning the best one. [1]
- After all trials complete, the best hyperparameters are returned in a `trainer.utils.BestRun`. [1]

**Code example** [2]
```py
training_args = TrainingArguments(
    output_dir="qwen3-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"every_n_layers": 4},
    bf16=True,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

**Sources :**
- Hyperparameter search >  — https://huggingface.co/docs/transformers/hpo_train
- Fine-tuning > Training configuration — https://huggingface.co/docs/transformers/training#training-configuration
- Image classification > Train — https://huggingface.co/docs/transformers/tasks/image_classification#train
- Pipeline > Parameters > Batch inference — https://huggingface.co/docs/transformers/pipeline_tutorial#batch-inference

### q086 — Why is it so much slower than yesterday?

**Réponse du bot** (décision : no_answer, confiance 11%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q087 — My boss wants a full production pipeline: fine-tune, quantize, deploy on-premise and in the cloud, add monitoring, guarantee 99.9% uptime and pass a s

**Réponse du bot** (décision : uncertain, confiance 20%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Deploy the same model to cloud providers or run it on mobile and edge devices. [1]
- You don't need to rewrite the model from scratch for each deployment environment. [1]
- Freely deploy across any inference ecosystem. [1]

**Sources :**
- Exporting to production >  — https://huggingface.co/docs/transformers/serialization
- Fine-tuning >  — https://huggingface.co/docs/transformers/training
- Choose a method > Start from your goal — https://huggingface.co/docs/transformers/quantization/selecting#start-from-your-goal
- Fine-tuning >  — https://huggingface.co/docs/transformers/training

### q088 — Traceback (most recent call last):

**Réponse du bot** (décision : uncertain, confiance 32%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- A data collator assembles individual dataset samples into a batch for the model. [1]
- It can also dynamically pad samples to the longest sequence in *each batch*, which is more efficient than padding to a global maximum length. [1]
- DataCollatorMixin when you need custom padding logic, multiple paired inputs per sample, or a batch structure the tokenizer can't produce on its own. [1]

**Code example** [3]
```py
>>> import torch

>>> from dataclasses import dataclass, field
>>> from typing import Any, Dict, List, Optional, Union

>>> @dataclass
... class DataCollatorCTCWithPadding:
...     processor: AutoProcessor
...     padding: Union[bool, str] = "longest"

...     def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
...         # split inputs and labels since they have to be of different lengths and need
...         # different padding methods
...         input_features = [{"input_values": feature["input_values"][0]} for feature in features]
```

**Sources :**
- Data collators >  — https://huggingface.co/docs/transformers/data_collators
- Data collators > DataCollatorWithPadding — https://huggingface.co/docs/transformers/data_collators#datacollatorwithpadding
- Automatic speech recognition > Preprocess — https://huggingface.co/docs/transformers/tasks/asr#preprocess
- Data collators > DataCollatorMixin — https://huggingface.co/docs/transformers/data_collators#datacollatormixin

### q089 — Please debug my notebook (about 400 cells) and tell me why the loss is so high.

**Réponse du bot** (décision : no_answer, confiance 11%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.


### q090 — We need a legal confirmation that fine-tuned models can be used commercially. Who can sign off on this?

**Réponse du bot** (décision : uncertain, confiance 20%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- While prompting is a powerful way to work with LLMs, there are scenarios where a fine-tuned model or even fine-tuning a model works better. [1]
- Here are some examples scenarios where a fine-tuned model makes sense. [1]
- In all of these scenarios, ensure that you have a large enough domain-specific dataset to train your model with, have enough time and resources, and the cost of fine-tuning is worth it. [1]

**Code example** [3]
```py
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
```

**Sources :**
- Prompt engineering > Fine-tuning — https://huggingface.co/docs/transformers/tasks/prompting#fine-tuning
- Fine-tuning >  — https://huggingface.co/docs/transformers/training
- Fine-tuning > Loading a model — https://huggingface.co/docs/transformers/training#loading-a-model
- Image classification >  — https://huggingface.co/docs/transformers/tasks/image_classification

### q091 — Training on two clusters gives different perplexities. Maybe the distributed sampler is buggy? Can you confirm, find the root cause and tell me if I s

**Réponse du bot** (décision : no_answer, confiance 8%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.

Note: la question contient plusieurs demandes distinctes. A human expert may be better placed to help.


### q092 — Should I learn Transformers or classical machine learning first?

**Réponse du bot** (décision : uncertain, confiance 60%) :

**Answer (extracted from the documentation)** — I am not certain these excerpts answer your question exactly.

- Tip: Read our Philosophy to learn more about Transformers' design principles. [1]
- Transformers is designed for developers and machine learning engineers and researchers. [1]
- A subfield of machine learning which focuses on learning meaningful representations of raw data. [2]

**Sources :**
- Transformers > Design — https://huggingface.co/docs/transformers/index#design
- Glossary > R > representation learning — https://huggingface.co/docs/transformers/glossary#representation-learning
- Glossary > S > self-supervised learning — https://huggingface.co/docs/transformers/glossary#self-supervised-learning
- Glossary >  — https://huggingface.co/docs/transformers/glossary

### q093 — Tell me everything about artificial intelligence.

**Réponse du bot** (décision : no_answer, confiance 7%) :

I couldn't find a reliable answer to this in the Hugging Face Transformers documentation, so I would rather not guess. I can pass your question to a human expert.



## Demande d'un humain (3 questions)

### q094 — I would like to speak with a support agent.

**Réponse du bot** (décision : human_requested, confiance 100%) :

Understood — I'm creating a ticket for a human expert, with a summary of our conversation.


### q095 — Please connect me to a human expert.

**Réponse du bot** (décision : human_requested, confiance 100%) :

Understood — I'm creating a ticket for a human expert, with a summary of our conversation.


### q096 — J'aimerais être mis en relation avec un conseiller humain.

**Réponse du bot** (décision : human_requested, confiance 100%) :

C'est noté — je crée un ticket pour un expert humain, avec le résumé de notre conversation.

