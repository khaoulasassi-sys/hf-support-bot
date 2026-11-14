#!/usr/bin/env python3
"""Génère data/eval/questions.jsonl (source de vérité de l'évaluation).

Catégories
  in_scope     : question couverte par la doc. gold_docs = pages acceptables ; keywords = au moins un
                 de ces termes doit figurer dans le passage pour qu'il soit jugé « pertinent ».
  out_of_scope : hors périmètre (aucune réponse dans la doc) → le bot doit s'abstenir.
  ambiguous    : vague / multi-demandes / nécessite un diagnostic humain → le bot doit escalader.
  human        : demande explicite d'un humain → décision attendue : human_requested.
Découpage dev/test déterministe : 1 question sur 3 (par catégorie) va dans « dev » (calibration,
entraînement du re-ranker) ; le reste est le jeu de TEST, jamais utilisé pour régler quoi que ce soit.
"""
import json
from pathlib import Path

IN = [  # (question, gold_docs, keywords, lang)
 ("How do I install Transformers inside a virtual environment?", ["installation"], ["venv", "virtual environment"], "en"),
 ("Can I run Transformers without an internet connection?", ["installation", "troubleshooting"], ["HF_HUB_OFFLINE", "offline"], "en"),
 ("Where are downloaded models cached and how can I change that folder?", ["installation", "reference/environment_variables"], ["HF_HOME", "cache"], "en"),
 ("I want the latest unreleased features, how do I install from the main branch?", ["installation"], ["pip install git+", "source"], "en"),
 ("I keep getting CUDA out of memory errors when training, what can I try?", ["troubleshooting"], ["batch size", "gradient accumulation", "out of memory"], "en"),
 ("My script is running on a machine behind a firewall and can't download the weights. What should I do?", ["troubleshooting", "installation"], ["firewall", "offline", "HF_HUB_OFFLINE"], "en"),
 ("What does the 'CUDA error: device-side assert triggered' message mean and how do I debug it?", ["troubleshooting"], ["CUDA_LAUNCH_BLOCKING", "device-side assert"], "en"),
 ("The model output is wrong when my batch contains padded sequences", ["troubleshooting"], ["attention_mask", "attention mask"], "en"),
 ("How do I run a text generation pipeline on my GPU?", ["pipeline_tutorial"], ["device=0", "device_map", "device"], "en"),
 ("What is the best way to feed a very large dataset into a pipeline?", ["pipeline_tutorial"], ["KeyDataset", "generator", "large datasets"], "en"),
 ("How do I process several inputs at once with a pipeline?", ["pipeline_tutorial"], ["batch_size"], "en"),
 ("My model is too big for one GPU, how can I load it across several devices?", ["models", "nvidia_gpu"], ["device_map", "Big Model Inference"], "en"),
 ("How do I choose the precision a model is loaded in, so it isn't always float32?", ["models"], ["dtype", "torch_dtype"], "en"),
 ("How do I load a model in 4-bit using bitsandbytes?", ["quantization/bitsandbytes", "quantization/overview"], ["load_in_4bit", "BitsAndBytesConfig"], "en"),
 ("What does the outlier threshold do in LLM.int8()?", ["quantization/bitsandbytes"], ["llm_int8_threshold", "outlier"], "en"),
 ("Can I save a quantized model and upload it to the Hub?", ["quantization/save", "quantization/overview"], ["save_pretrained", "push_to_hub"], "en"),
 ("Which quantization method fits my hardware and memory budget?", ["quantization/selecting", "quantization/overview"], ["memory", "choose"], "en"),
 ("How can I load a GGUF checkpoint in Transformers?", ["quantization/gguf"], ["gguf_file"], "en"),
 ("How do I fine-tune with LoRA adapters through PEFT?", ["peft"], ["LoraConfig", "add_adapter", "adapter"], "en"),
 ("How can I trade compute for memory by recomputing activations during training?", ["grad_checkpointing"], ["gradient_checkpointing", "gradient checkpointing"], "en"),
 ("How do I simulate a bigger batch size with gradient accumulation in the Trainer?", ["grad_accumulation"], ["gradient_accumulation_steps"], "en"),
 ("How do I speed up training on Ampere GPUs with TF32?", ["mixed_precision_training"], ["tf32"], "en"),
 ("How do I stop training automatically when the evaluation metric stops improving?", ["trainer_callbacks"], ["EarlyStoppingCallback", "early_stopping"], "en"),
 ("How do I write my own callback to run code at the end of each epoch?", ["trainer_callbacks"], ["TrainerCallback", "on_epoch_end"], "en"),
 ("Can I replace the default loss of the Trainer with my own?", ["trainer_recipes"], ["compute_loss_func", "compute_loss"], "en"),
 ("Evaluation uses too much memory with a large vocabulary, how can I reduce it?", ["trainer_recipes"], ["eval_accumulation_steps", "preprocess_logits_for_metrics"], "en"),
 ("How can I run an evaluation before training begins?", ["trainer_recipes"], ["eval_on_start"], "en"),
 ("How do I reduce padding waste by batching similar-length samples together?", ["trainer_recipes"], ["group_by_length"], "en"),
 ("How do I search for the best hyperparameters with the Trainer?", ["hpo_train"], ["hyperparameter_search"], "en"),
 ("How do I launch multi-GPU training with DistributedDataParallel?", ["ddp", "perf_train_gpu_many"], ["torchrun", "DistributedDataParallel"], "en"),
 ("What sharding strategies does FSDP offer?", ["fsdp"], ["FULL_SHARD", "SHARD_GRAD_OP", "sharding"], "en"),
 ("How do I configure the ZeRO stages in DeepSpeed?", ["deepspeed"], ["zero_optimization", "stage"], "en"),
 ("My training loss suddenly becomes NaN. How can I find where it happens?", ["debugging"], ["DebugUnderflowOverflow", "NaN", "inf"], "en"),
 ("How can I display tokens one by one as the model generates them?", ["generation_features"], ["TextStreamer", "streamer"], "en"),
 ("What is the difference between greedy decoding, sampling and beam search?", ["generation_strategies"], ["num_beams", "do_sample"], "en"),
 ("My generated text always stops after about 20 tokens", ["llm_tutorial"], ["max_new_tokens"], "en"),
 ("Why are batched generations of poor quality, is it a padding problem?", ["llm_tutorial"], ["padding_side", "left"], "en"),
 ("How do I use a small draft model to make generation faster?", ["assisted_decoding"], ["assistant_model", "speculative"], "en"),
 ("Can the key-value cache be quantized to save memory?", ["kv_cache"], ["QuantizedCache", "quantized"], "en"),
 ("How do I move the KV cache to CPU memory during long generations?", ["kv_cache"], ["offload", "Offloaded"], "en"),
 ("How do I format a conversation for a chat model?", ["chat_templating", "conversations"], ["apply_chat_template"], "en"),
 ("What is add_generation_prompt for in chat templates?", ["chat_templating"], ["add_generation_prompt"], "en"),
 ("How do I let a model call external tools or functions?", ["chat_extras"], ["tools", "tool"], "en"),
 ("Can I expose a model through an OpenAI-compatible HTTP server?", ["serve-cli/serving"], ["transformers serve", "chat/completions"], "en"),
 ("How do I pad and truncate a batch of texts with a tokenizer?", ["fast_tokenizers"], ["padding", "truncation"], "en"),
 ("What is byte-pair encoding?", ["tokenizer_summary"], ["BPE", "byte-pair", "Byte-Pair"], "en"),
 ("What does the attention mask do?", ["glossary"], ["attention mask"], "en"),
 ("How do I share my fine-tuned model on the Hugging Face Hub?", ["model_sharing"], ["push_to_hub"], "en"),
 ("How can I export a model to ONNX?", ["serialization"], ["onnx", "optimum-cli"], "en"),
 ("How do I speed up inference using torch.compile?", ["perf_torch_compile"], ["torch.compile"], "en"),
 ("How do I fine-tune a model for sentiment classification on IMDb?", ["tasks/sequence_classification"], ["IMDb", "imdb"], "en"),
 ("How do I train a model for extractive question answering?", ["tasks/question_answering"], ["SQuAD", "squad"], "en"),
 ("Can I run models on a Mac GPU?", ["apple_silicon"], ["mps", "MPS"], "en"),
 ("How do I handle dynamic padding of batches during training?", ["data_collators"], ["DataCollatorWithPadding"], "en"),
 ("How do I set up speech recognition fine-tuning?", ["tasks/asr"], ["MInDS", "CTC", "Wav2Vec2"], "en"),
 ("How do I choose a different optimizer or learning rate scheduler for the Trainer?", ["optimizers"], ["optim", "lr_scheduler_type", "scheduler"], "en"),
 ("Which environment variable controls how verbose the logging is?", ["reference/environment_variables"], ["TRANSFORMERS_VERBOSITY"], "en"),
 ("How can Flash Attention reduce memory use for long inputs?", ["llm_tutorial_optimization", "nvidia_gpu"], ["flash", "Flash"], "en"),
 # --- questions en français (mesurent la limite cross-lingue ; rapportées à part)
 ("Comment installer Transformers dans un environnement virtuel ?", ["installation"], ["venv", "virtual environment"], "fr"),
 ("Comment charger un modèle en 4 bits avec bitsandbytes ?", ["quantization/bitsandbytes", "quantization/overview"], ["load_in_4bit", "BitsAndBytesConfig"], "fr"),
 ("Comment activer l'accumulation de gradient dans le Trainer ?", ["grad_accumulation"], ["gradient_accumulation_steps"], "fr"),
 ("Mon entraînement plante avec une erreur CUDA out of memory, que faire ?", ["troubleshooting"], ["batch size", "gradient accumulation", "out of memory"], "fr"),
 ("Comment appliquer un chat template à une conversation ?", ["chat_templating", "conversations"], ["apply_chat_template"], "fr"),
 ("Comment publier mon modèle sur le Hub Hugging Face ?", ["model_sharing"], ["push_to_hub"], "fr"),
]
OOS = [
 "How do I bake a chocolate cake?",
 "What will the weather be like in Tunis tomorrow?",
 "Who won the 2022 football World Cup?",
 "How do I reset my Netflix password?",
 "Can you recommend a good restaurant in Paris?",
 "Write me a poem about the sea.",
 "How do I configure an nginx reverse proxy with SSL certificates?",
 "How do I deploy a TensorFlow Serving model on Kubernetes?",
 "How do I build an agent with LangChain and Pinecone?",
 "How much does a Hugging Face Pro subscription cost?",
 "How do I fine-tune a model with LLaMA-Factory?",
 "How do I install PyTorch Lightning and use its LightningModule?",
 "How do I generate images with the OpenAI API?",
 "What is the capital of Australia?",
 "How do I deploy my web app on AWS Lambda?",
 "Can I get a refund for my Inference Endpoints invoice?",
 "How do I write a pivot table formula in Excel?",
]
AMB = [
 "It doesn't work, please help.",
 "My model is slow.",
 "Error: something went wrong when I ran the script.",
 "How can I make it better?",
 "Which model is the best?",
 "What is the best learning rate?",
 "Why does my accuracy not improve?",
 "I need to fine-tune, quantize, serve and monitor a 70B model across three clouds on a $100 budget, with GDPR compliance. What exactly should I do? Which GPUs? Which vendor? How long will it take? Who will maintain it?",
 "Traceback (most recent call last):\n  File \"train.py\", line 52, in <module>\n    trainer.train()\n  File \"/usr/lib/python3/site-packages/torch/nn/modules/module.py\", line 1501, in _call_impl\n    return forward_call(*args, **kwargs)\nRuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!\nWhy is this happening, how do I fix it, and is my dataloader also wrong?",
 "Can you review my 300-line training script and tell me everything wrong with it?",
 "My company needs a contractual SLA for this library. Who should we contact?",
 "I think there's a bug: the evaluation loss differs between two identical runs on two machines. Can you confirm it's a bug in the Trainer? What is the root cause? Should I open an issue? Is it the dataloader?",
 "Is PyTorch or JAX better for my project?",
 "Can you teach me everything about transformers?",
]
HUMAN = [
 "Can I speak to a human please?",
 "I'd like to talk to a real person about my problem.",
 "Je voudrais parler à un conseiller humain.",
]


def main():
    rows, n = [], 0
    def add(cat, q, **kw):
        nonlocal n
        n += 1
        rows.append({"id": f"q{n:03d}", "category": cat, "question": q, **kw})
    for i, (q, gd, kw, lang) in enumerate(IN):
        add("in_scope", q, gold_docs=gd, keywords=kw, lang=lang)
    for q in OOS:
        add("out_of_scope", q, lang="en")
    for q in AMB:
        add("ambiguous", q, lang="en")
    for q in HUMAN:
        add("human", q, lang="fr" if "conseiller" in q else "en")
    # découpage dev/test : 1 sur 3 par catégorie (les questions FR sont réparties de la même façon)
    counters: dict[str, int] = {}
    for r in rows:
        k = (r["category"], r["lang"])
        counters[k] = counters.get(k, 0) + 1
        r["split"] = "dev" if counters[k] % 3 == 1 else "test"
    out = Path(__file__).resolve().parents[1] / "data" / "eval" / "questions.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    from collections import Counter
    print(len(rows), "questions", Counter((r["category"], r["split"]) for r in rows))


if __name__ == "__main__":
    main()
