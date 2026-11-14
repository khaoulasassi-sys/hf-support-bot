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
 ("I'm on a MacBook with an M2 chip and an operator isn't implemented for the Metal backend. Is there an environment variable to avoid the crash?", ["apple_silicon"], ["PYTORCH_ENABLE_MPS_FALLBACK"], "en"),
 ("How can I limit how many CPU threads PyTorch uses when I run inference?", ["cpu"], ["OMP_NUM_THREADS"], "en"),
 ("What is the difference between WordPiece and Unigram tokenization?", ["tokenizer_summary"], ["WordPiece", "Unigram"], "en"),
 ("Which subword algorithm does SentencePiece rely on?", ["tokenizer_summary", "fast_tokenizers"], ["SentencePiece"], "en"),
 ("Why do ALiBi and rotary position embeddings matter for long prompts?", ["llm_tutorial_optimization"], ["ALiBi", "rotary"], "en"),
 ("Is there an optimizer that saves memory by projecting gradients into a low-rank space?", ["optimizers"], ["GaLore"], "en"),
 ("Can I train without having to tune a learning rate schedule?", ["optimizers"], ["schedule_free", "schedule-free"], "en"),
 ("What is LOMO and why would I use it for a big model?", ["optimizers"], ["LOMO"], "en"),
 ("Can the Trainer run its hyperparameter search with Optuna?", ["hpo_train"], ["Optuna", "optuna"], "en"),
 ("How do I attach a trained LoRA adapter to a base model I already loaded?", ["peft"], ["load_adapter"], "en"),
 ("What do the r and lora_alpha settings control in a LoRA configuration?", ["peft"], ["lora_alpha"], "en"),
 ("My training crashed after 6 hours. How do I pick it up again from the last saved checkpoint?", ["trainer_recipes"], ["resume_from_checkpoint"], "en"),
 ("Checkpoints are filling my disk, how can I keep only the most recent ones?", ["trainer_recipes"], ["save_total_limit"], "en"),
 ("Is there a trick that adds noise to embeddings during instruction fine-tuning?", ["trainer_recipes"], ["NEFTune", "neftune"], "en"),
 ("My GPU waits for data between steps, how do I load batches with several workers?", ["trainer_recipes"], ["dataloader_num_workers"], "en"),
 ("Which attention implementation relies on PyTorch's scaled dot product attention?", ["nvidia_gpu", "apple_silicon", "amd_gpu"], ["SDPA", "sdpa"], "en"),
 ("How can I keep memory constant for very long generations using a sliding window?", ["kv_cache"], ["sliding window", "Sliding"], "en"),
 ("When is a static cache worth using?", ["kv_cache"], ["StaticCache", "static cache"], "en"),
 ("How can I make generation stop with a custom rule instead of only the end-of-sequence token?", ["llm_tutorial"], ["StoppingCriteria"], "en"),
 ("How do I get several different completions for the same prompt?", ["pipeline_tutorial", "kv_cache"], ["num_return_sequences"], "en"),
 ("How can I get timestamps along with the transcription in a speech recognition pipeline?", ["pipeline_tutorial"], ["return_timestamps"], "en"),
 ("How do I put a pipeline behind a small web server?", ["pipeline_webserver"], ["uvicorn", "Starlette", "starlette"], "en"),
 ("How can I quickly build a demo interface for a pipeline?", ["pipeline_gradio"], ["gradio", "Gradio"], "en"),
 ("What sampling rate must my audio have before fine-tuning a speech model?", ["tasks/asr"], ["sampling_rate"], "en"),
 ("In question answering, how do I map the predicted span back to the original text?", ["tasks/question_answering"], ["offset_mapping"], "en"),
 ("What does double quantization change when loading a model in 4-bit?", ["quantization/bitsandbytes"], ["bnb_4bit_use_double_quant"], "en"),
 ("Which 4-bit data type is recommended for QLoRA?", ["quantization/bitsandbytes"], ["nf4", "NF4"], "en"),
 ("Can I offload some modules of an 8-bit model to the CPU?", ["quantization/bitsandbytes"], ["llm_int8_enable_fp32_cpu_offload"], "en"),
 ("What does the group size mean for AWQ quantization?", ["quantization/awq"], ["group_size"], "en"),
 ("Are there faster kernels for GPTQ models?", ["quantization/gptq", "quantization/selecting"], ["Marlin"], "en"),
 ("What does a name like Q4_K_M mean for a GGUF file?", ["quantization/gguf", "quantization/overview"], ["Q4_K_M"], "en"),
 ("How do I run Hub models with llama.cpp?", ["community_integrations/llama_cpp"], ["llama.cpp"], "en"),
 ("How is symmetric quantization different from asymmetric?", ["quantization/concept_guide"], ["symmetric"], "en"),
 ("Can I use my AMD graphics card with the library?", ["amd_gpu"], ["ROCm"], "en"),
 ("How can I restrict which GPUs my script can see?", ["troubleshooting", "nvidia_gpu", "debugging"], ["CUDA_VISIBLE_DEVICES"], "en"),
 ("My multi-GPU job hangs, how do I investigate communication problems?", ["debugging"], ["NCCL"], "en"),
 ("How can I move optimizer states to the CPU with DeepSpeed?", ["deepspeed"], ["offload_optimizer"], "en"),
 ("Should I choose fp16 or bf16 for mixed precision?", ["mixed_precision_training"], ["bf16"], "en"),
 ("How do I train a masked language model?", ["tasks/language_modeling", "training"], ["DataCollatorForLanguageModeling"], "en"),
 ("How can I send my training metrics to Weights & Biases?", ["hpo_train", "trainer_recipes"], ["wandb"], "en"),
 ("What do I need to do in the terminal before pushing a model to the Hub?", ["model_sharing", "custom_models", "quantization/save"], ["hf auth login", "login"], "en"),
 ("In which file format are weights saved by default?", ["models"], ["safetensors"], "en"),
 ("How can I load a model that ships its own code on the Hub?", ["custom_models", "models", "generation_strategies"], ["trust_remote_code"], "en"),
 ("Can I export a model to run on a phone with ExecuTorch?", ["serialization"], ["ExecuTorch"], "en"),
 ("How do I make the assistant continue a message it already started?", ["chat_templating"], ["continue_final_message"], "en"),
 ("How does a model return a structured tool call?", ["chat_extras"], ["tool_call"], "en"),
 ("Can I pass documents to a chat model so it answers from them?", ["chat_extras"], ["documents", "document"], "en"),
 ("Which generation setting discourages the model from repeating itself?", ["llm_tutorial"], ["repetition_penalty"], "en"),
 ("Which parameter controls how random the sampling is?", ["chat_extras", "serve-cli/serving", "assisted_decoding", "generation_strategies"], ["temperature"], "en"),
 ("What is prompt lookup decoding?", ["assisted_decoding"], ["prompt lookup", "Prompt lookup"], "en"),
 ("How do I tokenize a prompt without adding special tokens?", ["chat_templating", "assisted_decoding"], ["add_special_tokens"], "en"),
 ("My decoder-only model has no pad token and batching fails, what should I do?", ["llm_tutorial", "llm_tutorial_optimization"], ["pad_token"], "en"),
 ("What is the masked language modeling objective?", ["glossary"], ["masked language", "MLM", "mlm"], "en"),
 ("What does an image processor do?", ["tasks/image_classification", "glossary"], ["image processor"], "en"),
 ("How do I start FSDP training with accelerate launch?", ["fsdp", "accelerate"], ["accelerate launch"], "en"),
 ("Can I speed up CPU training with bf16 mixed precision?", ["cpu"], ["bf16"], "en"),
 # --- français (limite cross-lingue ; rapportées à part)
 ("Comment limiter le nombre de threads CPU utilisés par PyTorch ?", ["cpu"], ["OMP_NUM_THREADS"], "fr"),
 ("Quelle est la différence entre WordPiece et Unigram ?", ["tokenizer_summary"], ["WordPiece", "Unigram"], "fr"),
 ("Comment reprendre un entraînement après un plantage ?", ["trainer_recipes"], ["resume_from_checkpoint"], "fr"),
 ("Comment ne garder que les derniers checkpoints sur le disque ?", ["trainer_recipes"], ["save_total_limit"], "fr"),
 ("Comment charger un adaptateur LoRA sur un modèle de base ?", ["peft"], ["load_adapter"], "fr"),
 ("Quelle variable d'environnement permet de choisir les GPU visibles ?", ["troubleshooting", "nvidia_gpu", "debugging"], ["CUDA_VISIBLE_DEVICES"], "fr"),
]
OOS = [
 "How do I make a sourdough starter?",
 "What is the exchange rate between the Tunisian dinar and the euro today?",
 "Explain how castling works in chess.",
 "How do I fix a flat bicycle tire?",
 "How do I set up a PostgreSQL read replica?",
 "How do I train a YOLOv8 detector on my own images with Ultralytics?",
 "How do I write a Dockerfile for a Flask application?",
 "How do I use the OpenAI fine-tuning API on my dataset?",
 "How do I tune a model with scikit-learn's GridSearchCV?",
 "What are the opening hours of the Hugging Face office in Paris?",
 "Can you translate this paragraph into Arabic for me?",
 "How do I write a React component with hooks?",
 "Which Stable Diffusion sampler is best for portraits?",
 "How do I cancel my Hugging Face Enterprise contract?",
 "How do I connect to a MySQL database from Java?",
 "What is the tallest mountain in Africa?",
 "How can I make my Excel macros run faster?",
]
AMB = [
 "Nothing happens when I press run.",
 "The results look weird.",
 "How do I fix the bug?",
 "Which GPU should I buy?",
 "Is my dataset good enough?",
 "What is the best batch size for me?",
 "Why is it so much slower than yesterday?",
 "My boss wants a full production pipeline: fine-tune, quantize, deploy on-premise and in the cloud, add monitoring, guarantee 99.9% uptime and pass a security audit. Give me the complete plan with timelines and costs. Which vendor? Which team size? What about backups?",
 "Traceback (most recent call last):\n  File \"run.py\", line 88, in <module>\n    batch = collate(samples)\n  File \"collate.py\", line 14, in collate\n    return torch.tensor(rows)\nValueError: expected sequence of length 128 at dim 1 (got 97)\nWhy does this happen, how do I fix it, and is my tokenizer config also wrong?",
 "Please debug my notebook (about 400 cells) and tell me why the loss is so high.",
 "We need a legal confirmation that fine-tuned models can be used commercially. Who can sign off on this?",
 "Training on two clusters gives different perplexities. Maybe the distributed sampler is buggy? Can you confirm, find the root cause and tell me if I should file an issue? Or is it my seed?",
 "Should I learn Transformers or classical machine learning first?",
 "Tell me everything about artificial intelligence.",
]
HUMAN = [
 "I would like to speak with a support agent.",
 "Please connect me to a human expert.",
 "J'aimerais être mis en relation avec un conseiller humain.",
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
