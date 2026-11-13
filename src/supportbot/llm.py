"""Génération de la réponse à partir des passages retrouvés.

Trois backends derrière le même contrat `generate(question, passages, history, low_confidence)` :
* ExtractiveLLM      : aucun LLM — assemble une réponse à partir des phrases les plus pertinentes
                       des passages (fonctionne hors-ligne ; sert de repli si le LLM est en panne) ;
* OpenAICompatibleLLM: API /chat/completions (Ollama local, OpenAI, Groq, vLLM, LM Studio…) ;
* HFLocalLLM         : modèle `transformers` local (ex. Qwen2.5-1.5B-Instruct).
La liste des sources est ajoutée par le pipeline (déterministe), pas par le LLM : on ne laisse
jamais le modèle inventer une URL.
"""
from __future__ import annotations

import os
import re

import requests

from .textutil import split_sentences, tokenize
from .types import Candidate

SYSTEM_PROMPT = """You are the technical support assistant for the Hugging Face Transformers library.
Answer ONLY with the numbered documentation excerpts provided by the user message.

Rules:
1. Cite the excerpts you rely on with their number, e.g. [1], [2].
2. If the excerpts do not contain the answer, say so plainly. NEVER invent function names, arguments, flags or commands.
3. Reply in the same language as the user's question (code and identifiers stay in English).
4. Be pedagogical: start with a one-sentence answer, then numbered steps if it is a procedure, then a minimal code example taken from the excerpts if one exists. Briefly explain jargon.
5. Keep it short. Do not repeat the excerpts verbatim and do not write a sources list (it is added automatically)."""

LOW_CONF_NOTE = ("IMPORTANT: retrieval confidence is LOW. Begin by saying you are not fully sure and that the "
                 "excerpts may only partly answer the question; do not guess beyond them.")


class LLMError(RuntimeError):
    pass


def build_user_message(question: str, passages: list[Candidate], history: list[tuple[str, str]],
                       low_confidence: bool) -> str:
    ctx = "\n\n".join(f"[{i}] ({p.chunk.doc_title} > {p.chunk.section or 'intro'})\n{p.chunk.text}"
                      for i, p in enumerate(passages, 1))
    hist = ""
    if history:
        hist = "Conversation so far:\n" + "\n".join(f"User: {u}\nAssistant: {a[:300]}" for u, a in history) + "\n\n"
    note = f"{LOW_CONF_NOTE}\n\n" if low_confidence else ""
    return f"Documentation excerpts:\n{ctx}\n\n{hist}{note}Question: {question}"


# ----------------------------------------------------------------------------------------------
_NAV = re.compile(r"^(see|read|learn more|check out|refer to|for more|take a look|head over|next,|now that)\b", re.I)


def _clean_prose(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)              # le code est traité à part
    text = re.sub(r"\*\*Option: [^*]+\*\*", " ", text)                # marqueurs d'onglets de la doc
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)             # lien markdown → texte
    text = re.sub(r"\[`~?([^`\]]+)`\]", r"`\1`", text)               # [`Trainer`] → `Trainer`
    text = text.replace("**", "")
    return re.sub(r"^[#>*\-\s]+", "", text, flags=re.M)


class ExtractiveLLM:
    """Réponse « sans LLM » : les 3 phrases les plus proches de la question + le bloc de code le plus pertinent."""
    name = "extractive"

    def generate(self, question: str, passages: list[Candidate], history=None, low_confidence=False) -> str:
        fr = looks_french(question)
        q_toks = set(tokenize(question))
        best: list[tuple[float, int, int, str]] = []   # (score, rang du passage, position, phrase)
        for rank, p in enumerate(passages[:3]):
            for pos, sent in enumerate(split_sentences(_clean_prose(p.chunk.text))):
                n_words = len(sent.split())
                if n_words < 6 or n_words > 55 or _NAV.match(sent) or sent.endswith(":"):
                    continue
                overlap = len(q_toks & set(tokenize(sent))) / (len(q_toks) or 1)
                best.append((overlap - 0.08 * rank, rank, pos, sent))
        best.sort(key=lambda t: t[0], reverse=True)
        chosen = sorted(best[:3], key=lambda t: (t[1], t[2]))
        head = "**Réponse (extraite de la documentation)**" if fr else "**Answer (extracted from the documentation)**"
        if low_confidence:
            head += (" — je ne suis pas certain que ces extraits répondent exactement à votre question."
                     if fr else " — I am not certain these excerpts answer your question exactly.")
        lines = [head + "\n"] + [f"- {sent} [{rank + 1}]" for _, rank, _, sent in chosen]
        code = _best_code_block(passages[:3], q_toks)
        if code:
            lines.append(f"\n**{'Exemple de code' if fr else 'Code example'}** [{code[0]}]\n```py\n{code[1]}\n```")
        return "\n".join(lines)


def _best_code_block(passages: list[Candidate], q_toks: set[str]) -> tuple[int, str] | None:
    """Bloc de code dont le vocabulaire recoupe le mieux la question (au moins 1 terme commun)."""
    best, best_score = None, 0.0
    for i, p in enumerate(passages, 1):
        for m in re.finditer(r"```[a-zA-Z]*\n(.*?)```", p.chunk.text, flags=re.S):
            code = m.group(1).strip()
            if not code or code.count("\n") > 40:
                continue
            score = len(q_toks & set(tokenize(code))) - 0.1 * (i - 1)
            if score > best_score:
                best, best_score = (i, "\n".join(code.splitlines()[:14])), score
    return best


_FR_WORDS = {"comment", "quel", "quelle", "est", "pour", "je", "mon", "ma", "dans", "pas", "les", "des", "une",
             "sur", "avec", "puis", "peux", "faire", "modèle", "utiliser", "erreur", "pourquoi", "dois", "mes", "du"}


def looks_french(text: str) -> bool:
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ]+", text.lower())
    return sum(w in _FR_WORDS for w in words) >= 2 or bool(re.search(r"[éèêàùç]", text.lower()))


# ----------------------------------------------------------------------------------------------
class OpenAICompatibleLLM:
    def __init__(self, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY",
                 temperature: float = 0.1, max_tokens: int = 700, timeout_s: int = 120):
        self.name = f"openai:{model}"
        self.model, self.url = model, base_url.rstrip("/") + "/chat/completions"
        self.key = os.environ.get(api_key_env, "")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout_s

    def generate(self, question, passages, history=None, low_confidence=False) -> str:
        payload = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": build_user_message(question, passages, history or [],
                                                                               low_confidence)}]}
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        try:
            r = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            raise LLMError(f"Appel LLM impossible ({self.url}) : {e}") from e


class HFLocalLLM:
    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 700):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf-local:{model}"
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        self.temperature, self.max_tokens = temperature, max_tokens

    def generate(self, question, passages, history=None, low_confidence=False) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(question, passages, history or [], low_confidence)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True)
        out = self.model.generate(**ids, max_new_tokens=self.max_tokens, do_sample=self.temperature > 0,
                                  temperature=max(self.temperature, 1e-3))
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def make_llm(cfg: dict):
    l = cfg["llm"]
    if l["backend"] == "extractive":
        return ExtractiveLLM()
    if l["backend"] == "openai":
        return OpenAICompatibleLLM(l["model"], l["base_url"], l.get("api_key_env", "OPENAI_API_KEY"),
                                   l.get("temperature", 0.1), l.get("max_tokens", 700), l.get("timeout_s", 120))
    if l["backend"] == "hf-local":
        return HFLocalLLM(l["hf_model"], l.get("temperature", 0.1), l.get("max_tokens", 700))
    raise ValueError(f"Backend LLM inconnu : {l['backend']}")
