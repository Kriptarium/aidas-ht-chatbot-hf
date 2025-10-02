
from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

_MODEL_ID = "google/flan-t5-small"

class HFSingleton:
    _pipe = None

    @classmethod
    def get_pipe(cls):
        if cls._pipe is None:
            tok = AutoTokenizer.from_pretrained(_MODEL_ID)
            model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_ID)
            cls._pipe = pipeline("text2text-generation", model=model, tokenizer=tok, max_new_tokens=256)
        return cls._pipe

def generate_answer(prompt: str) -> str:
    pipe = HFSingleton.get_pipe()
    out = pipe(prompt)[0]["generated_text"]
    return out
