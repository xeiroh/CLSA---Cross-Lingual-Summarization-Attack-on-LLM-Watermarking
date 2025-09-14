from functools import lru_cache
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, BartTokenizer, BartForConditionalGeneration

@lru_cache(None)
def xlsum_attack(text, language="swahili"):
	pass