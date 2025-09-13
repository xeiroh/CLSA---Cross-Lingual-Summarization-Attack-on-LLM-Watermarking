from functools import lru_cache
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, BartTokenizer, BartForConditionalGeneration

@lru_cache(None)
def translate(text, target_lang="amharic", source_lang="en"):
	languages = {
		"english": "en",
		"swahili": "sw",
		"spanish": "es",
		"chinese": "zh",
		"amharic": "am"}
	target_lang = languages.get(target_lang.lower(), "am")
	model_name = "facebook/m2m100_1.2B"
	tokenizer = M2M100Tokenizer.from_pretrained(model_name)
	model = M2M100ForConditionalGeneration.from_pretrained(model_name)

	tokenizer.src_lang = languages.get(source_lang.lower(), "en")
	inputs = tokenizer(text, return_tensors="pt")
	outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
	return tokenizer.batch_decode(outputs, skip_special_tokens=True)

@lru_cache(None)
def summarize(text, max_length=250):
	tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
	model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

	inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
	summary_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
	return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
