from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, BartTokenizer, BartForConditionalGeneration


def translate(text, target_lang="swahili", source_lang="en"):
	languages = {
		"english": "en",
		"swahili": "sw",
		"spanish": "es",
		"chinese": "zh"}
	target_lang = languages.get(target_lang.lower(), "sw")
	model_name = "facebook/m2m100_1.2B"
	tokenizer = M2M100Tokenizer.from_pretrained(model_name)
	model = M2M100ForConditionalGeneration.from_pretrained(model_name)

	inputs = tokenizer(text, return_tensors="pt", src_lang=source_lang, tgt_lang=target_lang)
	outputs = model.generate(**inputs)
	return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def summarize(text, max_length=250):
	tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
	model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

	inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
	summary_ids = model.generate(inputs["input_ids"], max_length=max_length, num_beams=4, early_stopping=True)
	return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
