from .symbols import *

# Initialize symbol to ID mapping
_symbol_to_id = {s: i for i, s in enumerate(symbols)}

# Import other required modules and functions
from .chinese_mix import get_bert_feature as zh_mix_en_bert

# Preload the chinese_mix model at initialization to avoid delay during runtime
# Global model cache to store different language BERT models
loaded_models = {}

# Load the chinese_mix model at initialization
loaded_models['ZH_MIX_EN'] = zh_mix_en_bert
print("Chinese-Mix English BERT model preloaded")

# Define cleaned_text_to_sequence function
def cleaned_text_to_sequence(cleaned_text, tones, language, symbol_to_id=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      cleaned_text: string to convert to a sequence
      tones: list of tones corresponding to the symbols
      language: language code to specify which language is used
      symbol_to_id: optional symbol to ID mapping
    Returns:
      Tuple containing lists of integers corresponding to the phones, tones, and language IDs
    """
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for _ in phones]
    return phones, tones, lang_ids

# Define get_bert function to support other languages
def get_bert(norm_text, word2ph, language, device):
    from .chinese_bert import get_bert_feature as zh_bert
    from .english_bert import get_bert_feature as en_bert
    from .japanese_bert import get_bert_feature as jp_bert
    from .spanish_bert import get_bert_feature as sp_bert
    from .french_bert import get_bert_feature as fr_bert
    from .korean import get_bert_feature as kr_bert

    # Map language to the corresponding BERT feature extraction function
    lang_bert_func_map = {
        "ZH": zh_bert,
        "EN": en_bert,
        "JP": jp_bert,
        'ZH_MIX_EN': zh_mix_en_bert,
        'FR': fr_bert,
        'SP': sp_bert,
        'ES': sp_bert,
        "KR": kr_bert
    }

    # Load the model if it has not been loaded yet
    if language not in loaded_models:
        loaded_models[language] = lang_bert_func_map[language]
        print(f"Model for language {language} loaded")

    # Get the model from the cache and call the corresponding feature extraction function
    bert_feature_func = loaded_models[language]
    bert = bert_feature_func(norm_text, word2ph, device)
    return bert
