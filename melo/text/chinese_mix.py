import os
import re

import cn2an
from pypinyin import lazy_pinyin, Style
import jieba
import jieba.posseg as psg
from transformers import AutoTokenizer
from .symbols import language_tone_start_map
from .tone_sandhi import ToneSandhi
from .english import g2p as g2p_en
from . import chinese_bert
from .chinese import _g2p as _chinese_g2p  # 确保在使用之前导入

current_file_path = os.path.dirname(__file__)

# Load Bert tokenizer
model_id = 'bert-base-multilingual-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Initialize jieba tokenizer in advance
def initialize_jieba():
    jieba.lcut("初始化分词器")

initialize_jieba()

punctuation = ["!", "?", "…", ",", ".", "'", "-"]

pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

tone_modifier = ToneSandhi()

# Replace punctuation marks in text
def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    replaced_text = re.sub(r"[^\u4e00-\u9fa5_a-zA-Z\s" + "".join(punctuation) + r"]+", "", replaced_text)
    replaced_text = re.sub(r"[\s]+", " ", replaced_text)

    return replaced_text

# Another implementation of g2p (supports mixed Chinese and English)
def _g2p_v2(segments):
    spliter = '#$&^!@'

    phones_list = []
    tones_list = []
    word2ph = []

    for text in segments:
        assert spliter not in text
        text = re.sub('([a-zA-Z\s]+)', lambda x: f'{spliter}{x.group(1)}{spliter}', text)
        texts = text.split(spliter)
        texts = [t for t in texts if len(t) > 0]

        for text in texts:
            if re.match('[a-zA-Z\s]+', text):
                tokenized_en = tokenizer.tokenize(text)
                phones_en, tones_en, word2ph_en = g2p_en(text=None, pad_start_end=False, tokenized=tokenized_en)
                tones_en = [t + language_tone_start_map['EN'] for t in tones_en]
                phones_list += phones_en
                tones_list += tones_en
                word2ph += word2ph_en
            else:
                phones_zh, tones_zh, word2ph_zh = _chinese_g2p([text])
                phones_list += phones_zh
                tones_list += tones_zh
                word2ph += word2ph_zh
    return phones_list, tones_list, word2ph

# Grapheme-to-phoneme function (g2p)
def g2p(text, impl='v2'):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    if impl == 'v1':
        _func = _g2p
    elif impl == 'v2':
        _func = _g2p_v2
    else:
        raise NotImplementedError()
    phones, tones, word2ph = _func(sentences)
    assert sum(word2ph) == len(phones)
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph

# Main implementation of pinyin-to-phoneme conversion
def _g2p(segments):
    phones_list = []
    tones_list = []
    word2ph = []
    for seg in segments:
        seg_cut = psg.lcut(seg)
        initials = []
        finals = []
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        for word, pos in seg_cut:
            if pos == "eng":
                initials.append(['EN_WORD'])
                finals.append([word])
            else:
                sub_initials, sub_finals = _get_initials_finals(word)
                sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
                initials.append(sub_initials)
                finals.append(sub_finals)

        initials = sum(initials, [])
        finals = sum(finals, [])

        for c, v in zip(initials, finals):
            if c == 'EN_WORD':
                tokenized_en = tokenizer.tokenize(v)
                phones_en, tones_en, word2ph_en = g2p_en(text=None, pad_start_end=False, tokenized=tokenized_en)
                tones_en = [t + language_tone_start_map['EN'] for t in tones_en]
                phones_list += phones_en
                tones_list += tones_en
                word2ph += word2ph_en
            else:
                raw_pinyin = c + v
                if c == v:
                    assert c in punctuation
                    phone = [c]
                    tone = "0"
                    word2ph.append(1)
                else:
                    v_without_tone = v[:-1]
                    tone = v[-1]

                    pinyin = c + v_without_tone
                    assert tone in "12345"

                    if c:
                        v_rep_map = {
                            "uei": "ui",
                            "iou": "iu",
                            "uen": "un",
                        }
                        if v_without_tone in v_rep_map.keys():
                            pinyin = c + v_rep_map[v_without_tone]
                    else:
                        pinyin_rep_map = {
                            "ing": "ying",
                            "i": "yi",
                            "in": "yin",
                            "u": "wu",
                        }
                        if pinyin in pinyin_rep_map.keys():
                            pinyin = pinyin_rep_map[pinyin]
                        else:
                            single_rep_map = {
                                "v": "yu",
                                "e": "e",
                                "i": "y",
                                "u": "w",
                            }
                            if pinyin[0] in single_rep_map.keys():
                                pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                    assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                    phone = pinyin_to_symbol_map[pinyin].split(" ")
                    word2ph.append(len(phone))

                phones_list += phone
                tones_list += [int(tone)] * len(phone)
    return phones_list, tones_list, word2ph

# Get initials and finals from a Chinese word
def _get_initials_finals(word):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals

# Text normalization
def text_normalize(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    text = replace_punctuation(text)
    return text

# Get Bert feature representation
def get_bert_feature(text, word2ph, device):
    return chinese_bert.get_bert_feature(text, word2ph, model_id='bert-base-multilingual-uncased', device=device)

# Initialization function to call get_bert_feature in advance
def initialize_bert_feature():
    sample_text = "初始化 BERT 特征提取的示例文本"
    sample_text = text_normalize(sample_text)
    _, _, word2ph = g2p(sample_text, impl='v2')
    # Initialize BERT feature extraction (this will load the model)
    get_bert_feature(sample_text, word2ph, device='cuda:0')

# Execute BERT feature initialization when the module loads
initialize_bert_feature()

if __name__ == "__main__":
    text = "NFT啊！chemistry 但是《原神》是由,米哈游自主，  [研发]的一款全新开放世界冒险游戏"
    text = text_normalize(text)
    print(text)
    phones, tones, word2ph = g2p(text, impl='v2')
    bert = get_bert_feature(text, word2ph, device='cuda:0')
    print(phones)
    import pdb; pdb.set_trace()
