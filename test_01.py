from melo.api import TTS

# Speed is adjustable
speed = 1.0
#device = 'cpu' # or cuda:0
device = 'cuda:0' # or cuda:0




#text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
text = "近日，上海公共交通卡股份有限公司发布了严正声明，其中提到，有两家深圳的公司通过其运营的视频号，向公众发布数条有关上海公交卡的虚假优惠信息，给公众造成了误导，也给本市交通卡正常运营带来了极大干扰。上海公共交通卡股份有限公司从未开展过这些优惠活动，请公众不要轻信不法分子的营销手段。"
model = TTS(language='ZH', device=device)

speaker_ids = model.hps.data.spk2id


# 打印模型的参数类型
for name, param in model.named_parameters():
    print(f"Parameter name: {name}, dtype: {param.dtype}")


output_path = 'zh.wav'
model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=speed)

