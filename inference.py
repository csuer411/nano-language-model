from transformers import AutoTokenizer
from config import Config
from model import NLM
import torch
config = Config()
device = torch.device(config.device)
tokenizer = AutoTokenizer.from_pretrained("tokenizers")
nlm = NLM(config.n_layer, config.n_dim, config.n_head, config.n_vocabulary, config.max_len)
nlm.load("sft.pth")
nlm.to(device)
test = [{"role": "user", "content": "写一首绚丽的宋词"}]
print(f"测试提示: {test}")
input_ids = tokenizer.apply_chat_template(test, return_tensors="pt",
                                             add_generation_prompt=True).to(device)
output_ids = nlm.generate(input_ids, temperature=1)
# 解码生成的token ID为文本
print(output_ids)