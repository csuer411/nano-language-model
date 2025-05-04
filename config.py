from dataclasses import dataclass
import torch

@dataclass
class Config:
    n_layer: int = 8
    n_dim: int = 768
    n_head: int = 12
    n_vocabulary: int = 6400
    max_len: int = 512
    batch_size: int = 48
    lr: float = 8e-5
    epochs: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_path: str = "dir/pretrain_hq.jsonl"
    model_path: str = "nlm_pt.pth"
    tokenizer_path: str = "./tokenizers"
    test_prompt: str = "北京"
    # 学习率调度器参数
    scheduler_type: str = "cosine"  # 可选: linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
    num_warmup_steps: int = 0
    # 数据集缓存参数
    use_cache: bool = True  # 是否使用缓存
    cache_dir: str = "./cache"  # 缓存目录  # 预热步数
    conversation = [{"role": "user", "content": "“四大发明”是什么？"}]
    sft_data_path: str = "dir/sft_512.jsonl"
    sft_model_path: str = "sft.pth"
    