from transformers import AutoTokenizer
from model import NLM
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_scheduler
from config import Config
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
import json


# 加载配置
config = Config()


def init_model_and_tokenizer():
    """初始化tokenizer和模型"""
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    nlm = NLM(config.n_layer, config.n_dim, config.n_head, config.n_vocabulary, config.max_len)
    # 计算参数数量
    num_params = sum(p.numel() for p in nlm.parameters())
    print(f"模型参数数量: {num_params}")
    return tokenizer, nlm


class pretrain_dataset:
    def __init__(self, data_path, tokenizer):
        """jsonl格式，支持缓存功能"""
        self.data = []
        self.data_path = data_path
        
        # 创建缓存目录
        if config.use_cache and not os.path.exists(config.cache_dir):
            os.makedirs(config.cache_dir, exist_ok=True)
            
        # 生成缓存文件名（基于数据路径和最大长度）
        cache_filename = f"{os.path.basename(data_path)}_{config.max_len}_cache.pt"
        cache_path = os.path.join(config.cache_dir, cache_filename)
        
        # 检查缓存是否存在
        if config.use_cache and os.path.exists(cache_path):
            print(f"正在从缓存加载数据: {cache_path}")
            try:
                cache_data = torch.load(cache_path)
                # 验证缓存数据的有效性
                if cache_data.get("max_len") == config.max_len and cache_data.get("data_path") == data_path:
                    self.data = cache_data["tokenized_data"]
                    print(f"成功从缓存加载了 {len(self.data)} 条数据")
                    return
                else:
                    print("缓存数据参数不匹配，将重新处理数据")
            except Exception as e:
                print(f"加载缓存失败: {e}，将重新处理数据")
        
        # 如果没有缓存或缓存无效，则处理数据
        print(f"正在处理数据: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(tokenizer(json.loads(line.strip())["text"], truncation=True, 
                max_length=config.max_len)["input_ids"])
        
        # 保存到缓存
        if config.use_cache:
            print(f"正在保存数据到缓存: {cache_path}")
            cache_data = {
                "tokenized_data": self.data,
                "max_len": config.max_len,
                "data_path": data_path
            }
            torch.save(cache_data, cache_path)
            print("缓存保存完成")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def batch_padding(batch):
    max_len = max([len(x) for x in batch])
    batch = [x + [0] * (max_len - len(x)) for x in batch]
    input_ids = [x[:-1] for x in batch]
    target_ids = [x[1:] for x in batch]
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def prepare_data(tokenizer):
    """准备数据加载器"""
    pt_dataset = pretrain_dataset(config.data_path, tokenizer)
    data_loader = DataLoader(pt_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=batch_padding, num_workers=10)
    return data_loader


def setup_training(nlm):
    """设置优化器、设备和学习率调度器"""
    optimizer = AdamW(nlm.parameters(), lr=config.lr)
    device = torch.device(config.device)
    nlm.to(device)

    return optimizer, device


def setup_scheduler(optimizer, data_loader):
    """设置学习率调度器"""
    # 计算总训练步数
    num_training_steps = len(data_loader) * config.epochs

    # 创建学习率调度器
    scheduler = get_scheduler(
        name=config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return scheduler


def train_model(nlm, data_loader, optimizer, scheduler, device):
    """训练模型并记录损失和学习率"""
    # 创建保存图表的目录
    os.makedirs("plots", exist_ok=True)

    # 用于记录每个batch的损失值和学习率
    all_losses = []
    all_lr = []
    step_count = 0
    global_steps = []
    
    for epoch in range(config.epochs):
        epoch_losses = []
        for input_ids, target_ids in tqdm(data_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            step_count += 1
            global_steps.append(step_count)
            
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            optimizer.zero_grad()
            loss = nlm.compute_loss(input_ids, target_ids)
            loss.backward()
            optimizer.step()
            # 更新学习率
            scheduler.step()
            
            # 记录当前学习率和损失
            current_lr = optimizer.param_groups[0]["lr"]
            all_lr.append(current_lr)
            all_losses.append(loss.item())
            epoch_losses.append(loss.item())
            
            # 每10个batch打印一次当前损失
            if step_count % 10 == 0 or step_count == 1:
                print(f"Step {step_count}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # 计算并记录当前epoch的平均损失
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}/{config.epochs}, Avg Loss: {avg_loss:.4f}")
    
    return global_steps, all_losses, all_lr


def plot_metrics(global_steps, all_losses, all_lr):
    """绘制损失和学习率曲线"""
    # 设置中文字体支持
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 使用系统已有的DejaVu Sans字体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 解决中文显示问题
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    # 设置默认编码为utf-8
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    # 绘制损失曲线 (每个batch)
    plt.figure(figsize=(12, 6))
    plt.plot(global_steps, all_losses, alpha=0.3, color='blue', label='Loss per batch')
    
    # 添加平滑曲线 (使用移动平均)
    window_size = min(10, len(all_losses))
    if window_size > 0:
        smoothed_losses = np.convolve(all_losses, np.ones(window_size)/window_size, mode='valid')
        # 调整x轴以匹配平滑后的数据长度
        smooth_steps = global_steps[window_size-1:]
        plt.plot(smooth_steps, smoothed_losses, linewidth=2, color='red', label='Smoothed loss')
    
    plt.title('Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/loss_curve.png')

    # 绘制学习率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(global_steps, all_lr, color='green')
    plt.title('Learning Rate Changes')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig('plots/lr_curve.png')


def test_generation(nlm, tokenizer, device):
    """测试模型生成能力"""
    test = [config.test_prompt]
    input_ids = tokenizer(test, return_tensors="pt")["input_ids"].to(device)
    output = nlm.generate(input_ids)
    print(f"生成结果: {output}")


def main():
    # 初始化模型和tokenizer
    tokenizer, nlm = init_model_and_tokenizer()
    
    # 准备数据
    data_loader = prepare_data(tokenizer)
    
    # 设置训练
    optimizer, device = setup_training(nlm)
    scheduler = setup_scheduler(optimizer, data_loader)
    
    # 训练模型
    global_steps, all_losses, all_lr = train_model(nlm, data_loader, optimizer, scheduler, device)
    
    # 绘制指标
    plot_metrics(global_steps, all_losses, all_lr)
    # 保存模型
    nlm.save(config.model_path)
    print(f"模型已保存到 {config.model_path}")
    # 测试生成
    test_generation(nlm, tokenizer, device)


if __name__ == "__main__":
    main()