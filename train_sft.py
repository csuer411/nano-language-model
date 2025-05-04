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
    nlm.load(config.model_path)
    # 计算参数数量
    num_params = sum(p.numel() for p in nlm.parameters())
    print(f"模型参数数量: {num_params}")
    return tokenizer, nlm


class sft_dataset:
    def __init__(self, data_path, tokenizer):
        """jsonl格式，支持缓存功能"""
        self.data = []
        self.data_path = data_path
        self.bos = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        self.eos = tokenizer("<|im_end|>", add_special_tokens=False).input_ids
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
                # 使用正确的参数调用apply_chat_template，确保返回的是token IDs列表
                tokenized = tokenizer.apply_chat_template(json.loads(line.strip())["conversations"], 
                                                       truncation=True, 
                                                       max_length=config.max_len)
                self.data.append(tokenized)
        
        # 保存到缓存
        if config.use_cache:
            print(f"正在保存数据到缓存: {cache_path}")
            cache_data = {
                "tokenized_data": self.data,
                "max_len": config.max_len,
                "data_path": data_path
            }
            torch.save(cache_data, cache_pat)
            print("缓存保存完成")

    def __getitem__(self, index):
        """完成sft数据构建"""
        full_ids = self.data[index]
        

        x, y = full_ids[:-1], full_ids[1:]
        start = 0
        # 查找助手回复的开始标记
        while start < len(y):
            if start + len(self.bos) <= len(y) and y[start:start+len(self.bos)] == self.bos:
                break
            start += 1
        
        # 如果找到了助手标记，将非助手部分的标签设为0（只训练助手回复部分）
        if start < len(y):
            y[:start+len(self.bos)] = [0] * (start+len(self.bos))
        else:
            # 如果没有找到助手标记，这可能是数据格式问题，将所有标签设为0
            print(f"警告：在索引 {index} 的数据中未找到助手标记，请检查数据格式")
            y[:] = [0] * len(y)
            
        return x, y


    def __len__(self):
        return len(self.data)


def batch_padding(batch):
    """batch动态填充"""
    max_len = max([len(x) for x, _ in batch])
    x, y = zip(*batch)
    x = [xi + [0] * (max_len - len(xi)) for xi in x]
    y = [yi + [0] * (max_len - len(yi)) for yi in y]
    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def prepare_data(tokenizer):
    """准备数据加载器"""
    sft_dataset_obj = sft_dataset(config.sft_data_path, tokenizer)
    data_loader = DataLoader(sft_dataset_obj, batch_size=config.batch_size, shuffle=True, collate_fn=batch_padding)
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
            if step_count % 1 == 0 or step_count == 1:
                print(f"Step {step_count}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # 计算并记录当前epoch的平均损失
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}/{config.epochs}, Avg Loss: {avg_loss:.4f}")
    
    return global_steps, all_losses, all_lr


def plot_metrics(global_steps, all_losses, all_lr):
    """绘制损失和学习率曲线"""
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 绘制损失曲线 (每个batch)
    plt.figure(figsize=(12, 6))
    plt.plot(global_steps, all_losses, alpha=0.3, color='blue', label='每个batch的损失')
    
    # 添加平滑曲线 (使用移动平均)
    window_size = min(10, len(all_losses))
    if window_size > 0:
        smoothed_losses = np.convolve(all_losses, np.ones(window_size)/window_size, mode='valid')
        # 调整x轴以匹配平滑后的数据长度
        smooth_steps = global_steps[window_size-1:]
        plt.plot(smooth_steps, smoothed_losses, linewidth=2, color='red', label='平滑后的损失')
    
    plt.title('训练损失')
    plt.xlabel('训练步数')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/loss_curve.png')

    # 绘制学习率曲线
    plt.figure(figsize=(12, 6))
    plt.plot(global_steps, all_lr, color='green')
    plt.title('学习率变化')
    plt.xlabel('训练步数')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig('plots/lr_curve.png')


def test_generation(nlm, tokenizer, device):
    """测试模型生成能力"""
    test = config.conversation
    print(f"测试提示: {test[0]}")
    input_ids = tokenizer.apply_chat_template(test, return_tensors="pt",
                                             add_generation_prompt=True).to(device)
    
    # 使用改进后的generate方法，设置合理的最大生成长度
    output_ids = nlm.generate(input_ids)
    
    # 解码生成的token ID为文本
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(f"生成结果: {output_text}")
    
    # 同时打印原始token ID的长度，便于调试
    print(f"生成的token数量: {len(output_ids[0])}")



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
    nlm.save(config.sft_model_path)
    # 测试生成
    test_generation(nlm, tokenizer, device)
    
    print(f"模型已保存到 {config.sft_model_path}")


if __name__ == "__main__":
    main()