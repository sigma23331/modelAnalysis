import os
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration

# 请替换为你的实际路径
MODEL_PATH = "/data/users/yubo_wang/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"

def inspect_audio_tower():
    print(f"--> [System] Loading model structure from: {MODEL_PATH} ...")
    
    # 使用 meta device 加载，速度极快且几乎不占显存
    # 我们只看结构，不需要加载真实的权重数值
    try:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="meta", 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"加载出错: {e}")
        return

    print("\n" + "="*50)
    print("           Audio Tower 结构大揭秘")
    print("="*50)

    # 1. 确认 audio_tower 是否存在
    if hasattr(model.thinker, "audio_tower"):
        tower = model.thinker.audio_tower
        print(f"模块名称: model.thinker.audio_tower")
        print(f"类型: {type(tower)}")
        print("-" * 30)
        
        # 2. 打印完整结构
        print(tower)
        
        print("-" * 30)
        # 3. 智能分析输出层
        # 我们尝试找找这个 tower 的最后一层是什么
        # 通常是 list 的最后一个元素，或者是 Sequential 的最后一层
        print("【关键分析】:")
        
        # 尝试遍历子模块找到最后一个
        last_module = None
        for name, module in tower.named_modules():
            last_module = module
        
        if last_module:
            print(f"最后一层是: {last_module}")
            if isinstance(last_module, torch.nn.Linear):
                print(f"--> 发现 Linear 层！输出维度是: {last_module.out_features}")
                if last_module.out_features == 2048:
                    print("--> 结论: 这个 audio_tower 已经包含了投影层，输出可以直接对齐 LLM！")
                elif last_module.out_features == 1280:
                    print("--> 结论: 这个 audio_tower 输出原始声学特征，未经过投影。")
            else:
                print("--> 最后一层不是 Linear，请人工检查上方打印的结构。")
                
    else:
        print("[Error] model.thinker 下依然没有找到 'audio_tower'，请检查变量名。")
        print("当前 thinker 的子模块有:", [n for n, _ in model.thinker.named_children()])

if __name__ == "__main__":
    inspect_audio_tower()