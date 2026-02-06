import os
import torch
import sys
import librosa
import numpy as np

# ================= 1. 环境变量与路径配置 =================
os.environ["HF_HOME"] = "/data/users/yubo_wang/hf_cache"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

MODEL_PATH = "/data/users/yubo_wang/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"

# ================= 2. 依赖导入 =================
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

# ================= 3. 辅助函数 =================
def extract_audio_paths(conversation):
    paths = []
    for turn in conversation:
        if "content" in turn:
            for item in turn["content"]:
                if item.get("type") == "audio":
                    paths.append(item["audio"])
    return paths

# ================= 4. 核心 Actor 类 =================
class QwenAudioActor:
    def __init__(self, device="cuda"):
        # 注意：在 device_map="auto" 模式下，self.device 只是一个初始指引
        # 实际 tensors 会分布在不同卡上
        self.device = device 
        self.max_new_tokens = 64
        
        print(f"--> [System] Setting HF_HOME: {os.environ['HF_HOME']}")
        print(f"--> [System] Loading Qwen-Omni Actor from: {MODEL_PATH} ...")
        
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
        
        # 保持 device_map="auto" 以支持多卡推理
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="auto", 
            attn_implementation="sdpa",
            torch_dtype="auto",
        ).eval()
        
        self.model.config.use_cache = False
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = False

    @torch.inference_mode()
    def forward_get_logits(self, **kwargs):
        """[底层工具] 获取 Logits"""
        trunk = getattr(self.model, "thinker", None)
        if trunk is None:
            raise RuntimeError("Model has no model.thinker")

        out = trunk(**kwargs, use_cache=False, return_dict=True)

        if hasattr(out, "logits") and out.logits is not None:
            return out.logits
        
        x0 = out[0] if isinstance(out, (tuple, list)) else None
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        elif hasattr(out, "hidden_states"):
            hidden = out.hidden_states[-1]
        elif torch.is_tensor(x0):
            hidden = x0
        else:
            raise RuntimeError("Cannot extract hidden states")

        # 处理多卡：Hidden 在最后一张卡，Embed 权重可能在第一张卡
        emb = self.model.thinker.model.embed_tokens
        W = emb.weight
        
        # 如果权重和 Hidden 不在同一个设备，把权重拷过去计算 (只占用少量显存)
        if W.device != hidden.device:
            W = W.to(hidden.device)

        # 此时 logits 会在 hidden 所在的设备 (最后一张卡)
        logits = torch.nn.functional.linear(hidden, W)
        return logits

    @torch.inference_mode()
    def generate_description(self, audio_path: str, do_sample: bool = False):
        """
        [核心功能] 输入音频，输出描述。
        """
        prompt = "请详细描述音频"
        
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ]
        }]

        # 1. 预处理
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        raw_audio, _ = librosa.load(audio_path, sr=16000)
        audios = [raw_audio] 

        batch_inputs = self.processor(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )

        # 2. 移动输入到默认设备 (通常是 cuda:0)
        inputs = {}
        # 获取模型第一层的设备作为输入的基准设备
        # 如果 model.device 是 meta (加载阶段)，则默认用 cuda
        base_device = self.model.device if self.model.device.type != "meta" else self.device
        
        for k, v in dict(batch_inputs).items():
            if torch.is_tensor(v):
                v = v.to(base_device) # 输入数据放在第一张卡
                if v.is_floating_point():
                    v = v.to(dtype=self.model.dtype)
                inputs[k] = v
            else:
                inputs[k] = v

        # 3. 手动解码循环
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        fixed_inputs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}
        
        prompt_len = input_ids.shape[1]
        eos_id = self.processor.tokenizer.eos_token_id

        for _ in range(self.max_new_tokens):
            step_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, **fixed_inputs}
            
            # logits 在最后一张卡 (例如 cuda:2)
            logits = self.forward_get_logits(**step_inputs)
            next_token_logits = logits[:, -1, :] 

            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # =======================================================
            # 关键修复：将 next_token 移回 input_ids 所在的设备 (cuda:0)
            # =======================================================
            if next_token.device != input_ids.device:
                next_token = next_token.to(input_ids.device)
            
            # 现在都在同一设备了，可以安全拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # mask 同理
            new_mask = torch.ones_like(next_token)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break

        gen_ids = input_ids[:, prompt_len:]
        response = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        
        return response, gen_ids

# ================= 测试代码 =================
if __name__ == "__main__":
    test_audio = "/data/users/yubo_wang/ESC-50/audio/1-100032-A-0.wav"
    
    if os.path.exists(test_audio):
        print(f"\n[Test] Processing: {test_audio}")
        try:
            actor = QwenAudioActor() # 默认多卡
            desc, _ = actor.generate_description(test_audio, do_sample=True)
            print(f"Generated: {desc}")
        except Exception as e:
            import traceback
            traceback.print_exc()
    else:
        print(f"提示：请确保音频文件存在: {test_audio}")