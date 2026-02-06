import os
import json
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from torch.nn import functional as F
from tqdm import tqdm
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

# ================= 配置区 =================
MODEL_PATH = "/data/users/yubo_wang/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
PROBE_PRE_PATH = "/home/yubo_wang/model_analysis/esc50_linear_probe.pth"
PROBE_POST_PATH = "/home/yubo_wang/model_analysis/esc50_projection_probe.pth"
CHAMPIONS_FILE = "/home/yubo_wang/model_analysis/esc50_champions.json"

# 输出根目录
OUTPUT_ROOT = "/data/users/yubo_wang/experiment_large_scale_dual_llm"
PLOT_DIR = os.path.join(OUTPUT_ROOT, "plots")
DATA_DIR = os.path.join(OUTPUT_ROOT, "data")
TEMP_DIR = os.path.join(OUTPUT_ROOT, "temp")

OS_ENV_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNR_LEVELS = np.arange(-20, 21, 4)
# SNR_LEVELS = [-20,0,20]
# =========================================

# === 1. 探针类 ===
class SimpleProbe(nn.Module):
    def __init__(self, input_dim, num_classes=50):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

# === 2. 双流特征提取器 (集成所有修复) ===
class DualFeatureExtractor:
    def __init__(self, model_path):
        print(f"--> [System] Loading model from {model_path}...")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, device_map="auto", trust_remote_code=True, torch_dtype="auto"
        ).eval()
        # 目标层保持不变
        self.target_layer = self.model.thinker.audio_tower.proj2

    def extract_dual(self, audio_path):
        captured = {"pre": None, "post": None}

        def hook_both(module, input, output):
            inp = input[0].detach()
            out = output[0] if isinstance(output, tuple) else output
            out = out.detach()

            # 保持你的维度修复逻辑
            if inp.dim() == 2: inp = inp.unsqueeze(0)
            if out.dim() == 2: out = out.unsqueeze(0)

            # 保持 float32 转移到 CPU
            captured["pre"] = inp.to(torch.float32).cpu()
            captured["post"] = out.to(torch.float32).cpu()

        handle = self.target_layer.register_forward_hook(hook_both)

        try:
            raw_audio, _ = librosa.load(audio_path, sr=16000)
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": "extract"}
                ]
            }]
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                text=text, audio=[raw_audio], return_tensors="pt", padding=True
            )

            # ===== 核心修复：对齐 thinker 所在的具体设备 =====
            # thinker 可能不在 cuda:0，必须动态获取
            thinker_device = next(self.model.thinker.parameters()).device
            thinker_dtype = next(self.model.thinker.parameters()).dtype

            model_inputs = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    v = v.to(thinker_device)
                    if v.is_floating_point():
                        v = v.to(thinker_dtype)
                model_inputs[k] = v

            with torch.inference_mode():
                # 触发 thinker 运行
                self.model.thinker(**model_inputs)

        except Exception as e:
            print(f"Extraction Error: {e}")
        finally:
            handle.remove()

        return captured["pre"], captured["post"]

    
import re
import json

class LLMResponder:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.esc50_labels = [
            "dog", "rooster", "pig", "cow", "frog",
            "cat", "hen", "insects", "sheep", "crow",
            "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
            "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
            "crying_baby", "sneezing", "clapping", "breathing", "coughing",
            "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping",
            "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening",
            "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking",
            "helicopter", "chainsaw", "siren", "car_horn", "engine",
            "train", "church_bells", "airplane", "fireworks", "hand_saw"
        ]
        self.idx2label = {i: v for i, v in enumerate(self.esc50_labels)}

    def _to_esc50_label(self, x):
        """
        支持输入:
        - 纯索引: 17 / "17"
        - class_索引: "class_17"
        - 已有标签: "crying_baby" / "crying baby"
        """
        # 1) int 索引
        if isinstance(x, int):
            return self.idx2label.get(x, f"class_{x}")

        s = str(x).strip().lower()

        # 2) class_17 形式
        m = re.match(r"^class_(\d+)$", s)
        if m:
            idx = int(m.group(1))
            return self.idx2label.get(idx, s)

        # 3) "17" 形式
        if s.isdigit():
            idx = int(s)
            return self.idx2label.get(idx, s)

        # 4) 文本标签规范化
        s = s.replace(" ", "_")
        return s

    @torch.inference_mode()
    def _forward_get_logits(self, **kwargs):
        """参考成功代码：手动计算 Logits 以解决多卡权重分离问题"""
        trunk = self.model.thinker
        # 获取隐藏状态
        out = trunk(**kwargs, use_cache=False, return_dict=True)
        
        if hasattr(out, "logits") and out.logits is not None:
            return out.logits
        
        # 处理隐藏状态提取
        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        elif hasattr(out, "hidden_states"):
            hidden = out.hidden_states[-1]
        else:
            hidden = out[0] if isinstance(out, (list, tuple)) else out

        # 核心修复：手动将 Embedding 权重移到 Hidden 所在设备进行 Linear 计算
        emb_weight = trunk.model.embed_tokens.weight
        if emb_weight.device != hidden.device:
            emb_weight = emb_weight.to(hidden.device)
        
        return F.linear(hidden, emb_weight)

    def ask_json(self, audio_path, class_a_name, class_b_name, max_new_tokens=96):
        label_a = self._to_esc50_label(class_a_name)
        label_b = self._to_esc50_label(class_b_name)

        prompt = (
            "任务：判断混合音频中两个候选事件的相对显著性。\n"
            f"候选A标签: {label_a}\n"
            f"候选B标签: {label_b}\n"
            "已知音频只围绕这两个候选事件进行判断，不需要引入其他类别。\n"
            "请严格按以下规则输出：\n"
            "1) A_score 和 B_score 都是 0~1 的小数，且 A_score + B_score = 1。\n"
            "2) 如果你不确定，也必须给出最接近的概率分配，不要输出解释。\n"
            "3) mentioned_events 只能从 [A, B, A&B, none] 中选一个。\n"
            "4) 只输出一行 JSON，禁止 markdown 代码块。\n"
            '输出格式: {"A_score":0.00,"B_score":0.00,"mentioned_events":"A|B|A&B|none"}'
        )

        # print(prompt)

        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": prompt}
        ]}]
        
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        raw_audio, _ = librosa.load(audio_path, sr=16000)
        batch_inputs = self.processor(text=text, audio=[raw_audio], return_tensors="pt", padding=True)

        # 1. 初始设备对齐 (通常是第一张卡)
        base_device = next(self.model.thinker.parameters()).device
        inputs = {}
        for k, v in dict(batch_inputs).items():
            if torch.is_tensor(v):
                v = v.to(base_device)
                if v.is_floating_point(): v = v.to(self.model.dtype)
            inputs[k] = v

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        # 提取除 ids 和 mask 之外的特征 (如 audio features)
        fixed_inputs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask"]}
        
        prompt_len = input_ids.shape[1]
        eos_id = self.processor.tokenizer.eos_token_id

        # 2. 手动自回归解码循环
        for _ in range(max_new_tokens):
            step_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, **fixed_inputs}
            
            # 获取 Logits (计算发生在最后一张卡)
            logits = self._forward_get_logits(**step_inputs)
            next_token_logits = logits[:, -1, :]
            
            # 贪婪搜索
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 关键修复：将生成的 Token 移回 input_ids 所在的设备 (base_device)
            if next_token.device != input_ids.device:
                next_token = next_token.to(input_ids.device)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

            if next_token.item() == eos_id:
                break

        # 3. 解码生成的文本
        gen_ids = input_ids[:, prompt_len:]
        text_out = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        
        # --- 解析逻辑 ---
        a_score, b_score = 0.0, 0.0
        mentioned = []
        parse_ok = 0
        try:
            # 尝试纯 JSON 解析
            clean_text = re.search(r'\{.*\}', text_out, re.DOTALL)
            if clean_text:
                obj = json.loads(clean_text.group())
                a_score = float(obj.get("A_score", 0.0))
                b_score = float(obj.get("B_score", 0.0))
                mentioned = obj.get("mentioned_events", [])
                parse_ok = 1
        except Exception:
            # 正则 fallback
            m1 = re.search(r'"?A_score"?\s*:\s*([0-9]*\.?[0-9]+)', text_out)
            m2 = re.search(r'"?B_score"?\s*:\s*([0-9]*\.?[0-9]+)', text_out)
            a_score = float(m1.group(1)) if m1 else 0.0
            b_score = float(m2.group(1)) if m2 else 0.0

        return {
            "raw_text": text_out,
            "A_score": float(max(0.0, min(1.0, a_score))),
            "B_score": float(max(0.0, min(1.0, b_score))),
            "Delta_L": a_score - b_score,
            "parse_ok": parse_ok,
            "mentioned_events": mentioned
        }

# === 3. 混合工具 ===
def mix_audio(path_a, path_b, snr_db):
    try:
        wav_a, _ = librosa.load(path_a, sr=16000, mono=True)
        wav_b, _ = librosa.load(path_b, sr=16000, mono=True)
        min_len = min(len(wav_a), len(wav_b))
        wav_a, wav_b = wav_a[:min_len], wav_b[:min_len]
        
        rms_a = np.sqrt(np.mean(wav_a**2))
        rms_b = np.sqrt(np.mean(wav_b**2))
        if rms_b == 0: return wav_a # 防止除零
        
        target_rms_b = rms_a / (10 ** (snr_db / 20))
        scale_b = target_rms_b / (rms_b + 1e-8)
        
        mixed = wav_a + wav_b * scale_b
        max_val = np.max(np.abs(mixed))
        if max_val > 0.99: mixed = mixed / max_val * 0.99
        return mixed
    except Exception as e:
        print(f"Mixing Error: {e}")
        return np.zeros(16000)


def save_alignment_plot(id_a, id_b, snr_list, data):
    delta_pre = np.array(data["pre"]["prob_a"]) - np.array(data["pre"]["prob_b"])
    delta_post = np.array(data["post"]["prob_a"]) - np.array(data["post"]["prob_b"])
    delta_l = np.array(data["llm"]["delta_l"])

    plt.figure(figsize=(10, 6))
    plt.plot(snr_list, delta_pre, 'k--', label="Delta_pre")
    plt.plot(snr_list, delta_post, 'b-o', label="Delta_post")
    plt.plot(snr_list, delta_l, 'r-s', label="Delta_LLM")
    plt.axhline(0, linestyle=':', linewidth=1)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Delta (A-B)")
    plt.title(f"Alignment Curve: Pair {id_a} vs {id_b}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig(os.path.join(PLOT_DIR, f"align_{id_a}_{id_b}.png"))
    plt.close()

# === 4. 绘图与保存函数 ===
def json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def save_results(id_a, id_b, snr_list, data, output_root):
    # 1. 保存 JSON 数据
    result_dict = {
        "id_a": id_a,
        "id_b": id_b,
        "snr_levels": list(snr_list),
        "pre_prob_a": data["pre"]["prob_a"],
        "pre_prob_b": data["pre"]["prob_b"],
        "post_prob_a": data["post"]["prob_a"],
        "post_prob_b": data["post"]["prob_b"],
        "llm_a_score": data["llm"]["a_score"],
        "llm_b_score": data["llm"]["b_score"],
        "llm_delta_l": data["llm"]["delta_l"],
        "llm_raw_text": data["llm"]["raw_text"],
        "llm_parse_ok": data["llm"]["parse_ok"]
    }
    json_path = os.path.join(DATA_DIR, f"pair_{id_a}_{id_b}.json")
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False, default=json_default)
        
    save_alignment_plot(id_a, id_b, snr_list, data)

def append_pair_to_master_csv(id_a, id_b, snr_list, data, csv_path):
    rows = []
    for i, snr in enumerate(snr_list):
        pre_a = data["pre"]["prob_a"][i]
        pre_b = data["pre"]["prob_b"][i]
        post_a = data["post"]["prob_a"][i]
        post_b = data["post"]["prob_b"][i]
        llm_a = data["llm"]["a_score"][i]
        llm_b = data["llm"]["b_score"][i]
        delta_pre = pre_a - pre_b
        delta_post = post_a - post_b
        delta_l = data["llm"]["delta_l"][i]

        rows.append({
            "id_a": id_a, "id_b": id_b, "snr": float(snr),
            "pre_prob_a": pre_a, "pre_prob_b": pre_b, "delta_pre": delta_pre,
            "post_prob_a": post_a, "post_prob_b": post_b, "delta_post": delta_post,
            "llm_a_score": llm_a, "llm_b_score": llm_b, "delta_l": delta_l,
            "llm_parse_ok": data["llm"]["parse_ok"][i],
            "llm_raw_text": data["llm"]["raw_text"][i],
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


# === 5. 主流程 ===
def run_large_scale():
    # 初始化目录
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 加载冠军样本
    with open(CHAMPIONS_FILE, "r") as f:
        champions = json.load(f)
        
    # 加载模型 (只加载一次)
    extractor = DualFeatureExtractor(MODEL_PATH)
    
    probe_pre = SimpleProbe(1280).to(OS_ENV_DEVICE)
    probe_pre.load_state_dict(torch.load(PROBE_PRE_PATH, map_location=OS_ENV_DEVICE))
    probe_pre.eval()
    
    probe_post = SimpleProbe(2048).to(OS_ENV_DEVICE)
    probe_post.load_state_dict(torch.load(PROBE_POST_PATH, map_location=OS_ENV_DEVICE))
    probe_post.eval()

    llm = LLMResponder(extractor.model, extractor.processor)

    
    # 生成对战列表 (只跑上三角矩阵，避免重复和自对战)
    pairs = []
    for i in range(10):
        for j in range(i + 1, 10): # i < j
            pairs.append((i, j))
            
    print(f"--> Total pairs to process: {len(pairs)}")
    
    # 进度条
    pbar = tqdm(pairs, desc="Processing Pairs")
    
    for id_a, id_b in pbar:
        # 检查是否已存在 (断点续传)
        if os.path.exists(os.path.join(DATA_DIR, f"pair_{id_a}_{id_b}.json")):
            continue
            
        file_a = champions[str(id_a)]["file"]
        file_b = champions[str(id_b)]["file"]
        
        # 存储当前 Pair 的数据
        current_data = {
            "pre": {"prob_a": [], "prob_b": []},
            "post": {"prob_a": [], "prob_b": []},
            "llm": {
                "a_score": [], "b_score": [], "delta_l": [],
                "raw_text": [], "parse_ok": []
            }
        }
        
        # SNR 循环
        # 为了速度，这里不显示内层进度条
        for snr in SNR_LEVELS:
            print(f"[RUN] pair=({id_a},{id_b}) snr={snr}")
            temp_wav = os.path.join(TEMP_DIR, f"mix_{id_a}_{id_b}_{int(snr)}.wav")
            
            # 混合
            mixed = mix_audio(file_a, file_b, snr)
            sf.write(temp_wav, mixed, 16000)
            
            # 提取
            feat_pre, feat_post = extractor.extract_dual(temp_wav)
            
            if feat_pre is None:
                current_data["pre"]["prob_a"].append(0.0)
                current_data["pre"]["prob_b"].append(0.0)
                current_data["post"]["prob_a"].append(0.0)
                current_data["post"]["prob_b"].append(0.0)

                current_data["llm"]["a_score"].append(0.0)
                current_data["llm"]["b_score"].append(0.0)
                current_data["llm"]["delta_l"].append(0.0)
                current_data["llm"]["raw_text"].append("EXTRACTION_FAILED")
                current_data["llm"]["parse_ok"].append(0)
                continue
                
            # Pooling & Probe
            v_pre = torch.mean(feat_pre, dim=1).to(OS_ENV_DEVICE)
            v_post = torch.mean(feat_post, dim=1).to(OS_ENV_DEVICE)
            
            with torch.no_grad():
                # Pre
                p_pre = F.softmax(probe_pre(v_pre), dim=1).cpu().numpy()[0]
                current_data["pre"]["prob_a"].append(float(p_pre[id_a]))
                current_data["pre"]["prob_b"].append(float(p_pre[id_b]))
                
                # Post
                p_post = F.softmax(probe_post(v_post), dim=1).cpu().numpy()[0]
                current_data["post"]["prob_a"].append(float(p_post[id_a]))
                current_data["post"]["prob_b"].append(float(p_post[id_b]))

            name_a = champions[str(id_a)].get("label", f"class_{id_a}")
            name_b = champions[str(id_b)].get("label", f"class_{id_b}")

            llm_res = llm.ask_json(temp_wav, name_a, name_b)

            current_data["llm"]["a_score"].append(llm_res["A_score"])
            current_data["llm"]["b_score"].append(llm_res["B_score"])
            current_data["llm"]["delta_l"].append(llm_res["Delta_L"])
            current_data["llm"]["raw_text"].append(llm_res["raw_text"])
            current_data["llm"]["parse_ok"].append(llm_res["parse_ok"])

        MASTER_CSV = os.path.join(OUTPUT_ROOT, "alignment_rows.csv")

        # 保存本轮结果
        save_results(id_a, id_b, SNR_LEVELS, current_data, OUTPUT_ROOT)
        append_pair_to_master_csv(id_a, id_b, SNR_LEVELS, current_data, MASTER_CSV)

if __name__ == "__main__":
    run_large_scale()