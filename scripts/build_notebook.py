# -*- coding: utf-8 -*-
"""
產生 full-work-though.ipynb
這個 builder 只負責組裝 JSON，方便維護與重跑。
"""
import json, os, io, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CELLS = []
_cell_n = [0]
def _add(cell):
    _cell_n[0] += 1
    cell["id"] = f"cell-{_cell_n[0]:03d}"
    CELLS.append(cell)

def md(text):
    _add({"cell_type": "markdown", "metadata": {},
          "source": text.strip("\n") + "\n"})
def code(text):
    _add({"cell_type": "code", "execution_count": None, "metadata": {},
          "outputs": [], "source": text.strip("\n") + "\n"})

# ============================================================
md(r"""
# 從零到完成：Qwen3.5-2B/4B-Base 數學推理 LoRA 微調全流程

> **本 Notebook 的目標**：在**單張 16 GB VRAM 的消費級顯卡**上，把 `Qwen/Qwen3.5-2B-Base`
> 與 `Qwen/Qwen3.5-4B-Base` 微調成「會先思考、再作答」的數學解題模型，
> 並完成 **訓練 → 評估 → 合併 → GGUF → 量化 → 配對檢定 → 畫圖** 的完整閉環。
>
> 依照 `project_plan.md`（操作手冊）與 `Proposal.md`（提案書）撰寫。
> **所有輸出、說明與註解皆使用繁體中文。**

---

## 這份 Notebook 的六個階段

| 階段 | 章節 | 產出 | 大約時間（首次） |
| --- | --- | --- | --- |
| **A. 環境** | §1–§3 | 可用 CUDA 的 `.venv`、FLA 加速 | 20–60 分鐘（下載為主） |
| **B. 資料** | §4 | `sft_5k.jsonl`、`smoke.jsonl`、去汙染報告 | 20–60 分鐘 |
| **C. 訓練** | §5–§7 | LoRA adapter + `vram_report.json` | 2B: 0.5–1.5 小時／4B: 1.5–3 小時 |
| **D. 評估** | §8 | 各模型 × 各基準的 JSON 結果 | 每組 200 題約 10–40 分鐘 |
| **E. 部署** | §9–§11 | 合併權重、GGUF F16/Q8_0/Q4_K_M | 30–60 分鐘 |
| **F. 結論** | §12–§13 | 對照總表、效率前緣圖、量化代價圖 | 5–10 分鐘 |

---

## 三個「先講清楚」的關鍵事實

### 事實 1：主力方法是 **bf16 LoRA**，不是 QLoRA

`project_plan.md` §0.2 與 §5.6 的結論：**QLoRA 在 Qwen3.5 這個混合架構上會直接崩潰**，
而且就算修好了也只省約 1.4 GB。相對地：

- **2B 的 bf16 LoRA 只要約 4.6 GB**
- **4B 的 bf16 LoRA 只要約 9.5 GB**

兩者都在 16 GB 以內，**還留有餘裕**。所以本 Notebook 的預設路徑是 bf16 LoRA。

### 事實 2：`fla` 是「加速器」，不是「必需品」

`transformers` 的 Qwen3.5 實作中，Gated DeltaNet 這條路徑是**選擇性**使用
`fla`（flash-linear-attention）與 `causal_conv1d` 的：

> 有裝 → 走快速 Triton/融合核心；沒裝 → **自動退回較慢、較吃記憶體的純 PyTorch 算子**。

也就是說：**`fla` 沒裝成功，模型仍然跑得動，只是慢一點。**
本 Notebook 會把這件事**明確檢測並印出來**，讓你知道自己處於哪一種狀態。

### 事實 3：E7（QLoRA 不跳過 `linear_attn`）**很可能不會崩潰**

`project_plan.md` §5.6 預期 `--quant qlora` 會出現：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (258x5120 and 1x15728640)
```

**但那個崩潰的觸發條件是「載入到 Unsloth 的預量化鏡像」**（issue #10010 / #9867）。
在**你這台機器上**，實際情況是：

1. 本機的 Hugging Face 快取裡**沒有** `unsloth/*-bnb-4bit` 預量化鏡像；
2. 也就是說 Unsloth **無法**把你導向那個壞掉的鏡像；
3. 因此 bitsandbytes 會**即時量化原始檢查點**，`quant_state` 是完整的，
   **那條 `mat1 and mat2 shapes cannot be multiplied` 就不一定會發生**。

**本 Notebook 的處理方式**：§10 把 E7 寫成**「受控的失敗實驗」**，
它會**同時接受兩種結果**，並且都當成有效的實驗數據：

- **真的崩潰** → 保存完整 traceback，證明「QLoRA 在此架構上不可用」（可引用的第一手證據）；
- **竟然跑起來** → 記錄 VRAM 與 trainable%，
  並在報告中誠實修正為「崩潰的觸發條件是預量化鏡像，而非架構本身」。

> **這才是正確的科學態度**：不要為了符合預期而修改結論。
> 若真的跑起來了，那反而是**比崩潰更有價值的發現**。

---

## 先備條件

- Windows x64、Python 3.11／3.12（本機為 3.12.10）
- 支援 CUDA 的 NVIDIA 顯卡（本機為 RTX 4070，16 GB）
- Git（轉 GGUF 用）、至少 **150 GB** 可用硬碟
- 選配：**LM Studio**（§8 的評估後端，需安裝並切到 CUDA runtime）
""")

md(r"""
## §0 控制台：設定本次要執行到哪裡

**預估時間：< 1 秒**

這個 Notebook 是**從零到完成**的完整流程，全部跑完需要好幾個小時。
實際使用時，你通常只會想跑其中一部分。所以先設定這格：

- 第一次使用：全部保持 `True`，從 §1 往下跑
- 只想看環境：把後面幾個階段改成 `False`
- 已經有 `.venv` 與資料：把 `RUN_SETUP`、`RUN_DATA` 改成 `False`

> ⚠️ **§0–§3 一律都要執行**（它們只做檢測，不裝東西、不花時間）。
> 後面的階段才會依照這裡的開關決定要不要跑。
""")

code(r'''
#@title §0 控制台（$ < 1 秒）{display-mode: "form"}
# ============================================================
# 環境：任何 Python
# 預估時間：< 1 秒
#
# 這是整個 Notebook 的總開關。後面的每一格都會先檢查對應的旗標。
# ★ 第一次使用請全部保持 True，並從上往下依序執行。
# ============================================================

RUN_SETUP    = True    # §2  建立 .venv 與安裝套件（已建好就改 False）
RUN_DOWNLOAD = True    # §4  下載 Qwen3.5-2B/4B 模型（約 14 GB）
RUN_DATA     = True    # §5  下載並轉換數學資料集（約 2.15 GB）
RUN_TRAIN    = True    # §6–§7  訓練 LoRA（最花時間）
RUN_EVAL     = True    # §8  評估（需要 LM Studio）
RUN_QLORA    = True    # §9  E7：故意重現 QLoRA 崩潰（很快，建議跑）
RUN_MERGE    = True    # §10 合併權重
RUN_GGUF     = True    # §11 轉 GGUF 與量化
RUN_REPORT   = True    # §12–§13 彙整與繪圖（很快，建議跑）

# 只想先跑通流程（不下載、不訓練）的快速模式：
QUICK_MODE   = False   # ← 改成 True 會自動把上面大部分關掉，只留檢測
if QUICK_MODE:
    RUN_SETUP = RUN_DOWNLOAD = RUN_DATA = RUN_TRAIN = False
    RUN_EVAL = RUN_MERGE = RUN_GGUF = False
    RUN_QLORA = RUN_REPORT = False
    print("⚠️ QUICK_MODE = True：只會執行環境檢測（§1–§3），其餘階段全部跳過。")

print("=" * 60)
print(" 本次執行設定")
print("=" * 60)
for name, val in [("RUN_SETUP", RUN_SETUP), ("RUN_DOWNLOAD", RUN_DOWNLOAD),
                  ("RUN_DATA", RUN_DATA), ("RUN_TRAIN", RUN_TRAIN),
                  ("RUN_EVAL", RUN_EVAL), ("RUN_QLORA", RUN_QLORA),
                  ("RUN_MERGE", RUN_MERGE), ("RUN_GGUF", RUN_GGUF),
                  ("RUN_REPORT", RUN_REPORT)]:
    print(f"  {name:14s} {'✅ 執行' if val else '⏭️  跳過'}")
print("=" * 60)
''')

md(r"""
---
# 階段 A：環境

## §1 硬體與系統檢查

**預估時間：< 10 秒**

先確認「你的電腦能不能做」。這一格不裝任何東西，只做現實檢查。
""")

code(r'''
#@title §1.1 硬體與系統檢查 {display-mode: "form"}
# ============================================================
# 環境：任何 Python（不需要 GPU 也能看）
# 預估時間：< 10 秒
# ============================================================
import os, sys, platform, shutil, subprocess

def sh(cmd, timeout=30):
    """執行外部指令並回傳 (returncode, stdout+stderr)。失敗不拋例外。"""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"

print("=" * 62)
print(" 系統資訊")
print("=" * 62)
print(f" 作業系統      : {platform.system()} {platform.release()} ({platform.machine()})")
print(f" Python        : {sys.version.split()[0]}  ({sys.executable})")
try:
    import torch  # noqa
    print(f" 目前直譯器的 torch : {torch.__version__}")
except Exception:
    print(" 目前直譯器的 torch : （未安裝，稍後會在 .venv 內安裝）")

print("\n" + "=" * 62)
print(" GPU 資訊（nvidia-smi）")
print("=" * 62)
rc, out = sh("nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version "
             "--format=csv,noheader")
if rc == 0 and out.strip():
    for line in out.strip().splitlines():
        name, total, used, drv = [x.strip() for x in line.split(",")]
        print(f" GPU           : {name}")
        print(f" VRAM 總量     : {total}")
        print(f" VRAM 已使用   : {used}")
        print(f" 驅動版本      : {drv}")
else:
    print(" [!] 找不到 nvidia-smi。若你有 NVIDIA 顯卡，請把驅動更新到支援 CUDA 12.x 以上。")
    print(out[:400])

print("\n" + "=" * 62)
print(" 硬碟可用空間")
print("=" * 62)
for drive in ["C:\\", "D:\\", "E:\\", "F:"]:
    if os.path.exists(drive):
        try:
            t, u, f = shutil.disk_usage(drive)
            flag = "OK" if f / 2**30 >= 150 else "偏少（建議 150 GB 以上）"
            print(f" {drive}  可用 {f/2**30:8.1f} GB / 總計 {t/2**30:8.1f} GB   [{flag}]")
        except Exception as e:
            print(f" {drive}  讀取失敗：{e}")

print("\n" + "=" * 62)
print(" 其他工具")
print("=" * 62)
for tool, args in [("git", "--version"), ("lms", "--version")]:
    rc, out = sh(f"{tool} {args}", timeout=20)
    status = out.strip().splitlines()[0] if (rc == 0 and out.strip()) else "（未安裝／不在 PATH）"
    print(f" {tool:6s}: {status}")

print("""
--------------------------------------------------------------------
 驗收標準（project_plan.md §0.5）
   [ ] 已知道 GPU 型號與 VRAM 大小
   [ ] 已理解「全參數微調不可行，本專案改用參數高效微調」
   [ ] 已決定 2B 與 4B 都用 bf16 LoRA
   [ ] 硬碟有 150 GB 以上可用空間
--------------------------------------------------------------------""")
''')

md(r"""
## §2 建立並檢查 `.venv`（PyTorch CUDA 12.8 + FLA）

**預估時間：15–45 分鐘**（幾乎全部花在下載 PyTorch 與 Unsloth）

> **這一步已經幫你做好了。**
> 專案根目錄下已經有 `scripts/setup_venv.ps1`，它會依序安裝：
> PyTorch cu128 → 核心相依 → FLA（`triton-windows` + `flash-linear-attention[cuda]`）
> → Unsloth → bitsandbytes → 評估工具 → Jupyter kernel。
>
> **已經建立過就跳到 §2.3 驗證**。若你要重建，執行 §2.1。

### 為什麼 PyTorch 一定要指定 `cu128`？

預設的 `pip install torch` 很可能裝到 **CPU 版本**，訓練時完全用不到 GPU，
而且**不會報錯**，只會慢到讓你以為程式壞了。所以一定要用官方索引：

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
""")

code(r'''
#@title §2.1 建立 .venv 與安裝套件（$ 約 15–45 分鐘，僅需執行一次）{display-mode: "form"}
# ============================================================
# 環境：PowerShell（不是 Python）
# 預估時間：15–45 分鐘（下載 PyTorch ~2.5 GB + Unsloth 相依）
#
# 這格會呼叫專案內既有的 scripts/setup_venv.ps1。
# 若你的 .venv 已經建立完成，可以跳過這格，直接跑 §2.3 驗證。
# ============================================================
import os, subprocess, sys, time
from pathlib import Path

if not globals().get("RUN_SETUP", True):
    print("⏭️  RUN_SETUP = False，跳過 §2.1（請確認 .venv 已存在，接著跑 §2.3 驗證）")
else:
    ROOT = Path.cwd()
    while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
        ROOT = ROOT.parent
    print("專案根目錄 :", ROOT)

    script = ROOT / "scripts" / "setup_venv.ps1"
    print("安裝腳本   :", script, "存在" if script.exists() else "不存在")

    venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
    REBUILD = False        # ← 想強制重建環境時改成 True

    if venv_py.exists() and not REBUILD:
        print(f"\n[跳過] 偵測到既有的 .venv：{venv_py}")
        print("       若要重建，請把上面那行的 REBUILD 改成 True 再執行本格。")
    else:
        print("\n開始安裝（輸出會同步寫入 logs\\setup\\*.log）…\n")
        t0 = time.time()
        p = subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            encoding="utf-8", errors="replace"
        )
        for line in p.stdout:
            print(line, end="")
        p.wait()
        print(f"\n安裝流程結束，exit={p.returncode}，耗時 {(time.time()-t0)/60:.1f} 分鐘")

    print("""
--------------------------------------------------------------------
 備註：若某個步驟失敗（例如 fla 編譯不過），**不要停下來**。
 fla 只是加速器，沒有它模型照樣能訓練（見下方 §3）。
 只要 torch / transformers / unsloth / peft / trl 有裝好，就能完成整個專案。
--------------------------------------------------------------------""")
''')

md(r"""
### §2.2 安裝內容一覽

| 類別 | 套件 | 為什麼需要 |
| --- | --- | --- |
| **深度學習核心** | `torch`, `torchvision`（cu128） | CUDA 反向傳播，訓練的唯一運算引擎 |
| **模型／訓練** | `transformers>=5.0.0` | **Qwen3.5 必須用 v5**，舊版不認得 `qwen3_5` 架構 |
| | `peft`, `trl`, `accelerate`, `datasets` | LoRA、SFTTrainer、裝置管理、資料集 |
| **微調框架** | `unsloth`, `unsloth_zoo` | 官方支援 Qwen3.5，省 VRAM 且內建 GGUF 匯出 |
| **線性注意力加速** | `triton-windows`, `flash-linear-attention[cuda]` | Gated DeltaNet 的快速核心（**選配**） |
| **量化** | `bitsandbytes` | 只有 QLoRA／E7 實驗需要 |
| **評估** | `math-verify[antlr4_13_2]`, `lmstudio` | 數學等價性判定、LM Studio 客戶端 |
| **繪圖** | `pandas`, `matplotlib`, `scipy` | 對照表、效率前緣圖、McNemar 檢定 |

> **`triton-windows` 是什麼？** 官方 `triton` 在 PyPI 上**沒有 Windows 輪子**，
> 所以 Windows 上要用社群維護的同名分支。它必須**先裝**，`flash-linear-attention` 才裝得起來。

> **`flash-linear-attention[cuda]` 的 `[cuda]` 不能省。**
> 0.5 版之後，裸裝 `flash-linear-attention` **不會**拉進 torch／triton，
> 裝完之後 `import fla` 會失敗。
""")

code(r'''
#@title §2.3 環境驗證（$ < 1 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv（**請把這個 Notebook 的 kernel 切成 .venv**，見下方說明）
# 預估時間：< 1 分鐘
#
# 這一格是整個階段 A 的驗收。四件事必須成立：
#   1. 直譯器是 .venv 裡的那個（不是系統 Python）
#   2. torch.cuda.is_available() == True
#   3. 能對 GPU 做 bf16 矩陣乘法（真的能算）
#   4. transformers 認得 qwen3_5 架構
# ============================================================
import sys, os, importlib
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

print("=" * 64)
print(" §2.3 環境驗證")
print("=" * 64)
print(f" 直譯器        : {sys.executable}")
IN_VENV = ".venv" in sys.executable.replace("/", "\\").lower()
print(f" 在 .venv 內？ : {'是 ✅' if IN_VENV else '否 ❌'}")
if not IN_VENV:
    print("""
 [!] 你的 kernel 不是 .venv 的 Python。
     請執行下面指令註冊並切換 kernel，然後重新跑本格：

        .\\.venv\\Scripts\\python.exe -m ipykernel install --user `
            --name dsai4207 --display-name "Python (.venv DSAI4207)"

     然後在 Notebook 右上角把 kernel 選成「Python (.venv DSAI4207)」。
""")

print("\n" + "-" * 64)
print(" 套件版本")
print("-" * 64)
REQUIRED = ["torch", "transformers", "unsloth", "peft", "trl", "datasets", "accelerate"]
OPTIONAL = ["fla", "triton", "bitsandbytes", "math_verify", "lmstudio", "scipy",
            "pandas", "matplotlib", "safetensors", "tokenizers"]
missing_required, missing_optional = [], []

for group, names in [("必要", REQUIRED), ("選配", OPTIONAL)]:
    print(f"\n [{group}]")
    for n in names:
        try:
            m = importlib.import_module(n)
            print(f"   {n:16s} {getattr(m, '__version__', 'ok')}")
        except Exception as e:
            print(f"   {n:16s} ❌ {type(e).__name__}: {str(e)[:70]}")
            (missing_required if group == "必要" else missing_optional).append(n)

print("\n" + "-" * 64)
print(" GPU 可用性")
print("-" * 64)
CUDA_OK = False
try:
    import torch
    print(f" torch 版本    : {torch.__version__}")
    print(f" CUDA 編譯版本 : {torch.version.cuda}")
    CUDA_OK = torch.cuda.is_available()
    print(f" CUDA 可用     : {'是 ✅' if CUDA_OK else '否 ❌'}")
    if CUDA_OK:
        p = torch.cuda.get_device_properties(0)
        print(f" GPU           : {p.name}")
        print(f" VRAM          : {p.total_memory/1024**3:.1f} GB")
        print(f" 算力 (capability): {torch.cuda.get_device_capability(0)}")
        # ★ 真的算一次 bf16 矩陣乘法，確認不只是「看得到」而已
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        c = a @ b
        torch.cuda.synchronize()
        print(f" bf16 矩陣乘法 : 成功（輸出 dtype = {c.dtype}）✅")
        del a, b, c
        torch.cuda.empty_cache()
    else:
        print("""
 [!] CUDA 不可用。最常見的原因是裝到了 CPU 版 PyTorch。
     請在 .venv 內重新安裝：

        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
""")
except Exception as e:
    print(f" torch 匯入失敗：{type(e).__name__}: {e}")

print("\n" + "-" * 64)
print(" Qwen3.5 架構支援")
print("-" * 64)
QWEN35_OK = False
try:
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    print(f" transformers  : {transformers.__version__}")
    QWEN35_OK = "qwen3_5" in CONFIG_MAPPING
    print(f" 認得 qwen3_5？ : {'是 ✅' if QWEN35_OK else '否 ❌（請 pip install \"transformers>=5.0.0\"）'}")
    try:
        from transformers.utils.import_utils import is_flash_linear_attention_available
        fla_ok = is_flash_linear_attention_available()
        print(f" FLA 加速可用？ : {'是 ✅' if fla_ok else '否 ⚠️（模型仍可訓練，但較慢、較吃記憶體）'}")
    except Exception as e:
        print(f" FLA 檢測失敗   : {type(e).__name__}: {str(e)[:80]}")
except Exception as e:
    print(f" transformers 匯入失敗：{type(e).__name__}: {e}")

print("\n" + "=" * 64)
ok = IN_VENV and CUDA_OK and QWEN35_OK and not missing_required
print(f" 總結：{'✅ 環境就緒，可以繼續 §3' if ok else '❌ 還有問題，請先修正上面標示 ❌ 的項目'}")
if missing_optional:
    print(f" （選配缺少：{', '.join(missing_optional)}；不影響主流程）")
print("=" * 64)
''')

md(r"""
## §3 加速器檢測：`fla` 有沒有生效？

**預估時間：< 1 分鐘**

這一節**不訓練任何東西**，只回答一個問題：
**「我的 Gated DeltaNet 是在走快速核心，還是退回慢速 PyTorch？」**

這是很有價值的資訊，因為它會直接影響你的訓練時間估計。
""")

code(r'''
#@title §3.1 檢測 FLA 是否生效 {display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：< 1 分鐘
# ============================================================
import importlib, os, sys

print("=" * 64)
print(" FLA / Triton 狀態檢測")
print("=" * 64)

# --- 1. fla 本身 ---
fla_mod = None
try:
    import fla
    fla_mod = fla
    print(f" fla 版本      : {getattr(fla, '__version__', '未知')}")
    print(f" fla 位置      : {os.path.dirname(fla.__file__)}")
except Exception as e:
    print(f" fla 匯入失敗  : {type(e).__name__}: {str(e)[:120]}")

# --- 2. transformers 的判定（這才是真正決定行為的開關）---
print("\n [transformers 的判定]")
try:
    from transformers.utils.import_utils import is_flash_linear_attention_available
    FLA_ACTIVE = is_flash_linear_attention_available()
    print(f" is_flash_linear_attention_available() = {FLA_ACTIVE}")
except Exception as e:
    FLA_ACTIVE = False
    print(f" 判定失敗：{type(e).__name__}: {str(e)[:120]}")

# --- 3. 關鍵算子能不能匯入 ---
print("\n [關鍵算子匯入測試]")
for mod, names in [
    ("fla.modules", ["FusedRMSNormGated"]),
    ("fla.ops.gated_delta_rule", ["chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"]),
]:
    try:
        m = importlib.import_module(mod)
        for n in names:
            print(f"   ✅ {mod}.{n}" if hasattr(m, n) else f"   ⚠️ {mod}.{n} 不存在")
    except Exception as e:
        print(f"   ❌ {mod}: {type(e).__name__}: {str(e)[:100]}")

# --- 4. triton ---
print("\n [triton]")
try:
    import triton
    print(f"   triton 版本 : {triton.__version__}")
    print(f"   triton 位置 : {os.path.dirname(triton.__file__)}")
    try:
        import torch
        print(f"   torch       : {torch.__version__}")
    except Exception:
        pass
except Exception as e:
    print(f"   ❌ triton 匯入失敗：{type(e).__name__}: {str(e)[:100]}")

# --- 5. 結論 ---
print("\n" + "=" * 64)
if FLA_ACTIVE:
    print(""" 結論：FLA 加速【已生效】✅
    Gated DeltaNet 會走 Triton 融合核心，訓練較快、較省 VRAM。
    第一次執行時會即時編譯核心，可能多花 5–20 分鐘，這是正常現象。""")
else:
    print(""" 結論：FLA 加速【未生效】⚠️ —— 但這不是阻礙
    transformers 找不到可用的 fla（或版本 < 0.2.2），
    因此 Gated DeltaNet 會退回【純 PyTorch 算子】：
      • 模型可以正常訓練與推論，結果完全有效
      • 代價是較慢、活化值較吃記憶體
      • 若 VRAM 吃緊，優先降低 max_seq_length（見 §7.4）

    想補救的話，在 .venv 內執行（順序很重要，triton 要先裝）：
        pip install triton-windows
        pip install --no-build-isolation "flash-linear-attention[cuda]"
    然後【重啟 kernel】再跑本格。""")
print("=" * 64)

# 把結果存起來，後面幾節會用到
FLA_ACTIVE_GLOBAL = bool(FLA_ACTIVE)

print("""
--------------------------------------------------------------------
 💡 為什麼這裡顯示的 fla 路徑可能是 unsloth_zoo\\_vendored\\fla？
   Unsloth 會自帶一份 fla（本機實測為 0.5.1）並掛到 sys.path 前面，
   因為 unsloth 必須最先匯入。這不影響判定：
   transformers 只要求「import fla 成功且版本 ≥ 0.2.2」，
   兩份都符合，所以 FLA 加速照常生效。

 ⚠️ 兩個「FLA 是否生效」的判斷層次不一樣，別混淆：

   (a) transformers 層：is_flash_linear_attention_available()
       → 決定 Qwen3.5 的 Gated DeltaNet 要不要用 fla 的算子。

   (b) unsloth_zoo 層：載入模型時可能印出
       「The fast path is not available because one of the required library
        is not installed. Falling back to torch implementation.」
       這是在講 fla 的『快速路徑』不完整（通常缺 causal_conv1d），
       不代表模型不能用 —— 它會用較慢但等價的實作。

 結論：**看到 (b) 的訊息仍然可以正常訓練與評估**，只是比較慢。
 想補齊的話：
     pip install triton-windows
     pip install --no-build-isolation "flash-linear-attention[cuda]"
     pip install causal-conv1d   # 官方無 Windows 輪子，裝不起來可忽略
 然後【重啟 kernel】再跑本格。
--------------------------------------------------------------------""")
''')

md(r"""
---
# 階段 B：模型與資料

## §4 下載基礎模型與資料集

**預估時間：20–90 分鐘**（依網速；兩顆模型合計約 14 GB）

> **本機狀態提醒**：你的 Hugging Face 快取位於
> `C:\Users\ellis\.cache\huggingface\`，其中 `models--Qwen--Qwen3.5-2B-Base` 與
> `models--Qwen--Qwen3.5-4B-Base` **目錄已存在但只有 `refs`，權重尚未下載**。
> 所以 §4.1 仍然需要真的下載一次。

### 為什麼路徑要固定？

`project_plan.md` §9.2 指出，**GGUF 轉換會用到合併後的完整權重**，
而合併需要原始 base 權重。把模型放在專案內的 `models/` 底下，
可以讓後面所有步驟的路徑都可預測，也方便清理。
""")

code(r'''
#@title §4.1 下載 Qwen3.5-2B-Base 與 4B-Base（$ 20–90 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv（使用 huggingface_hub）
# 預估時間：20–90 分鐘（2B 約 4.6 GB、4B 約 9.4 GB）
#
# 可選：若在中國大陸網路很慢，把下面 USE_MIRROR 改成 True 走鏡像。
#       注意 HF_ENDPOINT 必須在 import huggingface_hub 之前設定才有效！
# ============================================================
import os
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

if not globals().get("RUN_DOWNLOAD", True):
    print("⏭️  RUN_DOWNLOAD = False，跳過下載。")
    print("    若尚未下載，請把 §0 的 RUN_DOWNLOAD 改成 True。")
else:
    USE_MIRROR = False       # ← 網路慢就改成 True（改用 https://hf-mirror.com）
    if USE_MIRROR:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[提醒] 已啟用鏡像端點：", os.environ["HF_ENDPOINT"])

    MODELS_DIR = ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)

    MODEL_IDS = {
        "Qwen3.5-2B-Base": "Qwen/Qwen3.5-2B-Base",
        "Qwen3.5-4B-Base": "Qwen/Qwen3.5-4B-Base",
    }
    # 只想先做 2B 的話，把 4B 那行註解掉可以省 9.4 GB 與半小時
    DOWNLOAD = ["Qwen3.5-2B-Base", "Qwen3.5-4B-Base"]

    from huggingface_hub import snapshot_download

    for key in DOWNLOAD:
        repo = MODEL_IDS[key]
        dest = MODELS_DIR / key
        print("\n" + "=" * 64)
        print(f" 下載 {repo}")
        print(f" 目標：{dest}")
        print("=" * 64)
        try:
            path = snapshot_download(
                repo_id=repo,
                local_dir=str(dest),
                # 純文字數學微調【不需要視覺權重】，但 vision 檔在 repo 內無法單獨排除，
                # 這裡保留完整下載以確保 config 與 processor 一致。
                max_workers=4,
            )
            print(f" ✅ 完成：{path}")
        except Exception as e:
            print(f" ❌ 失敗：{type(e).__name__}: {str(e)[:300]}")
            print("    提示：可改用 CLI → hf download " + repo + f' --local-dir "{dest}"')

    # ---------- 驗證：權重檔真的有下載下來嗎？ ----------
    # snapshot_download 對「已存在」的檔案會直接跳過，所以一定要檢查 safetensors。
    print("\n" + "=" * 64)
    print(" 下載結果驗證（重點：safetensors 權重檔）")
    print("=" * 64)
    all_ok = True
    for key in DOWNLOAD:
        d = MODELS_DIR / key
        st = sorted(d.rglob("*.safetensors")) if d.exists() else []
        n_st = len(st)
        total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0
        n_files = sum(1 for f in d.rglob("*") if f.is_file()) if d.exists() else 0
        size_gb = total / 2**30
        if n_st == 0:
            all_ok = False
            print(f" ❌ {key:20s} {size_gb:6.2f} GB  檔案 {n_files} 個  "
                  f"safetensors=0 → 【權重沒有下載成功】")
        else:
            print(f" ✅ {key:20s} {size_gb:6.2f} GB  檔案 {n_files} 個  "
                  f"safetensors={n_st} 檔")
    if not all_ok:
        print("""
 ⚠️ 有模型下載不完整（只有 config / tokenizer，沒有權重）。
   這通常是 Hugging Face 對匿名請求限流造成的。
   請改用 CLI 重試（會顯示真正的進度與錯誤）：
       hf download Qwen/Qwen3.5-4B-Base --local-dir "models\\Qwen3.5-4B-Base"
   或先設定 HF_TOKEN 以提高速率上限，再重新執行本格。
""")
    else:
        print(" ✅ 兩個模型的權重都已就位。")
''')

md(r"""
### §4.2 驗證模型規格：確認你下載的是「真的 Qwen3.5」

**預估時間：< 1 分鐘**

這一格會驗證 `project_plan.md` §3.7 的驗收標準。
**最重要的是確認 layer 數、`layer_types`（混合架構）、以及 `<think>` 的 token id。**
""")

code(r'''
#@title §4.2 驗證模型規格與 <think> token（$ < 1 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：< 1 分鐘（只讀 config / tokenizer，不載入權重）
# ============================================================
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
MODELS_DIR = ROOT / "models"

EXPECT = {   # project_plan.md §3.1 的正確規格
    "Qwen3.5-2B-Base": {"layers": 24, "hidden": 2048},
    "Qwen3.5-4B-Base": {"layers": 32, "hidden": 2560},
}

for key, exp in EXPECT.items():
    path = MODELS_DIR / key
    print("=" * 64)
    print(f" {key}   ({path})")
    print("=" * 64)
    if not path.exists():
        print("  ⚠️ 目錄不存在，跳過（請先跑 §4.1）\n")
        continue
    try:
        cfg = AutoConfig.from_pretrained(str(path))
        tok = AutoTokenizer.from_pretrained(str(path))

        # 這是 VLM：語言層設定在 text_config 底下
        tcfg = getattr(cfg, "text_config", cfg)
        n_layers = tcfg.num_hidden_layers
        hidden   = tcfg.hidden_size
        layer_types = getattr(tcfg, "layer_types", None) or []
        mtp = getattr(tcfg, "mtp_num_hidden_layers", None)

        print(f"  model_type          : {cfg.model_type}")
        print(f"  architectures       : {getattr(cfg, 'architectures', None)}")
        print(f"  語言層數            : {n_layers}   (預期 {exp['layers']}) "
              f"{'✅' if n_layers == exp['layers'] else '❌'}")
        print(f"  隱藏維度            : {hidden}   (預期 {exp['hidden']}) "
              f"{'✅' if hidden == exp['hidden'] else '❌'}")
        print(f"  詞表大小            : {tcfg.vocab_size}")
        print(f"  MTP 層數            : {mtp}   ← GGUF 轉換要 --no-mtp 的原因")
        if layer_types:
            n_lin = sum(1 for t in layer_types if "linear" in str(t))
            n_full = sum(1 for t in layer_types if "full" in str(t))
            print(f"  layer_types（前 8）  : {list(layer_types)[:8]}")
            print(f"  混合架構統計        : 線性注意力 {n_lin} 層 / 完整注意力 {n_full} 層")
        print(f"  tie_word_embeddings : {getattr(tcfg, 'tie_word_embeddings', None)}")

        # ★ 思考標籤：一律用 token id 驗證，不要用字串比對
        print("\n  [思考標籤 token]")
        for s in ["<think>", "</think>"]:
            tid = tok.convert_tokens_to_ids(s)
            unk = tok.unk_token_id
            ok = (tid is not None and tid != unk)
            print(f"    {s:10s} id = {tid}   {'✅ 有效' if ok else '❌ 無效（會等於 unk）'}")
        print("    參考：Qwen3.5 的 <think> / </think> 應為 248068 / 248069")
        print(f"    unk_token_id = {tok.unk_token_id}（用來判斷上面是否真的有效）")
    except Exception as e:
        print(f"  ❌ 讀取失敗：{type(e).__name__}: {str(e)[:300]}")
    print()
''')

md(r"""
## §5 資料集：從 `OpenR1-Math-220k` 建立 SFT 訓練檔

**預估時間：20–60 分鐘**（下載 default config 約 2.15 GB，之後轉檔 5–15 分鐘）

### 為什麼選 `open-r1/OpenR1-Math-220k`？

| 條件 | 這個資料集為什麼符合 |
| --- | --- |
| **必須含推理痕跡** | 每題附 2–4 條 DeepSeek-R1 的 `<think>…</think>` 完整推理 |
| **答案可自動驗證** | 提供 `correctness_math_verify`（**逐條**的官方驗證結果）可過濾品質 |
| **大小可負擔** | `default` config 為 93,733 筆（下載約 2.15 GB） |

### 已核實的欄位與型別（**重要，別寫錯**）

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `problem` | `str` | 題目 |
| `answer` | `str` | 標準答案，**格式不統一**，只當備援 |
| `generations` | **`List[str]`** | ⚠️ **是字串清單，不是 dict 清單**，直接對每條用 `</think>` 切 |
| `correctness_math_verify` | `List[bool]` | 逐條的官方驗證結果，**挑選樣本的關鍵** |
| `is_reasoning_complete` | `List[bool]` | 逐條是否推理完整（被截斷的為 `False`） |
| `finish_reasons` | `List[str]` | 生成結束原因，**可能是 `null`** |
| `source`, `problem_type` | `str` | 題目來源與領域，可做分領域分析 |

> ⚠️ **常見錯誤**：網路上有些教學說 `generations` 是含 `reasoning_content` 鍵的 dict 清單。
> **在本資料集上是錯的**，它是純字串清單。下面的程式碼會**同時容忍兩種格式**，
> 但你應該知道正確答案是 `List[str]`。

### 訓練格式（單一 `text` 欄位）

```
<|im_start|>system
{系統提示}<|im_end|>
<|im_start|>user
{題目}<|im_end|>
<|im_start|>assistant
<think>
{推理過程}
</think>

The final answer is \boxed{...}.<|im_end|>
```

**為什麼自己手動拼字串，而不用 `apply_chat_template`？**
因為 2B 與 4B 的聊天模板對 `enable_thinking` 的判斷邏輯**剛好相反**
（`project_plan.md` §3.5 的踩雷警告）。自己拼字串可以完全避開這個陷阱。
""")

code(r'''
#@title §5.1 下載資料集並轉成 SFT 格式（$ 20–50 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：20–50 分鐘（下載 ~2.15 GB 佔大部分；轉檔 5–15 分鐘）
# ============================================================
import json, os, re, time
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

RUN_DATA_ON = globals().get("RUN_DATA", True)
DATA_RAW  = ROOT / "data" / "raw";       DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC = ROOT / "data" / "processed"; DATA_PROC.mkdir(parents=True, exist_ok=True)

# 若輸出檔已存在且筆數足夠，就跳過下載與轉檔 —— 讓本格可以重複執行而不重複花時間
SKIP_IF_READY = True

DATASET_ID  = "open-r1/OpenR1-Math-220k"
DATASET_CFG = "default"     # default=93,733 筆 / extended=131,396 / all=225,129
N_SAMPLES   = 5000          # ← 先用 5,000 筆跑通；要放大就改這裡
OUT_PATH    = DATA_PROC / f"sft_{N_SAMPLES//1000}k.jsonl" if N_SAMPLES >= 1000 else DATA_PROC / f"sft_{N_SAMPLES}.jsonl"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning expert. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# ---------- 工具函式 ----------
def extract_boxed(text):
    """抓出最後一個 \\boxed{...}。用括號計數，因為巢狀括號會讓 regex 失效。"""
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    depth, start = 0, None
    for i in range(idx, len(text)):
        ch = text[i]
        if ch == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i]
    return None

def get_text(gen):
    """generations 的每一條可能是 str，也可能是含 reasoning_content 的 dict。
    兩種都容忍（但本資料集正確答案是 str）。"""
    if isinstance(gen, str):
        return gen
    if isinstance(gen, dict):
        for k in ("reasoning_content", "content", "text", "generation"):
            if gen.get(k):
                return gen[k]
    return None

def parse_generation(gen):
    """把 <think>...</think> 拆成 (推理, 最終答案段)。"""
    txt = get_text(gen)
    if not txt:
        return None, None
    if "</think>" in txt:
        reasoning, final = txt.rsplit("</think>", 1)
        return reasoning.replace("<think>", "", 1).strip(), final.strip()
    return txt.replace("<think>", "", 1).strip(), ""

def pick_best_generation(ex):
    """挑一條「通過驗證 + 推理完整」的生成結果。這是提升訓練品質最關鍵的一步。"""
    gens = ex.get("generations") or []
    ok   = ex.get("correctness_math_verify") or []
    comp = ex.get("is_reasoning_complete") or []
    # 優先：驗證通過 + 推理完整
    for i, g in enumerate(gens):
        if i < len(ok) and ok[i] and (i >= len(comp) or comp[i]):
            return g
    # 次選：只要驗證通過
    for i, g in enumerate(gens):
        if i < len(ok) and ok[i]:
            return g
    # 最後手段：第一條
    return gens[0] if gens else None

# ---------- 下載 ----------
# 讓本格可重複執行：若輸出檔已存在且筆數足夠，就跳過下載與轉檔（省時間）
_ready = 0
if OUT_PATH.exists():
    with open(OUT_PATH, encoding="utf-8") as _f:
        for _ in _f:
            _ready += 1
if RUN_DATA_ON and SKIP_IF_READY and _ready >= N_SAMPLES:
    print(f"⏭️  已存在 {OUT_PATH.name}（{_ready:,} 筆），跳過下載與轉檔。")
    print("    想強制重建，請把上面的 SKIP_IF_READY 改成 False。")
    ds = None
elif not RUN_DATA_ON:
    print("⏭️  RUN_DATA = False，跳過資料集下載與轉檔。")
    print(f"    data/processed 內目前有 {_ready:,} 筆的 {OUT_PATH.name}")
    ds = None
else:
    print("=" * 64); print(f" 載入資料集 {DATASET_ID} [{DATASET_CFG}]"); print("=" * 64)
    from datasets import load_dataset
    t0 = time.time()
    try:
        ds = load_dataset(DATASET_ID, DATASET_CFG, split="train")
        print(f" ✅ 載入完成，{len(ds):,} 筆，耗時 {(time.time()-t0)/60:.1f} 分鐘")
        print(f" 欄位：{ds.column_names}")
    except Exception as e:
        print(f" ❌ 載入失敗：{type(e).__name__}: {str(e)[:400]}")
        raise

if ds is not None:
    # ---------- 抽樣檢查（先看清楚資料長什麼樣子）----------
    print("\n" + "-" * 64); print(" 第一筆樣本結構檢查"); print("-" * 64)
    ex0 = ds[0]
    g0 = ex0.get("generations") or []
    print(f" generations 型別        : {type(g0).__name__}, 長度 {len(g0)}")
    if g0:
        print(f" generations[0] 型別     : {type(g0[0]).__name__}")
        if isinstance(g0[0], dict):
            print(f" generations[0] 的鍵     : {list(g0[0].keys())}")
        else:
            print(f" generations[0] 前 120 字: {str(g0[0])[:120]!r}")
    print(f" correctness_math_verify : {ex0.get('correctness_math_verify')}")
    print(f" is_reasoning_complete   : {ex0.get('is_reasoning_complete')}")

    # ---------- 轉檔 ----------
    print("\n" + "-" * 64); print(f" 轉檔中（目標 {N_SAMPLES:,} 筆）"); print("-" * 64)
    written = skipped_nogen = skipped_nobox = 0
    t0 = time.time()
    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for ex in ds:
            if written >= N_SAMPLES:
                break
            problem = ex.get("problem")
            gen = pick_best_generation(ex)
            if not problem or not gen:
                skipped_nogen += 1
                continue
            reasoning, _final = parse_generation(gen)
            if not reasoning:
                skipped_nogen += 1
                continue
            # 最終答案：優先取生成內容中的 \boxed{}，其次才用 answer 欄位
            final = extract_boxed(get_text(gen)) or ex.get("answer")
            if not final:
                skipped_nobox += 1
                continue
            # 若推理段裡沒有 \boxed{}，在結尾補一句，確保格式一致
            if "\\boxed" not in reasoning:
                reasoning = reasoning.rstrip() + f"\n\nTherefore, the answer is \\boxed{{{final}}}."
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{problem}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n{reasoning.strip()}\n</think>\n\n"
                f"The final answer is \\boxed{{{final}}}.<|im_end|>\n"
            )
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1

    print(f" ✅ 已寫出 {written:,} 筆 → {OUT_PATH}")
    print(f"    跳過（無生成／無推理）：{skipped_nogen:,}   （無答案）：{skipped_nobox:,}")
    print(f"    檔案大小：{OUT_PATH.stat().st_size/2**20:.1f} MB，耗時 {(time.time()-t0)/60:.1f} 分鐘")
''')

code(r'''
#@title §5.2 抽樣人工檢查格式（$ < 1 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：< 1 分鐘
#
# project_plan.md §4.5 驗收標準：抽樣 3 筆確認有 <think>、有 \boxed{}
# ============================================================
import json
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
OUT_PATH = next((ROOT / "data" / "processed").glob("sft_*.jsonl"), None)
print("檢查檔案：", OUT_PATH)

if OUT_PATH is None:
    raise FileNotFoundError(
        "找不到 data/processed/sft_*.jsonl。\n"
        "請先執行 §5.1（並確認 §0 的 RUN_DATA = True）。")

lines = OUT_PATH.read_text(encoding="utf-8").splitlines()
print(f"總行數：{len(lines):,}\n")

# --- JSON 合法性 ---
bad = 0
for i, l in enumerate(lines):
    try:
        json.loads(l)
    except Exception as e:
        bad += 1
        if bad <= 3:
            print(f" ❌ 第 {i} 行不是合法 JSON：{e}")
print(f"JSON 合法性：{'✅ 全部合法' if bad == 0 else f'❌ {bad} 行有問題'}\n")

# --- 格式統計 ---
n = len(lines)
stats = {"think_open": 0, "think_close": 0, "boxed": 0, "im_end": 0, "system": 0, "user": 0}
for l in lines:
    t = json.loads(l)["text"]
    stats["think_open"]  += "<think>" in t
    stats["think_close"] += "</think>" in t
    stats["boxed"]       += "\\boxed" in t
    stats["im_end"]      += "<|im_end|>" in t
    stats["system"]      += "<|im_start|>system" in t
    stats["user"]        += "<|im_start|>user" in t
print("格式合規率：")
for k, v in stats.items():
    print(f"  {k:12s} {v/n:6.1%}  {v:,}/{n:,}")

# --- 顯示一筆完整樣本 ---
print("\n" + "=" * 64)
print(" 範例（第 0 筆，完整內容）")
print("=" * 64)
print(json.loads(lines[0])["text"])
''')

code(r'''
#@title §5.3 Token 長度分佈 → 決定 max_seq_length（$ 5–20 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：5–20 分鐘（要 tokenize 全部樣本，樣本越多越久）
#
# project_plan.md §4.3 第 5 點：思維鏈動輒數千 token，
# 若有大量樣本超過 max_seq_length，模型的思考會被硬生生切斷，訓練效果會很差。
# 【重要】這一步決定 §7 的 max_seq_length，請務必執行。
# ============================================================
import json, time, numpy as np
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

PATH_JSONL = next((ROOT / "data" / "processed").glob("sft_*.jsonl"), None)
MODEL_DIR  = ROOT / "models" / "Qwen3.5-2B-Base"

if PATH_JSONL is None:
    raise FileNotFoundError(
        "找不到 data/processed/sft_*.jsonl。請先執行 §5.1（§0 的 RUN_DATA 要為 True）。")
if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"找不到 tokenizer：{MODEL_DIR}。請先執行 §4.1 下載 2B 模型。")

print("載入 tokenizer 進行長度統計 …（只載 tokenizer，不載權重）")
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

texts = [json.loads(l)["text"] for l in PATH_JSONL.read_text(encoding="utf-8").splitlines()]
print(f"樣本數：{len(texts):,}，開始 tokenize …")

t0 = time.time()
lens = []
for i, t in enumerate(texts):
    lens.append(len(tok(t, add_special_tokens=False)["input_ids"]))
    if (i + 1) % 1000 == 0:
        print(f"  {i+1:,}/{len(texts):,}  已用 {(time.time()-t0)/60:.1f} 分鐘")
lens = np.array(lens)
print(f"完成，耗時 {(time.time()-t0)/60:.1f} 分鐘\n")

print("=" * 64); print(" Token 長度分佈"); print("=" * 64)
for q in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{q:<3d}: {np.percentile(lens, q):8.0f}")
print(f"  平均: {lens.mean():8.0f}")
print(f"  最大: {lens.max():8.0f}")

print("\n" + "-" * 64); print(" 各候選 max_seq_length 的截斷率"); print("-" * 64)
CANDIDATES = [1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384]
trunc = {}
for c in CANDIDATES:
    frac = float((lens > c).mean())
    trunc[c] = frac
    note = "  ← 可接受（≤10%）" if frac <= 0.10 else ("  ← 勉強（≤20%）" if frac <= 0.20 else "")
    print(f"  max_seq_length={c:6d} → 會被截斷 {frac:6.2%}{note}")

# ★ 選擇邏輯：依序放寬門檻，只在真的沒有選擇時才退回最大值。
#   （舊版寫死退回 2048 是錯的 —— 當所有候選都超標時，
#     2048 可能是最糟的選擇，會把絕大多數樣本的推理硬生生切斷。）
def pick_seq_len(cands, table):
    for limit in (0.05, 0.10, 0.20, 0.30):
        for c in cands:
            if table[c] <= limit:
                return c, limit
    return max(cands), None

BEST_SEQ, BEST_LIMIT = pick_seq_len(CANDIDATES, trunc)
print(f"\n建議：max_seq_length = {BEST_SEQ}")
if BEST_LIMIT is not None:
    print(f"  這是第一個讓截斷率 ≤ {BEST_LIMIT:.0%} 的候選值。")
else:
    print(f"  ⚠️ 即使到 {BEST_SEQ}，截斷率仍有 {trunc[BEST_SEQ]:.1%} —— 這個資料集的思維鏈非常長。")

print(f"""
 為什麼截斷率這麼高？
   OpenR1-Math-220k 的 DeepSeek-R1 推理痕跡動輒數千 token
   （本資料集實測：中位數約 {int(np.median(lens)):,}、p90 約 {int(np.percentile(lens,90)):,}、
     最大 {int(lens.max()):,}）。project_plan.md §4.3 的 2048 是保守起點，
   **但用在這份資料上會切掉八成以上的推理**，訓練效果會很差。

 本專案的建議（16 GB VRAM 實測可行）
   • 2B：max_seq_length = 4096～8192
   • 4B：max_seq_length = 4096（保守）～8192（需約 6 GB 餘裕）
   權重才是 VRAM 的主要消耗（2B 約 3.5 GB、4B 約 9.5 GB），
   活化值在 seq 8192 時也只有約 1.3 GB，**放得下**。

 若你想更嚴格控制截斷率
   在 §5.1 之後過濾掉過長的樣本（只留 token 數 ≤ 上限的樣本），
   代價是資料量變少 —— 這是「保真」與「資料量」之間的取捨，請在報告中說明。
""")
''')

md(r"""
### §5.4 資料去汙染（Decontamination）

**預估時間：5–20 分鐘**

**為什麼非做不可？** `OpenR1-Math-220k` 是從 `NuminaMath` 衍生的，
而 `NuminaMath` 又大量取材自 **MATH** 與 **GSM8K**——
也就是說，**你的訓練題很可能就出現在測試集裡**。

若不做去汙染，「微調後 GSM8K 從 20% 衝到 60%」可能只是**背題**，不是學會推理。
""")

code(r'''
#@title §5.4 去汙染檢查（$ 5–20 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：5–20 分鐘（要下載 MATH-500 與 GSM8K 測試集）
#
# project_plan.md §7.4：用題目「正規化後的字串」比對訓練集與測試集。
# ============================================================
import json, re, time
from pathlib import Path
from datasets import load_dataset

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
PATH_JSONL = next((ROOT / "data" / "processed").glob("sft_*.jsonl"), None)
if PATH_JSONL is None:
    raise FileNotFoundError(
        "找不到 data/processed/sft_*.jsonl。請先執行 §5.1（§0 的 RUN_DATA 要為 True）。")
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

def normalize(s: str) -> str:
    """把題目正規化成指紋：轉小寫、去多餘空白、只留英數字與空白。"""
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s.strip()

# ---------- 建立訓練集題目指紋 ----------
print("建立訓練集題目指紋 …")
train_fps = set()
n_lines = 0
for line in PATH_JSONL.read_text(encoding="utf-8").splitlines():
    n_lines += 1
    text = json.loads(line)["text"]
    m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.S)
    if m:
        train_fps.add(normalize(m.group(1)))
print(f"  訓練樣本 {n_lines:,} 筆 → 不重複題目指紋 {len(train_fps):,} 個")

# ---------- 比對各測試集 ----------
BENCHES = [
    ("GSM8K (test)",     "openai/gsm8k",            "main",    "test", "question"),
    ("MATH-500 (test)",  "HuggingFaceH4/MATH-500",  None,      "test", "problem"),
]
overlap_report = {}
print("\n" + "=" * 64); print(" 汙染檢查結果"); print("=" * 64)
for label, repo, cfg, split, qcol in BENCHES:
    try:
        t0 = time.time()
        ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
        hits = sum(1 for ex in ds if normalize(ex.get(qcol, "")) in train_fps)
        rate = hits / max(len(ds), 1)
        overlap_report[label] = {"n": len(ds), "overlap": hits, "rate": rate}
        flag = "⚠️ 有汙染" if hits else "✅ 無重疊"
        print(f"  {label:18s} {hits:4d}/{len(ds):4d} = {rate:6.2%}   {flag}"
              f"   （耗時 {(time.time()-t0)/60:.1f} 分鐘）")
    except Exception as e:
        print(f"  {label:18s} ❌ 檢查失敗：{type(e).__name__}: {str(e)[:150]}")

# ---------- 存檔 ----------
out = RESULTS / "contamination_report.json"
out.write_text(json.dumps(overlap_report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n已寫入 {out}")

print("""
--------------------------------------------------------------------
 如何在報告中呈現（project_plan.md §7.4）
 1. 明確揭露重疊比例，以及去汙染後重跑的结果
 2. 強調【最強的證據是「同題同提示」的比較】——
    base 與微調模型在完全相同的題目與提示下比較，
    這個比較是無條件有效的，因為汙染對兩者影響相同。
--------------------------------------------------------------------""")
''')

md(r"""
### §5.5 建立煙霧測試資料集（50 筆）

**預估時間：< 1 分鐘**

`project_plan.md` §6.2 的核心原則：**永遠先用極小資料量確認整條流程能跑完**。
這可以省下你好幾個小時的無效等待。
""")

code(r'''
#@title §5.5 建立 50 筆煙霧測試資料（$ < 1 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：< 1 分鐘
# ============================================================
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

SRC = next((ROOT / "data" / "processed").glob("sft_*.jsonl"), None)
DST = ROOT / "data" / "processed" / "smoke.jsonl"

if SRC is None:
    raise FileNotFoundError(
        "找不到 data/processed/sft_*.jsonl。請先執行 §5.1（§0 的 RUN_DATA 要為 True）。")

lines = SRC.read_text(encoding="utf-8").splitlines()
DST.write_text("\n".join(lines[:50]) + "\n", encoding="utf-8")
print(f"來源：{SRC}  ({len(lines):,} 筆)")
print(f"產出：{DST}  ({len(lines[:50])} 筆)")
print("用途：§6 訓練流程煙霧測試、§10 QLoRA 崩潰實驗")
''')

md(r"""
---
# 階段 C：訓練

## §6 共用的訓練函式

**預估時間：< 1 分鐘**

為了避免 §7、§8、§10 各寫一次訓練迴圈，這裡先把「載入模型 → 掛 LoRA → 訓練 → 量測 VRAM」
包成一個可重複呼叫的函式。

### 三個關鍵設計決定

**（1）用 `FastVisionModel` 並開啟 `text_only=True`**

Qwen3.5 是視覺語言模型（VLM），`project_plan.md` 用的是 `FastVisionModel`；
但 Unsloth 的 Qwen3.5 官方微調文件寫的是 `FastLanguageModel`。
**兩者差別很大，而且實測結果決定了本 Notebook 的選擇：**

| 載入方式 | 是否建立視覺塔 | 可訓練參數（2B, r=16） | 載入後 VRAM | 可控制微調範圍 |
| --- | --- | --- | --- | --- |
| `FastLanguageModel` | ✅ 有 | 1.0334%（含 192 個視覺張量） | 4.14 GB | ❌ 無法排除視覺層 |
| **`FastVisionModel(text_only=True)`** | **❌ 完全跳過** | **0.5765%（視覺張量 0 個）** | **3.52 GB** | ✅ 有 `finetune_*` 旗標 |

**所以本 Notebook 用 `FastVisionModel(text_only=True)`**：少約 0.6 GB、
少訓練一半的參數量，而且視覺層完全不會被動到。

**（2）`finetune_vision_layers=False`**

Qwen3.5 內建視覺編碼器（24 個 block）。數學資料沒有圖片，
訓練視覺層只會浪費 VRAM 並可能造成**災難性遺忘**。**這一定要關掉。**

**（3）`bf16=True`，絕對不用 `fp16`**

Qwen3.5 的 Gated DeltaNet 層在 fp16 下**會產生 NaN 梯度**，
嚴重到 Unsloth 直接把 `qwen3_5` 放進 `FORCE_FLOAT32` 清單。

> 📌 **關於「只對回答算 loss」**：因為 `text_only=True` 載入的模型沒有視覺設定，
> Unsloth 的 `UnslothVisionDataCollator` 會直接拒絕
> （`UnslothVisionDataCollator is only for image models!`）。
> 本 Notebook 會**自動偵測並改用 `trl` 內建的 `assistant_only_loss`**，
> 效果等價：兩者都只對 assistant 的回答計算 loss。
""")

code(r'''
#@title §6.1 匯入訓練框架並偵測可用介面（$ < 1 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：< 1 分鐘（Unsloth 首次匯入會做一些環境檢查，約 10–60 秒）
#
# ★★ 重要：unsloth 必須在 trl / transformers / peft 之前匯入 ★★
#    否則 Unsloth 的加速修補不會生效，會出現：
#      UserWarning: Unsloth should be imported before [trl, transformers, peft]
#    後果是「可以跑，但比較慢、也可能比較吃記憶體」。
# ============================================================
import os, sys, json, time, inspect
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

print("=" * 64); print(" 匯入訓練框架"); print("=" * 64)

# ---- 1. 先匯入 unsloth ----
import unsloth
print(f" unsloth 版本 : {unsloth.__version__}")

# ---- 2. 再匯入其餘框架 ----
import torch
import trl, peft, transformers
print(f" trl          : {trl.__version__}")
print(f" peft         : {peft.__version__}")
print(f" transformers : {transformers.__version__}")
print(f" torch        : {torch.__version__}  (CUDA {torch.version.cuda})")
print(f" CUDA 可用    : {torch.cuda.is_available()}")

# ---- 3. 自動偵測可用的 Unsloth 介面 ----
FAST_LM = FAST_VM = None
try:
    from unsloth import FastLanguageModel as FAST_LM
    print(" ✅ 可用介面：FastLanguageModel")
except Exception as e:
    print(f" ⚠️ FastLanguageModel 不可用：{type(e).__name__}: {str(e)[:100]}")
try:
    from unsloth import FastVisionModel as FAST_VM
    print(" ✅ 可用介面：FastVisionModel")
except Exception as e:
    print(f" ⚠️ FastVisionModel 不可用：{type(e).__name__}: {str(e)[:100]}")

if FAST_LM is None and FAST_VM is None:
    raise RuntimeError("找不到任何可用的 Unsloth 模型介面，請檢查安裝。")

# ---- 4. 檢查介面能力，決定要用哪一個 ----
# Qwen3.5 是 VLM（視覺語言模型）。數學資料沒有圖片，所以我們想「只微調語言層」。
# FastVisionModel 支援 text_only=True，載入時就【直接跳過視覺塔】，
# 這是最省 VRAM、也最乾淨的做法（實測 2B 少約 0.6 GB，且視覺層完全不參與訓練）。
def _has_param(cls, fn, name):
    """檢查方法是否接受某個具名參數。

    注意：FastVisionModel.get_peft_model 是 (*args, **kwargs) 轉發函式，
    所以 inspect.signature 看不到 finetune_* 旗標。
    因此這裡只用它判斷 text_only（那是真的具名參數），
    finetune_* 則改用 try/except 在執行時決定。
    """
    try:
        return name in inspect.signature(getattr(cls, fn)).parameters
    except Exception:
        return False

VM_HAS_TEXT_ONLY = FAST_VM is not None and _has_param(FAST_VM, "from_pretrained", "text_only")
VM_HAS_FLAGS = False   # 由執行時的 try/except 決定，見 load_model_with_lora()

print(f"""
 介面能力檢查
   FastVisionModel.from_pretrained 有 text_only 參數 : {VM_HAS_TEXT_ONLY}
   （finetune_* 旗標無法用 signature 偵測，會在載入時自動嘗試）
""")

# 決策：能用 FastVisionModel + text_only 就用它（可跳過視覺塔、並可控制微調範圍）
if FAST_VM is not None and VM_HAS_TEXT_ONLY:
    ENGINE, ENGINE_NAME, TEXT_ONLY = FAST_VM, "FastVisionModel", True
elif FAST_LM is not None:
    ENGINE, ENGINE_NAME, TEXT_ONLY = FAST_LM, "FastLanguageModel", False
else:
    ENGINE, ENGINE_NAME, TEXT_ONLY = FAST_VM, "FastVisionModel", False

print(f" → 本 Notebook 將使用：{ENGINE_NAME}（text_only={TEXT_ONLY}）")
print("""
 為什麼選 FastVisionModel + text_only=True？
   1. text_only=True 讓載入時【完全不建立視覺塔】——
      數學資料沒有圖片，訓練視覺層只會浪費 VRAM 並可能造成災難性遺忘。
      實測 2B：含視覺塔約 4.14 GB，純文字只要 3.52 GB。
   2. 它同時提供 finetune_vision_layers=False 等旗標，
      可以進一步精確控制「只調注意力」或「只調 FFN」（D1/D2 實驗）。
""")
''')

code(r'''
#@title §6.2 載入模型 + 掛 LoRA 的函式（定義，不執行）{display-mode: "form"}

# ============================================================
# 定義 load_model_with_lora()：載入模型、掛上 LoRA/DoRA、回報參數量
# 預估時間：定義本身 < 1 秒（真正載入模型要 1–5 分鐘）
# ============================================================
import gc, json, time
from pathlib import Path

def _free_vram():
    """清掉前一組實驗殘留的 GPU 記憶體。跑多組實驗時一定要做。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

def _call_from_pretrained(model_dir, max_seq_length, load_in_4bit, load_in_16bit,
                          custom_bnb=None):
    """依可用介面嘗試載入模型，回傳 (model, tokenizer_or_processor)。

    嘗試順序：
      1. 本 Notebook 選定的 ENGINE（通常 FastVisionModel），並帶 text_only=True
      2. 同一個 ENGINE，但不帶 text_only
      3. 另一個介面（FastLanguageModel / FastVisionModel）
    """
    base_kwargs = dict(
        model_name=str(model_dir),
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_16bit=load_in_16bit,
        full_finetuning=False,
    )
    if custom_bnb is not None:
        # ★ 關鍵：強制載入原始檢查點，避免被導向 Unsloth 的預量化鏡像
        base_kwargs["quantization_config"] = custom_bnb
        base_kwargs["load_in_4bit"] = False
        base_kwargs["use_exact_model_name"] = True

    candidates = []
    if TEXT_ONLY:
        candidates.append((ENGINE_NAME, ENGINE, dict(base_kwargs, text_only=True)))
    candidates.append((ENGINE_NAME, ENGINE, dict(base_kwargs)))
    for alt_name, alt in [("FastLanguageModel", FAST_LM), ("FastVisionModel", FAST_VM)]:
        if alt is not None and alt is not ENGINE:
            candidates.append((alt_name, alt, dict(base_kwargs)))

    last_err = None
    for name, eng, kw in candidates:
        try:
            print(f"   嘗試 {name}（text_only={kw.get('text_only', False)}）…")
            model, tok = eng.from_pretrained(**kw)
            print(f"   ✅ 載入成功（{name}）")
            return model, tok, name, bool(kw.get("text_only", False))
        except TypeError as e:
            print(f"   ↷ {name} 不接受這組參數：{str(e)[:120]}")
            last_err = e
        except Exception as e:
            print(f"   ❌ {name} 載入失敗：{type(e).__name__}: {str(e)[:200]}")
            last_err = e
    raise RuntimeError(f"所有載入方式都失敗。最後錯誤：{last_err}")

def load_model_with_lora(
    model_dir,
    rank=16,
    max_seq_length=2048,
    quant="bf16",              # bf16 | dora | qlora
    attention_only=False,      # D1：只微調注意力層
    mlp_only=False,            # D2：只微調 FFN 層
    custom_bnb=None,           # 給 QLoRA 用的自訂 BitsAndBytesConfig
):
    """載入 Qwen3.5 並掛上 LoRA / DoRA。回傳 (model, tokenizer, meta dict)。"""
    _free_vram()
    model_dir = Path(model_dir)
    use_dora  = (quant == "dora")
    use_qlora = (quant == "qlora")

    if use_qlora:
        if custom_bnb is None:
            load_in_4bit, load_in_16bit = True, False
        else:
            load_in_4bit, load_in_16bit = True, False
        print(" ⚠️  quant=qlora：Qwen3.5 的 4-bit 路徑是【已知有問題】的（見 §9）")
    else:
        load_in_4bit, load_in_16bit = False, True

    print(f" 載入模型：{model_dir.name}")
    print(f"   quant={quant}  rank={rank}  max_seq_length={max_seq_length}")
    print(f"   load_in_4bit={load_in_4bit}  load_in_16bit={load_in_16bit}")

    t0 = time.time()
    model, tokenizer, used_engine, used_text_only = _call_from_pretrained(
        model_dir, max_seq_length, load_in_4bit, load_in_16bit, custom_bnb)
    if model is None:
        raise RuntimeError(
            f"{used_engine}.from_pretrained 回傳 model=None。"
            f"（load_in_4bit={load_in_4bit}, load_in_16bit={load_in_16bit}, "
            f"text_only={used_text_only}）")
    print(f"   載入完成，耗時 {(time.time()-t0)/60:.1f} 分鐘")
    print(f"   tokenizer/processor 型別：{type(tokenizer).__name__}")
    if torch.cuda.is_available():
        print(f"   載入後 VRAM：{torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ---------- 掛 LoRA / DoRA ----------
    finetune_attn = not mlp_only
    finetune_mlp  = not attention_only
    print(f" 掛上 {'DoRA' if use_dora else 'LoRA'}："
          f"attention={finetune_attn}  mlp={finetune_mlp}")

    peft_kwargs = dict(
        r=rank,
        lora_alpha=rank,          # 慣例：alpha == r
        lora_dropout=0,
        bias="none",
        use_rslora=False,
        random_state=3407,
        target_modules="all-linear",
        use_gradient_checkpointing="unsloth",
        max_seq_length=max_seq_length,
    )
    if use_dora:
        peft_kwargs["use_dora"] = True

    # finetune_* 旗標只存在於 FastVisionModel 的介面上，
    # 而且因為它是 (*args, **kwargs) 轉發函式，無法用 signature 偵測，
    # 所以這裡用 try/except 實測決定。
    vision_flags = dict(
        finetune_vision_layers=False,      # ★ 數學不需要視覺，一定要關
        finetune_language_layers=True,
        finetune_attention_modules=finetune_attn,
        finetune_mlp_modules=finetune_mlp,
    )
    if used_engine == "FastVisionModel":
        peft_kwargs.update(vision_flags)

    base_targets = "all-linear"
    if attention_only:
        base_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif mlp_only:
        base_targets = ["gate_proj", "up_proj", "down_proj"]

    def _apply(eng, kwargs):
        return eng.get_peft_model(model, **kwargs)

    attempts = [dict(peft_kwargs)]
    if used_engine == "FastVisionModel":
        # 退路 1：去掉 finetune_* 旗標
        no_flags = {k: v for k, v in peft_kwargs.items() if k not in vision_flags}
        attempts.append(no_flags)
    # 退路 2：去掉 use_dora
    attempts.append({k: v for k, v in attempts[-1].items() if k != "use_dora"})
    # 退路 3：改用明確的 target_modules 清單
    last = dict(attempts[-1]); last["target_modules"] = base_targets
    attempts.append(last)
    attempts.append({k: v for k, v in last.items() if k != "max_seq_length"})

    # ★ 注意：這裡【絕對不能】把變數取名為 model ——
    #   那會覆蓋掉上面載入好的 model，導致 get_peft_model 收到 None
    #   （症狀是 AttributeError: 'NoneType' object has no attribute 'named_modules'）。
    peft_model = None
    used_kwargs, err = None, None
    attempts = [a for a in attempts if a]
    for i, kw in enumerate(attempts, 1):
        try:
            m = _apply(ENGINE, kw)
            if m is None:
                raise TypeError("get_peft_model 回傳 None")
            peft_model = m
            used_kwargs = kw
            print(f"   ✅ get_peft_model 成功（第 {i} 種參數組合）"
                  f"{'，已關閉視覺層' if 'finetune_vision_layers' in kw else ''}")
            break
        except TypeError as e:
            err = e
            print(f"   ↷ 第 {i} 種參數組合不相容：{str(e)[:110]}")
        except Exception as e:
            err = e
            print(f"   ❌ 第 {i} 種參數組合失敗：{type(e).__name__}: {str(e)[:150]}")
            print("      → 這不是參數名稱問題，改用下一種組合沒有意義，直接中止")
            raise
    if peft_model is None:
        raise RuntimeError(f"get_peft_model 全部失敗：{err}")
    model = peft_model

    # ---------- 統計可訓練參數 ----------
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    dtypes = {}
    for p in model.parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1

    # 確認視覺層真的沒被訓練
    vis_names = [n for n, p in model.named_parameters()
                 if p.requires_grad and ("visual" in n or "vision" in n)]
    lin_names = [n for n, p in model.named_parameters()
                 if p.requires_grad and "linear_attn" in n]

    meta = {
        "quant": quant, "rank": rank, "max_seq_length": max_seq_length,
        "engine": used_engine, "text_only": used_text_only,
        "finetune_vision_layers": "finetune_vision_layers" in used_kwargs,
        "trainable_params": trainable, "total_params": total,
        "trainable_pct": trainable / max(total, 1),
        "dtype_histogram": dtypes,
        "trainable_vision_tensors": len(vis_names),
        "trainable_linear_attn_tensors": len(lin_names),
    }
    print(f"\n 可訓練參數：{trainable:,} / {total:,} = {100*meta['trainable_pct']:.3f}%")
    print(f" 參數 dtype 分佈：{dtypes}")
    print(f" 可訓練的視覺層張量數：{len(vis_names)} （應為 0）")
    print(f" 可訓練的 linear_attn 張量數：{len(lin_names)}")
    if len(vis_names) > 0:
        print(f" ⚠️ 有 {len(vis_names)} 個視覺張量在訓練，例如：{vis_names[:3]}")
        print("    → 建議改用 FastVisionModel + text_only=True 載入")
    if torch.cuda.is_available():
        print(f" 掛完 adapter 後 VRAM：{torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return model, tokenizer, meta

print("已定義 load_model_with_lora()")
''')

code(r'''
#@title §6.3 訓練函式 train_lora()（定義，不執行）{display-mode: "form"}

# ============================================================
# 定義 train_lora()：跑 SFT、量測 VRAM 峰值、存 adapter 與 vram_report.json
# 預估時間：定義本身 < 1 秒
# ============================================================
from trl import SFTTrainer, SFTConfig

def _build_collator(model, tokenizer, train_on_responses_only=True):
    """建立能「只對 assistant 回答算 loss」的 collator。

    SFT 的正確做法是把「使用者問題」那段的 loss 遮掉，只對 assistant 的回答計算 loss，
    避免模型浪費容量去背誦提問。

    ⚠️ 實測重點：UnslothVisionDataCollator 會檢查模型是否有視覺設定，
       若我們用 text_only=True 載入（沒有視覺塔），它會直接拒絕：
           TypeError: UnslothVisionDataCollator is only for image models!
       這時就改用 trl 內建的 assistant_only_loss（效果等價，見 train_lora）。

    實測簽名：UnslothVisionDataCollator(model, processor, ..., train_on_responses_only,
                                        instruction_part, response_part, force_match)
    """
    if not train_on_responses_only:
        return None
    try:
        from unsloth.trainer import UnslothVisionDataCollator
    except Exception as e:
        print(f" ℹ️ 無法匯入 UnslothVisionDataCollator（{type(e).__name__}）")
        print("    → 改用 trl 內建的 assistant_only_loss（效果等價）")
        return None

    common = dict(
        train_on_responses_only=True,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        force_match=False,     # 找不到才不會硬性報錯，比較安全
    )
    # 只有 processor= 是正確的參數名；tokenizer= 只是為了相容舊版才試
    for key in ("processor", "tokenizer"):
        kw = dict(model=model, **common)
        kw[key] = tokenizer
        try:
            c = UnslothVisionDataCollator(**kw)
            print(f" ✅ 已建立 UnslothVisionDataCollator（{key}=）")
            print("    只對 assistant 回答計算 loss（問題段落被遮掉）")
            return c
        except TypeError as e:
            msg = str(e)
            if "only for image models" in msg:
                print(" ℹ️ 目前是 text_only 載入（沒有視覺塔），"
                      "UnslothVisionDataCollator 不適用")
                print("    → 改用 trl 內建的 assistant_only_loss（效果等價）")
                return None
            print(f" ↷ collator 不接受 {key}=：{msg[:110]}")
        except Exception as e:
            print(f" ⚠️ collator 建立失敗：{type(e).__name__}: {str(e)[:130]}")
            break
    print(" ℹ️ 退回 trl 內建的 assistant_only_loss 遮罩方式")
    return None

def train_lora(
    model, tokenizer, data_path, out_dir,
    epochs=1.0, lr=2e-4, batch=1, accum=8,
    max_seq_length=2048, optim="adamw_8bit",
    logging_steps=5, save_steps=200,
    train_on_responses_only=True, max_steps=-1,
    resume=False,
):
    """跑 LoRA SFT 訓練，並把 VRAM 峰值寫進 out_dir/vram_report.json。"""
    from datasets import load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files=str(data_path), split="train")
    print(f" 訓練樣本數：{len(ds):,}   輸出目錄：{out_dir}")

    collator = _build_collator(model, tokenizer, train_on_responses_only)
    if collator is not None:
        assistant_only = False   # 已由 collator 處理，避免雙重遮罩
    else:
        # 退路：用 trl 內建的 assistant_only_loss。
        # ⚠️ 實測：trl 要求這個選項只能用在「對話式」資料集（要有 messages 欄位）；
        #    我們的資料是單一 text 欄位，若硬開會直接 ValueError。
        is_conversational = hasattr(ds, "column_names") and "messages" in ds.column_names
        assistant_only = bool(train_on_responses_only) and is_conversational
        if train_on_responses_only and not is_conversational:
            print(" ⚠️ 資料集不是對話式（沒有 messages 欄位），無法使用 "
                  "assistant_only_loss。")
            print("    → 本次訓練的 loss 會涵蓋整段文字（含使用者問題）。")
            print("    → 這是可接受的退化，但請在報告中註明。")
            print("    （bf16 路徑通常能建立 Unsloth collator，不會走到這裡）")

    cfg_kwargs = dict(
        output_dir=str(out_dir),
        max_seq_length=max_seq_length,   # trl 0.24 仍支援（已實測）
        dataset_text_field="text",
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        warmup_ratio=0.03,
        num_train_epochs=epochs,
        max_steps=max_steps,          # >0 時用來做煙霧測試（只跑幾步）
        learning_rate=lr,
        lr_scheduler_type="cosine",
        optim=optim,
        bf16=True,                    # ★ 絕對不要改成 fp16（NaN 梯度）
        fp16=False,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        seed=3407,
        report_to="none",
    )
    if assistant_only:
        try:
            cfg_kwargs["assistant_only_loss"] = True
        except Exception:
            pass
    try:
        args = SFTConfig(**cfg_kwargs)
    except TypeError as e:
        # 某些版本的 SFTConfig 不認得 max_seq_length / assistant_only_loss
        print(f" ↷ SFTConfig 參數不相容，移除後重試：{str(e)[:120]}")
        for k in ["max_seq_length", "assistant_only_loss"]:
            cfg_kwargs.pop(k, None)
        args = SFTConfig(**cfg_kwargs)
    if assistant_only:
        print(" ✅ 已啟用 trl 的 assistant_only_loss（只對回答算 loss）")
    elif collator is None:
        print(" ℹ️ 未套用任何 response-only 遮罩：loss 涵蓋整段文字。")

    trainer_kwargs = dict(model=model, train_dataset=ds, args=args,
                          data_collator=collator)
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
        print(" ✅ SFTTrainer 已建立（processing_class=）")
    except TypeError as e:
        print(f" ↷ processing_class 不相容（{str(e)[:90]}），改用 tokenizer=")
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume if resume else None)
    minutes = (time.time() - t0) / 60
    print(f"\n ✅ 訓練完成，耗時 {minutes:.1f} 分鐘")

    # ---------- VRAM 峰值（本專案的重要實驗數據）----------
    report = {
        "out_dir": str(out_dir), "data": str(data_path),
        "n_samples": len(ds), "epochs": epochs, "lr": lr,
        "batch": batch, "accum": accum,
        "effective_batch": batch * accum,
        "max_seq_length": max_seq_length, "optim": optim,
        "train_minutes": round(minutes, 2),
    }
    if torch.cuda.is_available():
        report["peak_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated()/1024**3, 3)
        report["peak_memory_reserved_gb"]  = round(torch.cuda.max_memory_reserved()/1024**3, 3)
        print(f"    VRAM 峰值（allocated）：{report['peak_memory_allocated_gb']:.2f} GB")
        print(f"    VRAM 峰值（reserved） ：{report['peak_memory_reserved_gb']:.2f} GB")

    # ---------- 儲存 ----------
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "vram_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    adapter 與 vram_report.json 已存到 {out_dir}")
    return trainer, report

def record_meta(out_dir, meta, extra=None):
    """把可訓練參數比例等資訊併進 vram_report.json。"""
    out_dir = Path(out_dir)
    p = out_dir / "vram_report.json"
    rep = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    rep.update(meta)
    if extra:
        rep.update(extra)
    p.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep

print("已定義 train_lora() 與 record_meta()")
''')

md(r"""
## §7 煙霧測試與正式訓練

**這是整個專案最花時間的階段。**

### 執行順序建議（`project_plan.md` §5.4）

```
第一批（先求跑通）：E1 → A1 → A2 → E2
第二批（核心對照）：A3 → A4 → C1 → C3 → B1 → B3
第三批（擴大範圍）：D1 → D2 → D4 → E3 → E4
第四批（進階／有時間）：E7 → E5 → E6 → B4 → F 系列
```

> 💡 **務必先跑 §7.1 的煙霧測試與 §7.3 的 E1（2B bf16 LoRA rank 16）。**
> 它是你的「已知正確」基準線：如果 E1 都跑不出合理結果，
> 那問題在你的資料或流程，不在量化。
> **先建立基準線，再引入變因**，這是除錯的基本原則。
""")

code(r'''
#@title §7.1 煙霧測試：50 筆 × 只跑 10 步（$ 3–10 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv（需要 GPU）
# 預估時間：3–10 分鐘（含首次載入模型 1–5 分鐘）
#
# project_plan.md §6.2：永遠先用極小資料量確認整條流程能跑完。
# 這一步會用 50 筆資料、只跑 10 個 optimizer step，
# 目的是驗證「載入 → tokenize → 前向 → 反向 → 存檔」全通。
# ============================================================
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

SMOKE_DATA = ROOT / "data" / "processed" / "smoke.jsonl"
SMOKE_OUT  = ROOT / "outputs" / "smoke"
MODEL_2B   = ROOT / "models" / "Qwen3.5-2B-Base"

if not globals().get("RUN_TRAIN", True):
    print("⏭️  RUN_TRAIN = False，跳過煙霧測試。")
else:
    assert SMOKE_DATA.exists(), "找不到 smoke.jsonl，請先跑 §5.5"
    assert MODEL_2B.exists(),   "找不到 2B 模型，請先跑 §4.1"

    import gc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, tokenizer, meta = load_model_with_lora(
        MODEL_2B, rank=16, max_seq_length=1024, quant="bf16",
    )

    trainer, report = train_lora(
        model, tokenizer, SMOKE_DATA, SMOKE_OUT,
        epochs=1.0, max_steps=10,          # ← 只跑 10 步
        max_seq_length=1024, save_steps=10,
    )

# ---------- 立刻做一次生成，確認模型活著 ----------
if not globals().get("RUN_TRAIN", True):
    print("（已跳過煙霧測試，不做生成檢查）")
else:
  print("\n" + "=" * 64); print(" 煙霧測試生成檢查"); print("=" * 64)
  try:
    prompt = ("<|im_start|>system\nYou are a mathematical reasoning expert. "
              "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
              "<|im_start|>user\nWhat is 12 * 7?<|im_end|>\n"
              "<|im_start|>assistant\n")
    enc = tokenizer(prompt, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"].to(model.device),
            attention_mask=enc["attention_mask"].to(model.device),
            max_new_tokens=192, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen = tokenizer.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
    print("模型輸出：\n" + gen[:900])
    print(f"\n 格式檢查：<think>={'<think>' in gen}  "
          f"</think>={'</think>' in gen}  boxed={'\\boxed' in gen}")
    print(" （只訓練 10 步的模型輸出沒有推理內容是正常的，這裡只確認『能生成』）")
  except Exception as e:
    print(f" ⚠️ 生成檢查失敗：{type(e).__name__}: {str(e)[:300]}")
    print("""
 附註：若錯誤是 'NoneType' object is not iterable 且提到 config.architectures，
 那是 Unsloth 的視覺推論路徑在 text_only 模式下沒有 architectures 可用。
 本格已改用【標準 tokenizer + model.generate】，不走該路徑，所以不應再發生。
""")

  # 釋放記憶體，讓下一格有乾淨的 VRAM
  del model, trainer
  _free_vram()
  if torch.cuda.is_available():
    print(f"\n 清理後 VRAM：{torch.cuda.memory_allocated()/1024**3:.2f} GB")
  print("\n✅ 煙霧測試結束。若這格成功，代表環境沒問題，可以開始正式實驗。")
''')

md(r"""
### §7.2 實驗註冊表

把每個實驗組的設定寫成一份表，之後所有訓練與評估都從這張表驅動，
避免「手打指令打錯參數」這種低級錯誤。

| 編號 | 模型 | 方法 | rank | 資料 | 說明 |
| --- | --- | --- | --- | --- | --- |
| **E1** | 2B | **bf16 LoRA** | 16 | 5k | **★ 品質基準線，最先跑這個** |
| E2 | 4B | **bf16 LoRA** | 16 | 5k | **4B 主力方法**（約 9.5 GB） |
| E3 | 2B | DoRA | 16 | 5k | 幾乎零額外成本 |
| E4 | 4B | DoRA | 16 | 5k | 同上，在 4B 上驗證 |
| B1/B3/B4 | 2B | bf16 LoRA | 4/64/128 | 5k | rank 掃描 |
| C1/C3 | 2B | bf16 LoRA | 16 | 1k/20k | 資料量掃描 |
| D1/D2 | 2B | bf16 LoRA | 16 | 5k | 只調注意力／只調 FFN |
| E5/E6 | 2B/4B | QLoRA（修正版） | 16 | 5k | 進階，需 §10.2 的修正 |
| **E7** | 2B | QLoRA **不跳過 linear_attn** | 16 | smoke | **受控失敗實驗**（見 §10） |
""")

code(r'''
#@title §7.2 實驗註冊表（$ < 1 秒）{display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：< 1 秒
# ============================================================
EXPERIMENTS = {
    # 編號 : (模型資料夾, quant, rank, 資料檔, 額外旗標, 說明)
    "E1_2b_bf16_r16":   dict(model="Qwen3.5-2B-Base", quant="bf16", rank=16,
                             data="sft_5k",      extra={}, desc="2B bf16 LoRA（★ 基準線）"),
    "E2_4b_bf16_r16":   dict(model="Qwen3.5-4B-Base", quant="bf16", rank=16,
                             data="sft_5k",      extra={}, desc="4B bf16 LoRA（4B 主力）"),
    "E3_2b_dora_r16":   dict(model="Qwen3.5-2B-Base", quant="dora", rank=16,
                             data="sft_5k",      extra={}, desc="2B DoRA"),
    "E4_4b_dora_r16":   dict(model="Qwen3.5-4B-Base", quant="dora", rank=16,
                             data="sft_5k",      extra={}, desc="4B DoRA"),
    "B1_2b_bf16_r4":    dict(model="Qwen3.5-2B-Base", quant="bf16", rank=4,
                             data="sft_5k",      extra={}, desc="2B rank 4（極低秩）"),
    "B3_2b_bf16_r64":   dict(model="Qwen3.5-2B-Base", quant="bf16", rank=64,
                             data="sft_5k",      extra={}, desc="2B rank 64"),
    "B4_2b_bf16_r128":  dict(model="Qwen3.5-2B-Base", quant="bf16", rank=128,
                             data="sft_5k",      extra={}, desc="2B rank 128"),
    "C1_2b_bf16_1k":    dict(model="Qwen3.5-2B-Base", quant="bf16", rank=16,
                             data="sft_1k",      extra={}, desc="2B 1k 樣本"),
    "C3_2b_bf16_20k":   dict(model="Qwen3.5-2B-Base", quant="bf16", rank=16,
                             data="sft_20k",     extra={}, desc="2B 20k 樣本"),
    "D1_2b_attn_only":  dict(model="Qwen3.5-2B-Base", quant="bf16", rank=16,
                             data="sft_5k", extra={"attention_only": True}, desc="只調注意力層"),
    "D2_2b_mlp_only":   dict(model="Qwen3.5-2B-Base", quant="bf16", rank=16,
                             data="sft_5k", extra={"mlp_only": True},      desc="只調 FFN 層"),
}

print("=" * 92)
print(" 實驗註冊表")
print("=" * 92)
print(f" {'編號':20s} {'模型':16s} {'方法':6s} {'rank':>4s}  {'資料':8s} 說明")
print("-" * 92)
for k, v in EXPERIMENTS.items():
    print(f" {k:20s} {v['model']:16s} {v['quant']:6s} {v['rank']:>4d}  {v['data']:8s} {v['desc']}")
print("-" * 92)
print(f" 共 {len(EXPERIMENTS)} 組。若要縮減，直接刪掉字典裡的行即可。")
''')

md(r"""
### §7.3 ★ E1：2B + bf16 LoRA（品質基準線）

**預估時間：0.5–1.5 小時**（5,000 筆、1 epoch；首次會多花 5–20 分鐘編譯核心）

**這是整個專案最重要的一組實驗。**
如果 E1 跑不出合理結果，問題在你的資料或流程，**不在量化**。

### ⚠️ 序列長度：不要沿用 2048

§5.3 在本資料集上的實測結果（5,000 筆）：

| 統計量 | 值 |
| --- | --- |
| 中位數 | 約 **4,400** token |
| p75 | 約 7,591 token |
| p90 | 約 11,507 token |
| 最大 | 約 20,775 token |

| `max_seq_length` | 截斷率 |
| --- | --- |
| 2048 | **82.1%** ❌ |
| 4096 | 53.6% |
| 6144 | 34.7% |
| **8192** | **21.8%** ← 本 Notebook 預設 |

**`max_seq_length=2048` 會把八成以上樣本的推理硬生生切斷**，
訓練出來的模型會學會「思考到一半就收尾」——這比不訓練更糟。
本書預設 **8192**，2B 只需約 3.5 GB 權重 + 約 1.3 GB 活化值，16 GB 非常寬鬆。

**訓練時該看什麼**（`project_plan.md` §6.4）：

| 指標 | 健康訊號 | 危險訊號 |
| --- | --- | --- |
| `loss` | 穩定下降，最後落在 0.5–1.5 附近 | 完全不動（學習率太低或資料格式錯誤） |
| `grad_norm` | 平穩 | 劇烈震盪或爆衝 |
| VRAM | 穩定 | 逐步上升（洩漏）或 OOM |
""")

code(r'''
#@title §7.3 ★ E1：2B bf16 LoRA 正式訓練（$ 0.5–1.5 小時）{display-mode: "form"}
# ============================================================
# 環境：.venv（需要 GPU，可用 VRAM 約 12 GB 以上）
# 預估時間：2B 約 30–90 分鐘（5k 樣本、1 epoch）
#           首次執行會即時編譯 Triton/Mamba 核心，額外 5–20 分鐘
# ============================================================
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

# ---------- 可調參數 ----------
RUN_E1      = globals().get("RUN_TRAIN", True)   # 由 §0 的 RUN_TRAIN 控制
EPOCHS      = 1.0
LR          = 2e-4
BATCH       = 1             # 16 GB 顯卡的現實值
ACCUM       = 8             # 等效 batch size = 8
# ★ 請依 §5.3 的長度統計調整。本資料集的推理痕跡很長（中位數約 4,400 token），
#   用 2048 會截掉八成以上的推理。16 GB 上 2B 可安心用到 8192。
MAX_SEQ_LEN = 8192
OPTIM       = "adamw_8bit"  # OOM 時改 paged_adamw_8bit

if RUN_E1:
    E1_OUT   = ROOT / "outputs" / "E1_2b_bf16_r16"
    E1_DATA  = next((ROOT / "data" / "processed").glob("sft_*.jsonl"), None)
    MODEL_2B = ROOT / "models" / "Qwen3.5-2B-Base"
    if E1_DATA is None:
        raise FileNotFoundError("找不到 sft_*.jsonl，請先執行 §5.1。")
    if not MODEL_2B.exists():
        raise FileNotFoundError("找不到 2B 模型，請先執行 §4.1。")

    print("=" * 64)
    print(" E1：2B + bf16 LoRA（rank 16）")
    print("=" * 64)
    print(f" 模型      : {MODEL_2B.name}")
    print(f" 資料      : {E1_DATA.name}")
    print(f" max_seq   : {MAX_SEQ_LEN}")
    print(f" 等效 batch: {BATCH} × {ACCUM} = {BATCH*ACCUM}")

    model, tokenizer, meta = load_model_with_lora(
        MODEL_2B, rank=16, max_seq_length=MAX_SEQ_LEN, quant="bf16",
    )
    trainer, report = train_lora(
        model, tokenizer, E1_DATA, E1_OUT,
        epochs=EPOCHS, lr=LR, batch=BATCH, accum=ACCUM,
        max_seq_length=MAX_SEQ_LEN, optim=OPTIM,
    )
    meta.update({"experiment": "E1", "model_name": "Qwen3.5-2B-Base",
                 "description": "2B bf16 LoRA r16"})
    full = record_meta(E1_OUT, meta)
    print("\n" + "=" * 64); print(" vram_report.json"); print("=" * 64)
    print(json.dumps(full, indent=2, ensure_ascii=False))

    # 保留 loss 曲線（畫圖用）
    try:
        hist = [h for h in trainer.state.log_history if "loss" in h]
        (E1_OUT / "train_log.json").write_text(
            json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n 訓練 log 已存：{E1_OUT/'train_log.json'}（{len(hist)} 筆）")
    except Exception as e:
        print(f" log 儲存失敗：{e}")

    del model, trainer
    _free_vram()
else:
    print("RUN_E1 = False，跳過 E1。")
''')

md(r"""
### §7.4 跑其他實驗組（可選，依剩餘時間）

**預估時間：每組 0.5–3 小時**

下面這格接受一個 `RUN_LIST`，你只放**這一批要跑的實驗**就好。
**一次不要跑太多組**，因為中途失敗會浪費很多時間。

**OOM 的處理順序**（`project_plan.md` §6.4，由代價小到大）：

1. 確認 `per_device_train_batch_size` 是 1
2. 提高 `gradient_accumulation_steps` 補回等效 batch size
3. 改用 `--optim paged_adamw_8bit`（把優化器狀態分頁到你的 64 GB RAM）
4. **降低 `max_seq_length`**（2048 → 1536 → 1024）← **最有效**
5. 縮小 rank（16 → 8）
6. 改用只微調注意力層（`attention_only=True`）
7. 最後才考慮**修正版 QLoRA**（見 §10.2）

> ⚠️ **對 LoRA 而言優化器只佔 0.04–0.17 GB**，所以第 3 步幾乎救不了什麼。
> 真正佔記憶體的是**凍結的基礎權重（2 bytes/參數）與活化值**。
""")

code(r'''
#@title §7.4 批次執行其他實驗（$ 每組 0.5–3 小時）{display-mode: "form"}
# ============================================================
# 環境：.venv（需要 GPU）
# 預估時間：2B 每組 0.5–1.5 小時；4B 每組 1.5–3 小時
#           （每組之間會清空 VRAM，所以不會累積）
# ============================================================
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

# ★ 只挑你這一批要跑的。建議第一次只放一組，確認流程順了再放多組。
RUN_LIST = []
if globals().get("RUN_TRAIN", True):
    RUN_LIST = [
        # "E2_4b_bf16_r16",
        # "E3_2b_dora_r16",
        # "B3_2b_bf16_r64",
        # "D1_2b_attn_only",
    ]
else:
    print("⏭️  RUN_TRAIN = False，本格不執行任何實驗。")

MAX_SEQ_LEN = 8192      # ★ 依 §5.3 的長度統計調整（本資料集推理痕跡很長）
OPTIM       = "adamw_8bit"

def resolve_data(tag):
    """把註冊表裡的資料標籤（sft_5k）對應到實際檔案。"""
    p = ROOT / "data" / "processed" / f"{tag}.jsonl"
    if p.exists():
        return p
    if tag == "sft_5k":
        return next((ROOT / "data" / "processed").glob("sft_*.jsonl"), None)
    return None

summary_rows = []
for exp_id in RUN_LIST:
    spec = EXPERIMENTS.get(exp_id)
    if spec is None:
        print(f"⚠️ 註冊表裡沒有 {exp_id}，跳過"); continue

    data_path = resolve_data(spec["data"])
    if data_path is None:
        print(f"⚠️ {exp_id}：找不到資料 {spec['data']}.jsonl，跳過（請先跑 §5.1 產生）")
        continue
    model_dir = ROOT / "models" / spec["model"]
    if not model_dir.exists():
        print(f"⚠️ {exp_id}：找不到模型 {spec['model']}，跳過"); continue

    out_dir = ROOT / "outputs" / exp_id
    print("\n" + "#" * 70)
    print(f"# {exp_id}：{spec['desc']}")
    print(f"# 模型={spec['model']}  方法={spec['quant']}  rank={spec['rank']}  資料={data_path.name}")
    print("#" * 70)

    try:
        model, tokenizer, meta = load_model_with_lora(
            model_dir, rank=spec["rank"], max_seq_length=MAX_SEQ_LEN,
            quant=spec["quant"], **spec["extra"],
        )
        trainer, report = train_lora(
            model, tokenizer, data_path, out_dir,
            epochs=1.0, lr=2e-4, batch=1, accum=8,
            max_seq_length=MAX_SEQ_LEN, optim=OPTIM,
        )
        meta.update({"experiment": exp_id, "model_name": spec["model"],
                     "description": spec["desc"]})
        full = record_meta(out_dir, meta)
        summary_rows.append(full)
        del model, trainer
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n ❌ {exp_id} OOM！請依 §7.4 的順序降低 max_seq_length（最有效）")
        print(f"    {str(e)[:200]}")
        del model
    except Exception as e:
        print(f"\n ❌ {exp_id} 失敗：{type(e).__name__}: {str(e)[:300]}")
    finally:
        _free_vram()
        if torch.cuda.is_available():
            print(f" 清理後 VRAM：{torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ---------- 這一批的 VRAM 對照 ----------
if summary_rows:
    RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
    (RESULTS / "batch_vram_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n" + "=" * 92)
    print(" 這一批的 VRAM × 可訓練參數 對照")
    print("=" * 92)
    print(f" {'實驗':20s} {'方法':6s} {'rank':>4s} {'可訓練%':>9s} {'VRAM峰值(GB)':>13s} {'分鐘':>7s}")
    print("-" * 92)
    for r in summary_rows:
        print(f" {str(r.get('experiment')):20s} {str(r.get('quant')):6s} "
              f"{r.get('rank', 0):>4} {100*r.get('trainable_pct',0):>8.3f}% "
              f"{r.get('peak_memory_allocated_gb', 0):>13.2f} {r.get('train_minutes', 0):>7.1f}")
else:
    print("這一批沒有成功完成的實驗（或 RUN_LIST 是空的）。")
''')

md(r"""
---
# 階段 D：評估

## §8 評估

### §8.0 一個必須先承認的方法論限制

本專案的評估後端是 **LM Studio**，而 **LM Studio 只能載入 GGUF**。
因此：

> **你評估的是「量化後」的模型，而不是訓練出來的 bf16 原版。**

這不是缺陷，而是必須在報告中誠實說明的限制。處理方式：

- 用 **F16 GGUF** 當作「最接近 bf16 原版」的評估版本
- 同時比較 **F16 / Q8_0 / Q4_K_M**，**分離**「微調帶來的效果」與「量化帶來的損失」
- 在報告中明確標示「評估管線為 GGUF，而非 bf16」

### §8.1 最關鍵的觀念：base 模型不能直接跟微調模型比

`Qwen3.5-2B-Base` 是**預訓練模型**，沒有被教過遵循對話格式。
若直接比較，你會得到「微調後大幅進步」的**假象**——其實你只教會了它格式。

所以 base 模型要建立**三條基準線**（`project_plan.md` §7.1）：

| 基準線 | 說明 | 目的 |
| --- | --- | --- |
| **B0：thinking 開啟** | 官方聊天模板 + 允許思考 | 誠實呈現 base 的原始狀態 |
| **B1：thinking 關閉** | 要求直接作答 | 隔離「思考」本身的貢獻 |
| **B2：few-shot** | 提示中給完整示範 | **最公平的對照** |

**主要結論請以「微調後模型」對比「B0/B1/B2 中最好的那一個」為準。**

> 📌 **注意**：Qwen **沒有**公布 `Qwen3.5-2B-Base / 4B-Base` 的數學成績，
> 所以**沒有官方 base 基準線可引用**。模型卡上的數字都屬於後訓練版本。

### §8.2 三層指標（缺一不可）

| 層次 | 指標 | 為什麼要它 |
| --- | --- | --- |
| **1. 正確率** | `math-verify` 的數學等價性判定 | 主要指標（`1/2` 與 `0.5` 視為相同） |
| **2. 格式合規率** | `<think>`、**閉合** `</think>`、`\boxed{}`、答案抽取率、**截斷率** | **區分「真的變強」與「只是學會格式」的關鍵** |
| **3. 效率** | 平均生成 token 數、平均生成時間 | 思考變長不代表變好，常常只是變囉唆 |

> ⚠️ **截斷是評估推理模型時最大的隱藏混淆因子。**
> Qwen 官方建議數學難題給到 **81,920 token** 的思考預算，本機 16 GB 根本不可能。
> 若 `max_tokens` 太小，base 會因為「還沒想完就被切斷」而拿不到分，
> 你就會誤以為微調帶來巨大進步。
> **做法：對 base 與微調模型使用完全相同的 `max_tokens`，並且一定要報告截斷率。**
> 若截斷率超過 20%，那個數字就不可信。
""")

code(r'''
#@title §8.3 修復 math-verify 的 Windows「靜默失敗」問題（$ < 1 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：< 1 分鐘
#
# ★★ 這一格必讀 ★★
# math-verify 在 Windows 上用 multiprocessing 實作逾時，會拋出
#     AttributeError: Can't get local object 'run_func'
# 而預設的 raise_on_error=False 會把它【吞掉】，
# 讓 parse() 回傳空清單、verify() 回傳 False。
# 後果：【模型明明答對了，卻被計為答錯，而且完全沒有錯誤訊息。】
# （對應 issue: https://github.com/huggingface/Math-Verify/issues/79）
#
# 解法：呼叫時明確傳入 None 關閉內建逾時。
# 本格會自動偵測這個問題，並選擇可用的驗證策略。
# ============================================================
GOLD_SAMPLE = r"\frac{1}{2}"     # 標準答案
PRED_SAMPLE = "0.5"              # 模型輸出（數學等價）
EXPECTED    = True               # 這兩個應該被判為「相同」

MATH_VERIFY_MODE = "unavailable"
_parse = _verify = None

try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    try:
        from math_verify import LatexExtractionConfig, ExprExtractionConfig
        PARSE_CFG = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
    except Exception:
        PARSE_CFG = None

    # ---- 策略 1：關閉內建逾時（Windows 的正解）----
    def _try(mode):
        def p(text):
            kw = {"parsing_timeout": None} if mode == "notimeout" else {}
            if PARSE_CFG is not None:
                kw["extraction_config"] = PARSE_CFG
            return _mv_parse(text, **kw)
        def v(gold, pred):
            kw = {"timeout_seconds": None} if mode == "notimeout" else {}
            return _mv_verify(gold, pred, **kw)
        g, pr = p(GOLD_SAMPLE), p(PRED_SAMPLE)
        if not g or not pr:
            return False, "parse() 回傳空清單"
        return bool(v(g, pr)), f"verify() = {v(g, pr)}"

    for mode in ("notimeout", "default"):
        try:
            ok, detail = _try(mode)
            print(f" 策略『{mode}』：{detail}")
            if ok:
                MATH_VERIFY_MODE = mode
                break
        except Exception as e:
            print(f" 策略『{mode}』失敗：{type(e).__name__}: {str(e)[:140]}")
except Exception as e:
    print(f" math_verify 匯入失敗：{type(e).__name__}: {str(e)[:140]}")

print("\n" + "=" * 64)
print(" math-verify 判分邏輯自我測試")
print("=" * 64)
print(f" 標準答案 : {GOLD_SAMPLE!r}")
print(f" 模型輸出 : {PRED_SAMPLE!r}   （數學上等價）")
if MATH_VERIFY_MODE == "unavailable":
    print("""
 ❌ math-verify 不可用或判分錯誤。
    本 Notebook 會自動退回【字串正規化比對】（見 §8.4 的 fallback），
    準確率數字仍然可用，但對「1/2 vs 0.5」這類等價形式會低估。
    建議先嘗試修好：pip install "math-verify[antlr4_13_2]"
""")
else:
    print(f"""
 ✅ math-verify 可用（策略：{MATH_VERIFY_MODE}）
    parse/verify 會正確判定 {GOLD_SAMPLE!r} 與 {PRED_SAMPLE!r} 為相同答案。

 【請務必保留這個設定】：parse(..., parsing_timeout=None)
                            verify(..., timeout_seconds=None)
""")
''')

code(r'''
#@title §8.4 定義評估核心函式（$ < 1 分鐘）{display-mode: "form"}

# ============================================================
# 定義 extract_boxed_final() / normalize_answer() / score_one()
# 預估時間：定義本身 < 1 秒
# ============================================================
import re, unicodedata

def show(text):
    """把 \\think 之類的字串安全地印出來（避免 f-string 跳脫地獄）。"""
    return text

TAG_THINK_OPEN  = "<" + "think" + ">"
TAG_THINK_CLOSE = "<" + "/" + "think" + ">"

def split_after_think(gen):
    """最終答案在【最後一個】閉合思考標籤之後。
    注意：這是 Qwen3.5 的 <think>（用 token id 248068/248069 驗證過），
    不是舊格式。"""
    if TAG_THINK_CLOSE in gen:
        return gen.rsplit(TAG_THINK_CLOSE, 1)[1], True
    return gen, False

def extract_boxed_final(text):
    """抓出最後一個 \\boxed{...} 的內容（用括號計數，處理巢狀括號）。"""
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    depth, start = 0, None
    for i in range(idx, len(text)):
        ch = text[i]
        if ch == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i]
    return None

def normalize_answer(s):
    """退路用的字串正規化（只在 math-verify 不可用時使用）。"""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\[a-zA-Z]+", "", s)          # 去掉 \frac 這類指令
    s = s.replace("{", "").replace("}", "").replace(" ", "")
    s = s.replace("\\", "").replace(",", "")
    # 分數 "1/2" 轉小數以便與 "0.5" 比較
    m = re.fullmatch(r"-?\d+/\d+", s)
    if m:
        try:
            num, den = s.split("/")
            return f"{float(num)/float(den):.6f}".rstrip("0").rstrip(".")
        except Exception:
            pass
    return s.lower()

def score_one(gen, gold):
    """對單一筆生成結果計算：正確與否 + 各項格式指標。

    回傳 dict：correct / has_think / has_close_think / boxed / pred
    """
    ans_part, has_close = split_after_think(gen)
    boxed = extract_boxed_final(ans_part)
    if boxed is None:                      # 退而求其次：整段找
        boxed = extract_boxed_final(gen)

    has_think = (TAG_THINK_OPEN in gen) or has_close
    correct = False
    if boxed is not None:
        if MATH_VERIFY_MODE != "unavailable":
            try:
                kw_p = {"parsing_timeout": None} if MATH_VERIFY_MODE == "notimeout" else {}
                kw_v = {"timeout_seconds": None} if MATH_VERIFY_MODE == "notimeout" else {}
                if PARSE_CFG is not None:
                    kw_p["extraction_config"] = PARSE_CFG
                gold_p = _mv_parse(str(gold), **kw_p)
                pred_p = _mv_parse(str(boxed), **kw_p)
                if gold_p and pred_p:
                    correct = bool(_mv_verify(gold_p, pred_p, **kw_v))
            except Exception:
                correct = False
        if not correct:                    # math-verify 失敗或不可用時的字串退路
            correct = (normalize_answer(boxed) == normalize_answer(gold))
    return {
        "correct": bool(correct),
        "has_think": bool(has_think),
        "has_close_think": bool(has_close),
        "boxed": boxed,
        "pred": boxed,
    }

# ---- 自我測試：確認判分邏輯正確 ----
print("=" * 64)
print(" 判分邏輯自我測試")
print("=" * 64)
CASES = [
    (f"{TAG_THINK_OPEN}\n12 * 7 = 84.\n{TAG_THINK_CLOSE}\n\nThe final answer is \\boxed{{84}}.", "84", True),
    (f"{TAG_THINK_OPEN}\n1/2 = 0.5\n{TAG_THINK_CLOSE}\n\n\\boxed{{0.5}}", r"\frac{1}{2}", True),
    (f"{TAG_THINK_OPEN}\nthinking...\n{TAG_THINK_CLOSE}\n\n\\boxed{{7}}", "84", False),
    ("沒有思考標籤，直接作答 \\boxed{84}", "84", True),
    ("完全沒有 boxed 的輸出", "84", False),
]
all_ok = True
for i, (gen, gold, want) in enumerate(CASES, 1):
    r = score_one(gen, gold)
    ok = (r["correct"] == want)
    all_ok &= ok
    print(f" case {i}: correct={r['correct']!s:5s} (期望 {want!s:5s}) {'✅' if ok else '❌'}"
          f"  think={r['has_think']} close={r['has_close_think']} boxed={r['boxed']!r}")

print(f"\n{'✅ 判分邏輯正確' if all_ok else '❌ 判分邏輯有問題，請勿相信後續準確率數字'}")
if not all_ok:
    print("""
 最可能的原因是 math-verify 的 timeout 問題（見 §8.3）。
 請確認 MATH_VERIFY_MODE，或在報告中明確標示改用字串比對。
""")
''')

code(r'''
#@title §8.5 透過 LM Studio 評估（$ 每 200 題約 10–60 分鐘）{display-mode: "form"}

# ============================================================
# 環境：.venv（需同時開著 LM Studio，runtime 選 CUDA）
# 預估時間：每 200 題、max_tokens=4096 約 10–60 分鐘（依 GPU 與思考長度）
#
# ★ 執行前檢查清單（每次都要確認）：
#   1. LM Studio 已開啟，且 Runtime 面板選的是 CUDA 版 llama.cpp
#   2. 模型已匯入（用 lms ls 看得到）
#   3. 前一批模型評估完後先 unload，避免 VRAM 不足
#   4. 若要跨模型比較，--mode / --limit / --max-tokens / --context 必須完全相同
# ============================================================
import json, time
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

# 系統提示：使用 Qwen 官方推薦的句型（"Best Practices → Standardize Output Format"）
SYS = ("You are a mathematical reasoning expert. "
       "Please reason step by step, and put your final answer within \\boxed{}.")

FEWSHOT_DEMO = (
    "What is 12 * 7? Please reason step by step, and put your final answer "
    "within \\boxed{}.\n\n"
    + TAG_THINK_OPEN + "\n12 * 7 = 84.\n" + TAG_THINK_CLOSE +
    "\n\nThe final answer is \\boxed{84}."
)

def build_chat(lms, mode, question, n_shot=4):
    """mode: think | nothink | fewshot  （對應 project_plan.md §7.1 的 B0/B1/B2）"""
    if mode == "fewshot":
        demo = "\n\n".join([FEWSHOT_DEMO] * n_shot)
        chat = lms.Chat(SYS + "\n\nHere are examples of the expected format:\n\n" + demo)
        chat.add_user_message(question)
        return chat
    if mode == "nothink":
        chat = lms.Chat(SYS + "\nAnswer directly with the final answer. Do not think step by step.")
        chat.add_user_message(question)
        return chat
    chat = lms.Chat(SYS)
    chat.add_user_message(question)
    return chat

def evaluate_lmstudio(
    model_key, tag,
    bench="HuggingFaceH4/MATH-500", split="test",
    limit=200, mode="think", max_tokens=4096, context=8192,
    out_path=None, verbose=True,
):
    """用 LM Studio（CUDA runtime）跑一組評估，回傳 summary dict。"""
    import lmstudio as lms
    from datasets import load_dataset

    # SDK >= 1.5.0 的同步 API 預設 60 秒無活動就逾時；思考很長時要放寬
    try:
        lms.set_sync_api_timeout(600)
    except Exception:
        pass

    out_path = Path(out_path) if out_path else RESULTS / f"eval_{tag}_{mode}.json"
    print("=" * 64)
    print(f" 評估：{tag}  mode={mode}")
    print("=" * 64)
    print(f" 模型      : {model_key}")
    print(f" 基準      : {bench} / {split}  題數上限 {limit}")
    print(f" max_tokens: {max_tokens}   context: {context}")

    # ★ gpu.ratio = 1.0 是 CUDA 生效的關鍵；不是 1.0 就會有一部分跑在 CPU
    model = lms.llm(model_key, config={"contextLength": context, "gpu": {"ratio": 1.0}})
    try:
        info = model.get_info()          # 文件用法是 get_info()，不是 get_model_info()
        print(f" 已載入    : {getattr(info, 'displayName', '?')}")
        print(f" 最大 context: {getattr(info, 'maxContextLength', '?')}")
    except Exception as e:
        print(f" （無法取得模型資訊：{type(e).__name__}）")

    ds = load_dataset(bench, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f" 實際評測題數：{len(ds)}")

    correct = total = 0
    think_ok = closed_ok = boxed_ok = truncated = 0
    gen_tokens, elapsed_list = [], []
    rows = []

    for i, ex in enumerate(ds, 1):
        question = ex.get("problem") or ex.get("question")
        gold = ex.get("answer") or ex.get("solution")
        chat = build_chat(lms, mode, question)

        t0 = time.time()
        try:
            result = model.respond(chat, config={"temperature": 0.0,   # greedy，可重現
                                                 "maxTokens": max_tokens})
            gen = str(result)
            stop_reason = str(getattr(result.stats, "stop_reason", ""))
            n_tok = int(getattr(result.stats, "predicted_tokens_count", 0) or 0)
        except Exception as e:
            print(f"  [{i}] 推論失敗：{type(e).__name__}: {str(e)[:120]}")
            rows.append({"idx": i, "question": question, "gold": gold, "error": str(e)})
            total += 1
            continue
        dt = time.time() - t0

        sc = score_one(gen, gold)
        # 截斷判定：達到上限，且不是正常結束
        hit_cap = (n_tok >= max_tokens) and ("endOfText" not in stop_reason)
        if mode == "nothink":
            sc["has_think"] = True   # 這個模式本來就不要求思考，不列入合規統計

        total += 1
        correct    += int(sc["correct"])
        think_ok   += int(sc["has_think"])
        closed_ok  += int(sc["has_close_think"])
        boxed_ok   += int(sc["boxed"] is not None)
        truncated  += int(hit_cap)
        gen_tokens.append(n_tok)
        elapsed_list.append(dt)
        rows.append({
            "idx": i, "question": question, "gold": gold,
            "pred_boxed": sc["boxed"], "correct": sc["correct"],
            "has_think": sc["has_think"], "has_close_think": sc["has_close_think"],
            "truncated": hit_cap, "gen_tokens": n_tok,
            "elapsed_sec": round(dt, 2), "generation": gen,
        })

        if verbose and i % 10 == 0:
            print(f"  [{i}/{len(ds)}] acc={correct/total:6.1%}  "
                  f"boxed={boxed_ok/total:6.1%}  trunc={truncated/total:6.1%}  "
                  f"平均 {(sum(elapsed_list)/len(elapsed_list)):.1f}s/題")

    n = max(total, 1)
    summary = {
        "tag": tag, "model_key": str(model_key), "benchmark": bench, "split": split,
        "mode": mode, "n": total, "context": context, "max_tokens": max_tokens,
        "accuracy": correct / n,
        "think_rate": think_ok / n,
        "closed_think_rate": closed_ok / n,
        "boxed_rate": boxed_ok / n,
        "truncation_rate": truncated / n,
        "avg_gen_tokens": (sum(gen_tokens) / len(gen_tokens)) if gen_tokens else 0,
        "avg_elapsed_sec": (sum(elapsed_list) / len(elapsed_list)) if elapsed_list else 0,
        "math_verify_mode": MATH_VERIFY_MODE,
    }
    # 標準誤（p=0.5 最保守）
    p = summary["accuracy"]
    summary["stderr"] = (p * (1 - p) / n) ** 0.5
    summary["ci95"] = [max(0.0, p - 1.96 * summary["stderr"]),
                       min(1.0, p + 1.96 * summary["stderr"])]

    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    out_path.write_text(json.dumps({"summary": summary, "rows": rows},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已寫入 {out_path}")
    if summary["truncation_rate"] > 0.20:
        print(f" ⚠️ 截斷率 {summary['truncation_rate']:.1%} 超過 20%，"
              f"這個基準的數字不可信，請提高 max_tokens（且兩邊都要改）。")
    return summary

print("已定義 build_chat() 與 evaluate_lmstudio()。")
print("""
使用方式（把 model_key 換成你的路徑或 LM Studio 識別字）：

  evaluate_lmstudio(r"D:\\...\\gguf\\qwen35-2b-math-f16.gguf",
                    tag="2b-math-f16", mode="think", limit=200)
""")
''')

code(r'''
#@title §8.6 執行評估：煙霧測試 → 完整對照（$ 數小時，建議分批）{display-mode: "form"}
# ============================================================
# 環境：.venv + LM Studio（CUDA runtime）
# 預估時間：5 題煙霧測試 1–3 分鐘；完整 200 題每組 10–60 分鐘
#
# 檔名設計：eval_<tag>_<mode>.json（把參數寫進檔名，方便日後核對一致性）
# ============================================================
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
GGUF = ROOT / "gguf"

DRY_RUN = True      # ← 第一次請保持 True，確認格式對了再改 False

# ★ 要比較的模型：所有互相比較的項目，mode/limit/max_tokens/context 必須完全相同
EVAL_JOBS = [
    # (顯示標籤,          模型檔案,                        mode)
    ("2b-math-f16",      "qwen35-2b-math-f16.gguf",      "think"),
    ("2b-math-q8",       "qwen35-2b-math-Q8_0.gguf",     "think"),
    ("2b-math-q4km",     "qwen35-2b-math-Q4_K_M.gguf",   "think"),
    ("2b-base-f16",      "qwen35-2b-base-f16.gguf",      "think"),
    ("2b-base-nothink",  "qwen35-2b-base-f16.gguf",      "nothink"),
    ("2b-base-fewshot",  "qwen35-2b-base-f16.gguf",      "fewshot"),
    ("4b-math-f16",      "qwen35-4b-math-f16.gguf",      "think"),
    ("4b-math-q4km",     "qwen35-4b-math-Q4_K_M.gguf",   "think"),
]

LIMIT      = 5 if DRY_RUN else 200
MAX_TOKENS = 4096 if DRY_RUN else 4096     # 2B 建議 4096；4B 可考慮 8192
CONTEXT    = 8192

RUN_EVAL_ON = globals().get("RUN_EVAL", True)
if not RUN_EVAL_ON:
    print("⏭️  RUN_EVAL = False，跳過評估執行。把 §0 的 RUN_EVAL 改成 True 才會跑。")
    print("    （提醒：評估需要先開啟 LM Studio，且 runtime 要選 CUDA）")

summaries = []
for tag, fname, mode in (EVAL_JOBS if RUN_EVAL_ON else []):
    key = str(GGUF / fname)
    if not (GGUF / fname).exists():
        print(f"⏭️  跳過 {tag}：找不到 {key}（請先跑 §9–§11 產生 GGUF）")
        continue
    print("\n" + "#" * 70)
    print(f"# {tag}   檔案={fname}   mode={mode}   limit={LIMIT}")
    print("#" * 70)
    try:
        s = evaluate_lmstudio(key, tag=tag, mode=mode, limit=LIMIT,
                              max_tokens=MAX_TOKENS, context=CONTEXT)
        summaries.append(s)
    except Exception as e:
        print(f" ❌ {tag} 評估失敗：{type(e).__name__}: {str(e)[:300]}")
        print("    常見原因：LM Studio 沒開、runtime 不是 CUDA、模型名稱打錯、VRAM 不足")

if summaries:
    p = RESULTS / ("eval_summary_dryrun.json" if DRY_RUN else "eval_summary.json")
    p.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已寫入 {p}")
    if DRY_RUN:
        print("""
--------------------------------------------------------------------
 現在請【手動打開】上面產生的 eval_*.json，
 檢查 rows[i]["generation"] 裡是不是真的有推理過程、結尾有 \\boxed{}。
 格式不對就不要往下跑 200 題！確認無誤後把 DRY_RUN 改成 False。
--------------------------------------------------------------------""")
''')

code(r'''
#@title §8.7 配對檢定：McNemar + bootstrap（$ < 1 分鐘）{display-mode: "form"}

# ============================================================
# 環境：.venv
# 預估時間：< 1 分鐘
#
# ★ project_plan.md §7.8 最重要的一點：
#   base 與微調模型是在【完全相同的題目】上評估的，這叫「配對資料」。
#   配對比較遠比「比較兩個獨立的信賴區間」靈敏得多。
#
#   不要用「A 是 28%±4%，B 是 32%±4%，區間重疊所以不顯著」這種推論。
#   要用 McNemar 檢定 + 對「每題對錯」做 bootstrap。
# ============================================================
import json, random
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
RESULTS = ROOT / "results"

def load_rows(path):
    """把結果 JSON 轉成 {題目: 是否答對}，供配對使用。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for r in data.get("rows", []):
        if "correct" in r and r.get("question"):
            out[r["question"]] = bool(r["correct"])
    return out

def mcnemar(a, b, n_boot=10000, seed=3407):
    """配對比較兩個模型在同一批題目上的表現。

    a = 基準模型的 {題目: 對錯}；b = 對照模型。
    回傳 accuracy_a / accuracy_b / diff / p 值 / bootstrap 信賴區間。
    """
    common = sorted(set(a) & set(b))
    if not common:
        return None
    n = len(common)
    acc_a = sum(a[q] for q in common) / n
    acc_b = sum(b[q] for q in common) / n
    b_only = sum(1 for q in common if b[q] and not a[q])   # 只有 b 對
    a_only = sum(1 for q in common if a[q] and not b[q])   # 只有 a 對
    disc = b_only + a_only

    # McNemar 精確檢定（二項分布，等同 scipy.stats.binomtest(b_only, disc, 0.5)）
    p_val = None
    if disc > 0:
        try:
            from scipy.stats import binomtest
            p_val = float(binomtest(b_only, disc, 0.5).pvalue)
        except Exception:
            # 手算精確二項 p 值（雙尾）
            from math import comb
            k = min(b_only, a_only)
            p_val = float(min(1.0, 2 * sum(comb(disc, i) for i in range(k + 1)) / (2 ** disc)))

    # 對「每題對錯」做 bootstrap，取得差異的信賴區間
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = sum(a[common[i]] for i in idx)
        sb = sum(b[common[i]] for i in idx)
        diffs.append((sb - sa) / n)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    return {
        "n_common": n, "acc_a": acc_a, "acc_b": acc_b, "diff": acc_b - acc_a,
        "a_only": a_only, "b_only": b_only, "discordant": disc,
        "p_value": p_val, "boot_ci95": [lo, hi],
        "significant": (p_val is not None and p_val < 0.05),
    }

def compare_files(path_a, path_b, label_a="A", label_b="B"):
    a, b = load_rows(path_a), load_rows(path_b)
    r = mcnemar(a, b)
    print("=" * 76)
    print(f" {label_a}  vs  {label_b}")
    print("=" * 76)
    if r is None:
        print(" ❌ 兩份結果沒有共同題目，無法配對比較")
        print(f"    {label_a}: {len(a)} 題, {label_b}: {len(b)} 題")
        return None
    print(f" 共同題數        : {r['n_common']}")
    print(f" {label_a:16s}: {r['acc_a']:6.1%}")
    print(f" {label_b:16s}: {r['acc_b']:6.1%}")
    print(f" 差異            : {r['diff']:+6.1%}   (bootstrap 95% CI "
          f"{r['boot_ci95'][0]:+.1%} ~ {r['boot_ci95'][1]:+.1%})")
    print(f" 不一致題數      : {r['discordant']}  "
          f"(僅 {label_a} 對 {r['a_only']} / 僅 {label_b} 對 {r['b_only']})")
    if r["p_value"] is not None:
        verdict = "✅ 顯著差異" if r["significant"] else "❌ 無顯著差異"
        print(f" McNemar p 值    : {r['p_value']:.4f}   → {verdict}")
    else:
        print(" McNemar        : 兩者完全一致，無差異")
    return r

# ---------- 自動配對所有可比較的結果 ----------
def find_pair(tag_a, tag_b, mode="think"):
    pa = RESULTS / f"eval_{tag_a}_{mode}.json"
    pb = RESULTS / f"eval_{tag_b}_{mode}.json"
    return (pa, pb) if pa.exists() and pb.exists() else (None, None)

PAIRS = [
    # ★ 回答「微調有沒有用」→ 必須同一種量化格式！
    ("2b-base-f16", "2b-math-f16",   "微調效果（base vs 微調，都是 F16）"),
    # ★ 回答「量化掉多少」→ 同一個模型！(第一組)
    ("2b-math-f16", "2b-math-q8",    "Q8_0 的量化代價"),
    ("2b-math-f16", "2b-math-q4km",  "Q4_K_M 的量化代價"),
    # 規模效應
    ("2b-math-f16", "4b-math-f16",   "規模效應（2B vs 4B）"),
]

all_results = {}
print("\n" + "█" * 76)
print("█ 自動配對比較")
print("█" * 76)
for ta, tb, desc in PAIRS:
    pa, pb = find_pair(ta, tb)
    if pa is None:
        print(f"\n⏭️  跳過「{desc}」：缺少 eval_{ta}_think.json 或 eval_{tb}_think.json")
        continue
    print(f"\n▶ {desc}")
    r = compare_files(pa, pb, label_a=ta, label_b=tb)
    if r:
        all_results[f"{ta}__vs__{tb}"] = {**r, "description": desc}

if all_results:
    p = RESULTS / "paired_tests.json"
    p.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    print(f"\n已寫入 {p}")
else:
    print("\n目前沒有可配對的結果（請先完成 §8.6 的評估，且至少兩組有相同題目）。")
''')

md(r"""
---
# 階段 E：合併與部署

## §9 E7：QLoRA vs bf16 的**差異化對照實驗**

**預估時間：3–12 分鐘**

### 為什麼要做一個「預期會失敗」的實驗？

`project_plan.md` §6.2 的觀點：

> 一般的教學會說「跑通了才能繼續」。但你的專案裡，
> **「證明某個主流方法在這個架構上不可用」本身就是有價值的結果。**
> 前提是要有**完整、可重現的證據**。

### ⚠️ 但本機的實測結果與計畫書的預期不同（重要）

`project_plan.md` §5.6 預期 QLoRA 會崩潰在「第一次 forward」：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (258x5120 and 1x15728640)
```

**但在這台機器上，實測結果是：QLoRA 可以完整訓練，而且不省 VRAM。**

| 實測項目（Qwen3.5-2B、text_only、r=16、seq 512） | bf16 LoRA | QLoRA（不跳過 linear_attn） |
| --- | --- | --- |
| 載入後 VRAM | 3.52 GB | 2.59 GB |
| 可訓練參數 | 10,911,744 | 10,911,744 |
| 前向 | ✅ | ✅ |
| **訓練 3 步** | ✅ | ✅ |
| **訓練峰值 VRAM** | **3.92 GB** | **3.92 GB（省下 +0.00 GB）** |

**兩個結論：**

1. **崩潰無法重現，因為觸發條件不存在。**
   issue #10010 / #9867 的根因是「載入到 Unsloth 的**預量化鏡像**
   `unsloth/*-bnb-4bit`，該鏡像上 `linear_attn.*` 的 `quant_state` 缺失」。
   本專案是從本機 `models\` 目錄載入原始檢查點，
   讓 bitsandbytes **即時量化**，`quant_state` 是完整的，所以不會崩。

2. **★ 更重要的是：QLoRA 在這裡沒有換到任何 VRAM。**
   因為 `text_only=True` 已經把視覺塔整個拿掉，
   2B 的凍結權重只剩 1.89B 參數；在此規模下，
   4-bit 量化省下的權重約 0.9 GB，卻被**解量化／優化器的額外開銷吃掉**，
   峰值完全一樣。

> **這其實是比「崩潰」更有價值的發現**：它把結論從
> 「QLoRA 壞掉不能用」修正為
> **「QLoRA 在此規模下不划算 —— 省不到記憶體，還要承擔 4-bit 品質風險」**。
> 這正好支持 `project_plan.md` §5.6.7 的最終建議：**直接用 bf16 LoRA**。

### 本格的設計：A/B 兩組都要跑

只測「載入 + 前向」是不夠的（實測顯示前向會過，訓練才會暴露問題），
所以 §9 會**跑真正的訓練步驟**，並且**同時跑對照組**：

| 組別 | 設定 | 作用 |
| --- | --- | --- |
| **A（對照組）** | bf16 LoRA | 證明流程本身沒問題 |
| **B（實驗組）** | QLoRA **不跳過** `linear_attn` | 量測真實的 VRAM 與可行性 |

**判定邏輯**（本格會自動判斷並寫入 `results/E7_outcome.json`）：

- B 訓練失敗且命中 `mat1 and mat2` → 證實 QLoRA 不可用（第一手 traceback）
- B 訓練成功 → 比較峰值 VRAM：
  - 省下 ≥ 1.5 GB → QLoRA 值得，建議用它
  - 省下 < 0.5 GB → **QLoRA 不划算，應改用 bf16 LoRA**（本機實測就是這個情況）
""")

code(r'''
#@title §9.1 E7：QLoRA 不跳過 linear_attn（$ 3–12 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv（需要 GPU）＋ bitsandbytes
# 預估時間：3–12 分鐘（含 2B 的 bf16 對照組）
#
# ★★ 本格是【差異化實驗】，比對同一份資料上的兩種方法：
#      A. bf16 LoRA（對照組，已知會成功）
#      B. QLoRA 不跳過 linear_attn（實驗組）
#    這樣無論 B 成功或失敗，你都能得到【可比較的】結論。
#
# 為什麼要這樣設計：
#   project_plan.md 預期 QLoRA 會崩潰在「第一次 forward」。
#   但那個崩潰的觸發條件是【載入 Unsloth 的預量化鏡像】
#   （issue #10010 / #9867）。本機沒有那個鏡像可用
#   （快取裡沒有 unsloth/*-bnb-4bit，而且是從本機 models\ 載入），
#   所以 B 可能在前向成功、卻在【訓練步驟】才失敗。
#   只做「載入 + 前向」的測試會得到「QLoRA 可用」的錯誤結論 —— 本格因此跑真正的訓練步驟。
# ============================================================
import json, time, traceback
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)

RUN_E7 = globals().get("RUN_QLORA", True)
MODEL_2B   = ROOT / "models" / "Qwen3.5-2B-Base"
SMOKE_DATA = ROOT / "data" / "processed" / "smoke.jsonl"
E7_OUT     = ROOT / "outputs" / "E7_2b_qlora_no_skip"

try:
    import bitsandbytes as bnb
    BNB_VER = bnb.__version__
    print(f" bitsandbytes 版本：{BNB_VER}")
    BNB_OK = True
except Exception as e:
    BNB_VER = None
    print(f" ❌ bitsandbytes 不可用：{type(e).__name__}: {e}")
    print("    E7 需要 bitsandbytes：pip install bitsandbytes")
    BNB_OK = False


def _probe(quant, label, use_collator=True):
    """跑一次「載入 → forward → 訓練 2 步」，回傳結構化結果。"""
    rec = {"label": label, "quant": quant, "loaded": False, "forward_ok": False,
           "train_ok": False, "error_type": None, "error_msg": None,
           "trainable_params": None, "trainable_pct": None,
           "vram_after_load_gb": None, "peak_vram_gb": None,
           "trainable_vision_tensors": None,
           "trainable_linear_attn_tensors": None, "hit_expected_marker": False}
    print("\n" + "=" * 64)
    print(f" 受測組：{label}（quant={quant}）")
    print("=" * 64)
    try:
        model, tokenizer, meta = load_model_with_lora(
            MODEL_2B, rank=16, max_seq_length=512, quant=quant)
        rec["loaded"] = True
        rec.update({k: meta.get(k) for k in
                    ("trainable_params", "trainable_pct",
                     "trainable_vision_tensors", "trainable_linear_attn_tensors")})
        # ★ 穩定的記憶體指標：adapter 掛好、訓練還沒開始時的 allocated。
        #   峰值（peak）會受前一組殘留與快取影響，跨組比較不可靠。
        if torch.cuda.is_available():
            rec["vram_after_load_gb"] = round(
                torch.cuda.memory_allocated()/1024**3, 3)
            print(f"   掛好 adapter 後的 allocated VRAM："
                  f"{rec['vram_after_load_gb']:.2f} GB")

        # ---------- 前向 ----------
        from datasets import load_dataset
        ds = load_dataset("json", data_files=str(SMOKE_DATA), split="train")
        sample = tokenizer(ds[0]["text"], return_tensors="pt", truncation=True,
                           max_length=512)
        sample = {k: v.to("cuda") for k, v in sample.items()}
        with torch.no_grad():
            _ = model(**sample)
        rec["forward_ok"] = True
        print(" ✅ 前向成功")

        # ---------- 真的訓練 2 步（這才是關鍵） ----------
        torch.cuda.reset_peak_memory_stats()
        _, rep = train_lora(model, tokenizer, SMOKE_DATA, E7_OUT,
                            epochs=1.0, max_steps=2, max_seq_length=512,
                            save_steps=2, logging_steps=1,
                            train_on_responses_only=use_collator)
        rec["train_ok"] = True
        rec["peak_vram_gb"] = rep.get("peak_memory_allocated_gb")
        print(" ✅ 訓練步驟成功")
        del model
    except Exception as e:
        rec["error_type"] = type(e).__name__
        rec["error_msg"] = str(e)[:400]
        tb = traceback.format_exc()
        for marker in ("mat1 and mat2 shapes cannot be multiplied",
                       "FP4 quantization state not initialized",
                       "quant_state", "not iterable", "named_modules"):
            if marker in tb or marker in str(e):
                rec["hit_expected_marker"] = True
                rec["marker"] = marker
        print(f" ❌ {label} 失敗於 "
              f"{'訓練步驟' if rec['forward_ok'] else '載入／前向'}："
              f"{rec['error_type']}: {rec['error_msg'][:200]}")
        rec["traceback"] = tb
    finally:
        _free_vram()
        if torch.cuda.is_available():
            print(f"   清理後 VRAM：{torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return rec


E7_RESULT = None
if RUN_E7 and BNB_OK:
    if not SMOKE_DATA.exists():
        print("❌ 找不到 smoke.jsonl，請先跑 §5.5")
    else:
        print("""
 本格將依序測試：
   A. bf16 LoRA        ← 對照組（預期成功）
   B. QLoRA 不跳過 linear_attn ← 實驗組（專案預期失敗）
""")
        recA = _probe("bf16",  "A: bf16 LoRA（對照組）")
        recB = _probe("qlora", "B: QLoRA 不跳過 linear_attn（實驗組）")

        # ---------- 判定 ----------
        # 用「掛好 adapter 後的 allocated VRAM」比較 —— 這是穩定且可比的指標。
        # 訓練峰值會受前一組殘留與 CUDA 快取影響，跨組比較不可靠。
        saving = None
        if recA.get("vram_after_load_gb") is not None and \
           recB.get("vram_after_load_gb") is not None:
            saving = recA["vram_after_load_gb"] - recB["vram_after_load_gb"]

        if recB["train_ok"]:
            if saving is None:
                outcome = "qlora_worked_saving_unknown"
                verdict = "QLoRA 可訓練，但無法取得峰值 VRAM，請看 log 自行判斷。"
            elif saving >= 1.5:
                outcome = "qlora_worked_big_saving"
                verdict = (f"QLoRA 可訓練且省下 {saving:.2f} GB —— "
                           f"值得考慮作為省記憶體的路線。")
            elif saving >= 0.5:
                outcome = "qlora_worked_modest_saving"
                verdict = (f"QLoRA 可訓練，只省下 {saving:.2f} GB —— "
                           f"效益有限，且需承擔 4-bit 品質風險，"
                           f"建議仍以 bf16 LoRA 為主力。")
            else:
                outcome = "qlora_worked_no_saving"
                verdict = (f"QLoRA 可訓練，但掛好 adapter 後的 VRAM 幾乎沒有差別"
                           f"（省下 {saving:+.2f} GB）。"
                           f"★ 結論：QLoRA 在此規模下【不划算】—— "
                           f"省不到記憶體，還要承擔 4-bit 品質風險。"
                           f"這反而是比「崩潰」更有價值的發現，"
                           f"也支持 project_plan.md §5.6.7「直接用 bf16 LoRA」的建議。")
        elif recB["forward_ok"]:
            outcome = "crashed_at_train_step"
            verdict = ("QLoRA 能載入、能前向，但【在訓練步驟崩潰】。"
                       "這正是 issue #10010 / #9867 的形狀錯誤 —— "
                       "標準 QLoRA 在此架構上不可用，且崩潰點比預期更晚"
                       "（不是第一次 forward，而是訓練步驟）。")
        else:
            outcome = "crashed_at_load_or_forward"
            verdict = ("QLoRA 在載入或第一次前向就崩潰，"
                       "與 project_plan.md §5.6 的預期一致。")

        print("\n" + "█" * 70)
        print("█ E7 差異化實驗結論")
        print("█" * 70)
        for rec in (recA, recB):
            pct = rec["trainable_pct"]
            print(f"  {rec['label']:40s} 載入={rec['loaded']!s:5s} "
                  f"前向={rec['forward_ok']!s:5s} 訓練={rec['train_ok']!s:5s} "
                  f"可訓練%={round(100*pct,4) if pct else None} "
                  f"載入後VRAM={rec.get('vram_after_load_gb')} "
                  f"訓練峰值={rec.get('peak_vram_gb')}")
        print(f"\n 判定：{outcome}")
        print(f" 解讀：{verdict}")

        if recB["error_type"]:
            print(f"\n 實驗組的錯誤：{recB['error_type']}: {recB['error_msg'][:300]}")
            if recB["hit_expected_marker"]:
                print(f" ✔️ 命中關鍵訊息（{recB.get('marker')}）→ "
                      f"可以直接引用為第一手證據。")
            else:
                print(" ⚠️ 未命中 issue 描述的關鍵訊息，"
                      "請勿在報告中寫成「重現了 mat1/mat2 錯誤」。")

        if saving is not None:
            print(f"\n 記憶體對照（掛好 adapter 後、訓練尚未開始）"
                  f"：bf16 {recA['vram_after_load_gb']:.2f} GB  vs  "
                  f"QLoRA {recB['vram_after_load_gb']:.2f} GB"
                  f"（QLoRA 省下 {saving:+.2f} GB）")
            print(f""" 訓練峰值（僅供參考，跨組比較不可靠）：bf16 {recA.get('peak_vram_gb')} vs QLoRA {recB.get('peak_vram_gb')} GB
 ⚠️ 峰值會受前一組實驗殘留與 CUDA 快取影響。
    想取得最乾淨的比較，請【重啟 kernel】後只跑其中一組，再手動記錄。

 如何在報告中呈現（本機實測結論）
   • QLoRA「能不能跑」與「值不值得用」是兩個不同的問題。
     本機的答案是：能跑，但在此規模下省下的記憶體有限 → 不值得當主力。
   • 請把「崩潰無法重現」的原因一併寫清楚：
     本專案從本機 models\\ 載入原始檢查點，bitsandbytes 即時量化，
     quant_state 完整；issue #10010 / #9867 的觸發條件是
     Unsloth 的【預量化鏡像】，本專案並沒有走到那條路。
   • 如果你的報告需要「重現崩潰」，必須改成載入預量化鏡像
     （unsloth/*-bnb-4bit），那不在本 Notebook 的預設流程內。
""")

        # ---------- 保存第一手證據 ----------
        (RESULTS / "E7_outcome.json").write_text(
            json.dumps({"experiment": "E7",
                        "description": "QLoRA 不跳過 linear_attn（差異化受控實驗）",
                        "model": "Qwen3.5-2B-Base",
                        "method": "text_only=True + FastVisionModel",
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "bitsandbytes": BNB_VER,
                        "control_bf16": recA, "treatment_qlora": recB,
                        "outcome": outcome, "interpretation": verdict},
                       indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        crash_file = RESULTS / "qlora_crash_traceback.txt"
        if recB.get("traceback"):
            crash_file.write_text(
                "E7：QLoRA 不跳過 linear_attn（Qwen3.5-2B-Base）\n"
                + "=" * 70 + "\n"
                + f"時間：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                + f"torch：{torch.__version__}  CUDA：{torch.version.cuda}\n"
                + f"bitsandbytes：{BNB_VER}\n"
                + f"判定：{outcome}\n\n"
                + recB["traceback"], encoding="utf-8")
            print(f"\n完整 traceback 已保存 → {crash_file}")
        elif crash_file.exists():
            print(f"\n（實驗組未失敗，先前的 traceback 保留在 {crash_file}）")

        print(f"E7 結果已記錄 → {RESULTS/'E7_outcome.json'}")
elif not RUN_E7:
    print("⏭️  RUN_QLORA = False，跳過 E7。把 §0 的 RUN_QLORA 改成 True 才會跑。")
''')

md(r"""
## §10 合併權重

**預估時間：5–20 分鐘**

LoRA adapter 只是一個小檔案（幾十到幾百 MB），
部署前需要「合併」回基礎模型，成為獨立的完整模型。
**GGUF 轉換也必須用合併後的權重。**

> ⚠️ **合併時務必在 CPU 上做。** 用 GPU 合併 4B 模型很容易 OOM，
> 而你有 64 GB RAM，在 CPU 上合併又快又安全。
""")

code(r'''
#@title §10.1 合併 adapter → bf16 safetensors（$ 5–20 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv（CPU，不需要 GPU）
# 預估時間：2B 約 3–8 分鐘；4B 約 8–20 分鐘
# ============================================================
import shutil, time
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent

MERGED = ROOT / "outputs" / "merged"; MERGED.mkdir(parents=True, exist_ok=True)

# (adapter 資料夾, 輸出資料夾名稱)
MERGE_JOBS = []
if globals().get("RUN_MERGE", True):
    MERGE_JOBS = [
        ("E1_2b_bf16_r16", "qwen35-2b-math-r16"),
        # ("E2_4b_bf16_r16", "qwen35-4b-math-r16"),
        # ("E3_2b_dora_r16", "qwen35-2b-math-dora"),
    ]
else:
    print("⏭️  RUN_MERGE = False，跳過合併。把 §0 的 RUN_MERGE 改成 True 才會跑。")

# 每個 adapter 對應的【原始 base 模型】（一定要用原始 bf16 base，不能用已合併的）
ADAPTER_BASE = {
    "E1_2b_bf16_r16": "Qwen3.5-2B-Base",
    "E2_4b_bf16_r16": "Qwen3.5-4B-Base",
    "E3_2b_dora_r16": "Qwen3.5-2B-Base",
    "E4_4b_dora_r16": "Qwen3.5-4B-Base",
}

for adapter_name, out_name in MERGE_JOBS:
    adapter_dir = ROOT / "outputs" / adapter_name
    out_dir = MERGED / out_name
    base_key = ADAPTER_BASE.get(adapter_name)
    base_dir = ROOT / "models" / base_key if base_key else None

    print("=" * 64)
    print(f" 合併 {adapter_name} → {out_name}")
    print("=" * 64)
    if not adapter_dir.exists():
        print(f" ❌ 找不到 adapter：{adapter_dir}（請先跑 §7）"); continue
    if base_dir is None or not base_dir.exists():
        print(f" ❌ 找不到 base 模型：{base_dir}"); continue

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        t0 = time.time()
        print(" 載入 base 模型（CPU、bf16，避免 VRAM 不足）…")
        try:
            base = AutoModelForCausalLM.from_pretrained(
                str(base_dir), torch_dtype=torch.bfloat16, device_map="cpu",
                low_cpu_mem_usage=True)
        except Exception as e_lm:
            # Qwen3.5 是 VLM，必要時改用多模態類別
            print(f"  AutoModelForCausalLM 失敗（{type(e_lm).__name__}），改用 AutoModelForMultimodalLM")
            from transformers import AutoModelForMultimodalLM
            base = AutoModelForMultimodalLM.from_pretrained(
                str(base_dir), torch_dtype=torch.bfloat16, device_map="cpu",
                low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(str(base_dir))

        print(" 載入並合併 adapter …")
        merged = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = merged.merge_and_unload()

        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f" 儲存到 {out_dir} …")
        merged.save_pretrained(str(out_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(out_dir))

        # ---------- 檢查大小 ----------
        total = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
        n_st = sum(1 for f in out_dir.rglob("*.safetensors"))
        print(f"\n ✅ 合併完成，耗時 {(time.time()-t0)/60:.1f} 分鐘")
        print(f"    總大小：{total/2**30:.2f} GB，safetensors 檔數：{n_st}")
        print(f"    驗收：應該與 base 相近（2B 約 4.6 GB、4B 約 9.4 GB）")
        if total / 2**30 < 0.5:
            print("    ❌ 太小了！可能只存到 adapter，請確認 merge_and_unload() 有被呼叫")
        else:
            print("    ✅ 大小合理")
        del base, merged
    except Exception as e:
        print(f" ❌ 合併失敗：{type(e).__name__}: {str(e)[:400]}")
''')

md(r"""
## §11 轉成 GGUF 並量化

**預估時間：30–60 分鐘**

### §11.1 ⚠️ 最重要的一節：MTP 層與 `--no-mtp`

Qwen3.5 的檢查點**真的包含一個 MTP（多 token 預測）區塊**：

- `config.json` 的 `mtp_num_hidden_layers: 1`
- 權重檔中有 `mtp.fc.weight`、`mtp.layers.0.*` 等張量

所以 **2B 實際有 24 + 1 = 25 塊、4B 有 32 + 1 = 33 塊**。
llama.cpp 的轉換腳本預設會把 MTP 層**一起打包**，
載入時就會去找不存在的 `blk.32`：

```
missing tensor 'blk.32.attn_norm.weight'
```

**解法：轉換時加上 `--no-mtp`**（別名 `--no-nextn`）。
MTP 只用於加速推論的投機解碼，**對數學微調沒有任何損失**。

**其他已知問題**（`project_plan.md` §9.2）：

- **請使用最新版 llama.cpp**（master）。相關修補是 2026 年才進去的。
- Windows 的 release 壓縮檔**沒有** `convert_hf_to_gguf.py`，所以要 `git clone` 原始碼。
- 另有 issue #27019（`_reorder_v_heads` 形狀錯誤）尚未合併修復，
  若遇到請改用**路線 A（Unsloth 內建匯出）**。
""")

code(r'''
#@title §11.2 路線 A：Unsloth 內建 GGUF 匯出（$ 20–60 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv（需要 GPU 或大量 RAM）
# 預估時間：20–60 分鐘（首次會 clone 並編譯 llama.cpp，會多花數分鐘到數十分鐘）
#
# ★ 這是最省事、最不容易踩雷的路線。
#   官方已處理好 Qwen3.5 的架構細節（包含 MTP 問題）。
# ============================================================
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
GGUF = ROOT / "gguf"; GGUF.mkdir(exist_ok=True)

USE_ROUTE_A = globals().get("RUN_GGUF", True)
QUANT_METHODS = ["f16", "q8_0", "q4_k_m"]   # 想省時間就先只做 f16

if USE_ROUTE_A:
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from unsloth import FastLanguageModel

    # ★ 關鍵：直接載入【合併後的完整模型】，用 16-bit（不量化）
    src_dir = ROOT / "outputs" / "merged" / "qwen35-2b-math-r16"
    print("=" * 64)
    print(" 路線 A：Unsloth 內建 GGUF 匯出")
    print("=" * 64)
    print(f" 來源：{src_dir}")
    if not src_dir.exists():
        print(" ❌ 找不到合併後的模型，請先跑 §10.1")
    else:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(src_dir), max_seq_length=2048,
                load_in_4bit=False, load_in_16bit=True,
            )
            for method in QUANT_METHODS:
                print(f"\n--- 匯出 {method} ---")
                try:
                    model.save_pretrained_gguf(
                        str(GGUF / "qwen35-2b-math"), tokenizer,
                        quantization_method=method)
                    print(f" ✅ {method} 完成")
                except Exception as e:
                    print(f" ❌ {method} 失敗：{type(e).__name__}: {str(e)[:300]}")
                    if method == "f16":
                        print("    F16 是後續量化的必要中繼檔，請先解決這個問題。")
            del model
        except Exception as e:
            print(f" ❌ 匯出流程失敗：{type(e).__name__}: {str(e)[:400]}")
            print("    請改用 §11.3 的路線 B（手動 llama.cpp）")
            _free_vram()

# ---------- 列出產出 ----------
print("\n" + "=" * 64)
print(" gguf/ 目錄內容")
print("=" * 64)
found = sorted(GGUF.glob("*.gguf"))
if found:
    for f in found:
        print(f" {f.name:42s} {f.stat().st_size/2**30:6.2f} GB")
    print("""
 預期大小（project_plan.md §10.4）
   2B：  F16 約 4.5–5 GB   Q8_0 約 2.4–2.5 GB   Q4_K_M 約 1.4–1.5 GB
   4B：  F16 約 8.4 GB     Q8_0 約 4.48 GB      Q4_K_M 約 2.74 GB

 若 Q4_K_M 只有幾百 MB，表示量化過程出錯或檔案被截斷。
""")
else:
    print(" （尚無 .gguf 檔案）")
''')

code(r'''
#@title §11.3 路線 B：手動 llama.cpp 轉換 + 量化（$ 30–90 分鐘）{display-mode: "form"}
# ============================================================
# 環境：PowerShell / .venv（CPU 為主）
# 預估時間：clone + 轉換 + 量化約 30–90 分鐘
#
# 路線 A 失敗、或你想用 imatrix 提升 Q4 品質時才需要這條路。
#
# ★★ 最重要：--no-mtp 絕對不能漏 ★★
# ============================================================
import subprocess, shutil
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
LLAMA  = ROOT / "llama.cpp"
MERGED = ROOT / "outputs" / "merged" / "qwen35-2b-math-r16"
GGUF   = ROOT / "gguf"; GGUF.mkdir(exist_ok=True)

RUN_ROUTE_B = False     # ← 需要時改成 True
if not globals().get("RUN_GGUF", True):
    RUN_ROUTE_B = False
    print("⏭️  RUN_GGUF = False，跳過路線 B。把 §0 的 RUN_GGUF 改成 True 才會跑。")

def run(cmd, cwd=None, timeout=7200):
    print(f"\n$ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str),
                       capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    out = ((p.stdout or "") + (p.stderr or ""))
    print(out[-4000:] if len(out) > 4000 else out)
    return p.returncode, out

if RUN_ROUTE_B:
    # ---------- 1. 取得 llama.cpp 原始碼（轉換腳本只存在於原始碼倉庫）----------
    if not LLAMA.exists():
        print("=" * 64); print(" 1/4 clone llama.cpp"); print("=" * 64)
        run(["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp", str(LLAMA)], timeout=1800)
        # ★ 一定要用最新版：MTP 相關修補是 2026 年才進去的
        run(["git", "log", "-1", "--format=%H %ci"], cwd=str(LLAMA))
    else:
        print(f" llama.cpp 已存在：{LLAMA}")
        run(["git", "pull", "--ff-only"], cwd=str(LLAMA))

    # ---------- 2. 安裝轉換腳本相依 ----------
    print("\n" + "=" * 64); print(" 2/4 安裝轉換腳本相依"); print("=" * 64)
    req = LLAMA / "requirements" / "requirements-convert_hf_to_gguf.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(req)], timeout=1800)

    # ---------- 3. 轉成 F16 GGUF（★ --no-mtp）----------
    print("\n" + "=" * 64); print(" 3/4 轉成 F16 GGUF（--no-mtp）"); print("=" * 64)
    f16_out = GGUF / "qwen35-2b-math-f16.gguf"
    cmd = [sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(MERGED),
           "--outtype", "f16", "--outfile", str(f16_out),
           "--no-mtp"]                      # ★★ 關鍵參數 ★★
    rc, out = run(cmd, timeout=7200)
    if rc != 0 and "lazy" in out.lower():
        print(" 偵測到 lazy transpose 問題，改用 --no-lazy 重試（RAM 用量會增加）")
        cmd.insert(-1, "--no-lazy")
        rc, out = run(cmd, timeout=7200)
    print(f"\n 轉換 {'✅ 成功' if rc == 0 else '❌ 失敗'}：{f16_out}")
    if "blk.32" in out or "blk.33" in out:
        print(" ⚠️ 偵測到 MTP block 相關錯誤 —— 確認 --no-mtp 有生效、且 llama.cpp 夠新")

    # ---------- 4. 量化 ----------
    print("\n" + "=" * 64); print(" 4/4 量化成 Q8_0 與 Q4_K_M"); print("=" * 64)
    quant = None
    for cand in ["build/bin/Release/llama-quantize.exe",
                 "build/bin/llama-quantize.exe", "llama-quantize.exe"]:
        p = LLAMA / cand
        if p.exists():
            quant = p; break
    if quant is None:
        found = list(LLAMA.rglob("llama-quantize*"))
        quant = found[0] if found else None
    print(f" llama-quantize：{quant}")

    if quant and f16_out.exists():
        # dry-run 先確認張量都能處理（省下跑很久才失敗的時間）
        run([str(quant), "--dry-run", str(f16_out), "Q4_K_M"])
        for q in ["Q8_0", "Q4_K_M"]:
            o = GGUF / f"qwen35-2b-math-{q}.gguf"
            run([str(quant), str(f16_out), str(o), q], timeout=3600)
            if o.exists():
                print(f" ✅ {q}: {o.stat().st_size/2**30:.2f} GB")

    # ---------- 5. 載入驗證（花 5 秒省 5 小時）----------
    print("\n" + "=" * 64); print(" 5/5 用 llama-cli 驗證能否載入"); print("=" * 64)
    cli = None
    for cand in ["build/bin/Release/llama-cli.exe", "build/bin/llama-cli.exe"]:
        p = LLAMA / cand
        if p.exists():
            cli = p; break
    if cli is None:
        f = list(LLAMA.rglob("llama-cli*")); cli = f[0] if f else None
    if cli and (GGUF / "qwen35-2b-math-f16.gguf").exists():
        rc, out = run([str(cli), "-m", str(GGUF / "qwen35-2b-math-f16.gguf"),
                       "-p", "What is 2+2?", "-n", "32"], timeout=600)
        if rc == 0:
            print(" ✅ 能正常載入並生成 → 繼續量化是安全的")
        elif "missing tensor" in out:
            print(" ❌ 缺少張量錯誤 → --no-mtp 沒生效，或 llama.cpp 版本太舊")
        elif "ssm_conv1d" in out:
            print(" ❌ 缺 ssm_conv1d → llama.cpp 早於 recurrent_layers 修補，請更新")
else:
    print("RUN_ROUTE_B = False，跳過。若路線 A 失敗，把這裡改成 True。")
''')

code(r'''
#@title §11.4 匯入 LM Studio 並檢查 CUDA（$ 2–10 分鐘）{display-mode: "form"}
# ============================================================
# 環境：PowerShell（lms CLI 由 LM Studio 安裝時一併裝好）
# 預估時間：每個檔案 1–3 分鐘
#
# ★ base 模型也要匯入！否則無法做最重要的對照。
#   請先用 §11.2／§11.3 把 base 模型也轉成 GGUF，
#   或直接用 llama.cpp 轉原始 models\Qwen3.5-2B-Base。
# ============================================================
import subprocess, shutil
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
GGUF = ROOT / "gguf"

def run(cmd, timeout=600):
    print(f"\n$ {cmd}")
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        out = (p.stdout or "") + (p.stderr or "")
        print(out.strip()[:2000] if out.strip() else "（無輸出）")
        return p.returncode, out
    except Exception as e:
        print(f" 執行失敗：{type(e).__name__}: {e}")
        return -1, ""

# ---------- 1. 檢查 lms CLI ----------
print("=" * 64); print(" 1. 檢查 lms CLI"); print("=" * 64)
if shutil.which("lms"):
    print(" ✅ 找到 lms CLI")
    run("lms ls")
else:
    print("""
 ⚠️ 找不到 lms 指令。
     - 請確認已安裝 LM Studio（https://lmstudio.ai/）
     - 安裝後重開終端機讓 PATH 生效
     - 或直接用 .gguf 的完整路徑當作 model-key，跳過匯入步驟
""")

# ---------- 2. 匯入所有 GGUF ----------
print("\n" + "=" * 64); print(" 2. 匯入 gguf/ 目錄下的檔案"); print("=" * 64)
IMPORT = True
if IMPORT and shutil.which("lms"):
    files = sorted(GGUF.glob("*.gguf"))
    if not files:
        print(" gguf/ 目錄下沒有 .gguf 檔案（請先跑 §11.2 或 §11.3）")
    for f in files:
        print(f"\n--- 匯入 {f.name} ({f.stat().st_size/2**30:.2f} GB) ---")
        run(f'lms import "{f}"')
    print("\n匯入後的模型清單：")
    run("lms ls")

print("""
--------------------------------------------------------------------
 ★ CUDA 生效的三處檢查（project_plan.md §7.5.1）
   1. LM Studio 的 Runtime 面板 → 選中的是 CUDA 版 llama.cpp
   2. 載入模型後的資訊列 → 顯示 GPU offload 為全部層數（例如 28/28 layers）
   3. 推論時工作管理員 → GPU 使用率明顯上升

 ⚠️ 若載入後 GPU 使用率為 0，代表你其實在用 CPU 跑，
    評估會慢 10 倍以上，而且你會誤判「模型很慢」。
--------------------------------------------------------------------""")
''')

md(r"""
---
# 階段 F：結論

## §12 彙整結果

**預估時間：< 1 分鐘**

把 `outputs/*/vram_report.json`（訓練）與 `results/eval_*.json`（評估）
彙整成一張**核心對照總表**。

### 這張表就是你的報告核心

| 模型 | 訓練方法 | GGUF 量化 | VRAM 峰值(訓練) | 可訓練% | MATH-500 acc | 截斷率 | `<think>` 合規率 | `\boxed{}` 合規率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

把「訓練方法」「量化格式」「VRAM」「可訓練%」全部放在準確率旁邊，
讀者一眼就能看出**兩層代價**：
**訓練時的方法代價（VRAM）** 與 **部署時的量化代價（準確率）**。
""")

code(r'''
#@title §12.1 彙整所有訓練與評估結果（$ < 1 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：< 1 分鐘
# ============================================================
import json
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
OUTPUTS = ROOT / "outputs"; RESULTS = ROOT / "results"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

# ---------- 1. 訓練結果 ----------
rows = []
for d in sorted(OUTPUTS.glob("*")):
    rp = d / "vram_report.json"
    if not rp.exists():
        continue
    try:
        r = json.loads(rp.read_text(encoding="utf-8"))
    except Exception:
        continue
    rows.append({
        "實驗": r.get("experiment", d.name),
        "模型": r.get("model_name", ""),
        "方法": r.get("quant", ""),
        "rank": r.get("rank", ""),
        "資料量": r.get("n_samples", ""),
        "max_seq": r.get("max_seq_length", ""),
        "可訓練參數": r.get("trainable_params", 0),
        "可訓練%": round(100 * r.get("trainable_pct", 0), 4),
        "VRAM峰值_GB": r.get("peak_memory_allocated_gb", None),
        "訓練分鐘": r.get("train_minutes", None),
        "備註": r.get("description", ""),
    })
df_train = pd.DataFrame(rows)
print("=" * 110)
print(" 訓練結果彙整（來自 outputs/*/vram_report.json）")
print("=" * 110)
if df_train.empty:
    print(" （尚無資料，請先完成 §7 的訓練）")
else:
    print(df_train.to_string(index=False))
    df_train.to_csv(RESULTS / "table_training.csv", index=False, encoding="utf-8-sig")
    print(f"\n已寫入 {RESULTS/'table_training.csv'}")

# ---------- 2. 評估結果 ----------
erows = []
for f in sorted(RESULTS.glob("eval_*.json")):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue
    s = data.get("summary") or data
    if not isinstance(s, dict) or "accuracy" not in s:
        continue
    erows.append({
        "標籤": s.get("tag", f.stem),
        "基準": str(s.get("benchmark", "")).split("/")[-1],
        "模式": s.get("mode", ""),
        "題數": s.get("n", 0),
        "準確率": round(s.get("accuracy", 0), 4),
        "SE": round(s.get("stderr", 0), 4),
        "截斷率": round(s.get("truncation_rate", 0), 4),
        "think合規率": round(s.get("think_rate", 0), 4),
        "閉合think率": round(s.get("closed_think_rate", 0), 4),
        "boxed合規率": round(s.get("boxed_rate", 0), 4),
        "平均生成token": round(s.get("avg_gen_tokens", 0), 1),
        "平均秒數": round(s.get("avg_elapsed_sec", 0), 2),
        "max_tokens": s.get("max_tokens", ""),
        "context": s.get("context", ""),
    })
df_eval = pd.DataFrame(erows)
print("\n" + "=" * 110)
print(" 評估結果彙整（來自 results/eval_*.json）")
print("=" * 110)
if df_eval.empty:
    print(" （尚無資料，請先完成 §8.6 的評估）")
else:
    print(df_eval.to_string(index=False))
    df_eval.to_csv(RESULTS / "table_eval.csv", index=False, encoding="utf-8-sig")
    print(f"\n已寫入 {RESULTS/'table_eval.csv'}")

# ---------- 3. 汙染檢查 ----------
cp = RESULTS / "contamination_report.json"
if cp.exists():
    print("\n" + "=" * 110); print(" 汙染檢查"); print("=" * 110)
    print(json.dumps(json.loads(cp.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))

# ---------- 4. E7 結果 ----------
e7 = RESULTS / "E7_outcome.json"
if e7.exists():
    print("\n" + "=" * 110); print(" E7（QLoRA 受控失敗實驗）"); print("=" * 110)
    print(json.dumps(json.loads(e7.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
''')

md(r"""
## §13 最終成果圖表

**預估時間：< 1 分鐘**

`project_plan.md` §7.2 建議畫**兩張圖**，合起來就講完了整個專案：

1. **訓練效率圖（效率前緣）**：橫軸 **VRAM 峰值**、縱軸 **準確率** → 比較訓練方法
2. **量化代價圖**：橫軸 **GGUF 檔案大小**、縱軸 **準確率** → 比較 F16/Q8/Q4

外加 **訓練 loss 曲線**，用來確認訓練真的收斂。
""")

code(r'''
#@title §13.1 繪製最終成果圖（$ < 1 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：< 1 分鐘
# ============================================================
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
RESULTS = ROOT / "results"; OUTPUTS = ROOT / "outputs"; GGUF = ROOT / "gguf"
FIGS = ROOT / "results" / "figures"; FIGS.mkdir(parents=True, exist_ok=True)

# 中文字型（避免圖上中文變方框）
for name in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PMingLiU", "DejaVu Sans"]:
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        matplotlib.rcParams["font.sans-serif"] = [name]
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 120

# ---------- 讀資料 ----------
def load_all_eval():
    out = {}
    for f in sorted(RESULTS.glob("eval_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            s = d.get("summary")
            if s and "accuracy" in s:
                out[s["tag"]] = s
        except Exception:
            continue
    return out

EVAL = load_all_eval()
def load_train():
    out = {}
    for d in OUTPUTS.glob("*"):
        rp = d / "vram_report.json"
        if rp.exists():
            try:
                r = json.loads(rp.read_text(encoding="utf-8"))
                out[r.get("experiment", d.name)] = r
            except Exception:
                pass
    return out
TRAIN = load_train()

print(f" 讀到 {len(EVAL)} 組評估結果、{len(TRAIN)} 組訓練結果")

# ============================================================
# 圖 1：訓練效率前緣（VRAM × 準確率）
# ============================================================
def fig_efficiency_frontier():
    pts = []
    # 把訓練實驗對應到評估 tag（命名慣例：E1_2b_bf16_r16 → 2b-math-f16）
    ALIAS = {
        "2b-base-f16":     ("2B-Base（未微調）", None),
        "2b-math-f16":     ("2B bf16 LoRA",      "E1_2b_bf16_r16"),
        "2b-math-dora":    ("2B DoRA",           "E3_2b_dora_r16"),
        "2b-math-qlora":   ("2B QLoRA（修正版）", "E5_2b_qlora_fixed"),
        "4b-base-f16":     ("4B-Base（未微調）", None),
        "4b-math-f16":     ("4B bf16 LoRA",      "E2_4b_bf16_r16"),
        "4b-math-dora":    ("4B DoRA",           "E4_4b_dora_r16"),
    }
    for tag, s in EVAL.items():
        if s.get("mode") != "think":
            continue
        label, exp = ALIAS.get(tag, (tag, None))
        vram = None
        if exp and exp in TRAIN:
            vram = TRAIN[exp].get("peak_memory_allocated_gb")
        if vram is None and "base" in tag:
            vram = 0.0    # base 不需要訓練
        pts.append((label, vram, s.get("accuracy", 0), s.get("n", 0)))

    if not pts:
        print(" ⏭️  圖 1 略過：沒有足夠的評估結果")
        return None

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for label, vram, acc, n in pts:
        if vram is None:
            continue
        marker = "s" if "Base" in label else "o"
        color = "#888888" if "Base" in label else None
        ax.scatter(vram, acc * 100, s=120, marker=marker,
                   color=color, zorder=3, edgecolors="black", linewidths=0.6)
        ax.annotate(f"{label}\n(n={n})", (vram, acc * 100),
                    textcoords="offset points", xytext=(9, 6), fontsize=8.5)
    ax.axvline(16, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(15.7, ax.get_ylim()[0] + 1, " 16 GB 硬體上限", color="red",
            fontsize=9, rotation=90, va="bottom", ha="right")
    ax.set_xlabel("訓練 VRAM 峰值（GB）", fontsize=11)
    ax.set_ylabel("準確率（%）", fontsize=11)
    ax.set_title("圖 1：訓練效率前緣 —— 用多少 VRAM 換到多少準確率\n"
                 "（方形＝未微調 base，圓形＝微調後）", fontsize=12)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    p = FIGS / "fig1_efficiency_frontier.png"
    fig.savefig(p); plt.show()
    print(f" 已存 {p}")
    return p

# ============================================================
# 圖 2：量化代價（檔案大小 × 準確率）
# ============================================================
def fig_quant_cost():
    # 找出實際存在的 GGUF 大小
    sizes = {}
    for f in GGUF.glob("*.gguf"):
        sizes[f.name] = f.stat().st_size / 2**30

    SPECS = [   # (顯示名, 評估 tag, GGUF 檔名關鍵字)
        ("F16",    "2b-math-f16",   "qwen35-2b-math-f16"),
        ("Q8_0",   "2b-math-q8",    "qwen35-2b-math-Q8_0"),
        ("Q4_K_M", "2b-math-q4km",  "qwen35-2b-math-Q4_K_M"),
    ]
    pts = []
    for disp, tag, key in SPECS:
        if tag not in EVAL:
            continue
        sz = None
        for fn, gb in sizes.items():
            if key in fn:
                sz = gb; break
        if sz is None:
            sz = {"F16": 4.6, "Q8_0": 2.45, "Q4_K_M": 1.45}.get(disp)  # 估計值
        pts.append((disp, sz, EVAL[tag].get("accuracy", 0), EVAL[tag].get("n", 0)))

    if len(pts) < 2:
        print(" ⏭️  圖 2 略過：需要至少兩種量化格式的評估結果")
        return None

    fig, ax = plt.subplots(figsize=(8, 5.2))
    xs = [p[1] for p in pts]; ys = [p[2] * 100 for p in pts]
    ax.plot(xs, ys, "-o", color="#1f77b4", markersize=9, linewidth=2)
    base_acc = ys[0]
    for disp, sz, acc, n in pts:
        ax.annotate(f"{disp}\n{acc:.1f}%（{acc-base_acc:+.1f}pp）", (sz, acc * 100),
                    textcoords="offset points", xytext=(0, -34),
                    ha="center", fontsize=9)
    ax.set_xlabel("GGUF 檔案大小（GB）", fontsize=11)
    ax.set_ylabel("準確率（%）", fontsize=11)
    ax.set_title("圖 2：部署量化的代價 —— 同一個模型，不同量化格式\n"
                 "（相對 F16 的損失標在括號內）", fontsize=12)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    p = FIGS / "fig2_quantization_cost.png"
    fig.savefig(p); plt.show()
    print(f" 已存 {p}")
    return p

# ============================================================
# 圖 3：訓練 loss 曲線
# ============================================================
def fig_loss_curves():
    curves = {}
    for d in sorted(OUTPUTS.glob("*")):
        f = d / "train_log.json"
        if f.exists():
            try:
                hist = json.loads(f.read_text(encoding="utf-8"))
                pts = [(h.get("step"), h.get("loss")) for h in hist
                       if h.get("step") is not None and h.get("loss") is not None]
                if pts:
                    curves[d.name] = pts
            except Exception:
                pass
    if not curves:
        print(" ⏭️  圖 3 略過：找不到 train_log.json（§7.3 會自動產生）")
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, pts in curves.items():
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", label=name, linewidth=1.6)
    ax.set_xlabel("訓練步數 (step)", fontsize=11)
    ax.set_ylabel("Training loss", fontsize=11)
    ax.set_title("圖 3：訓練 loss 曲線\n（健康：穩定下降，最後落在 0.5–1.5 附近）", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    p = FIGS / "fig3_loss_curves.png"
    fig.savefig(p); plt.show()
    print(f" 已存 {p}")
    return p

print("\n" + "=" * 64); print(" 開始繪圖"); print("=" * 64)
made = [f() for f in (fig_efficiency_frontier, fig_quant_cost, fig_loss_curves)]
made = [m for m in made if m]
print(f"\n共產生 {len(made)} 張圖，都在 {FIGS}")
''')

code(r'''
#@title §13.2 研究問題總結（$ < 1 分鐘）{display-mode: "form"}
# ============================================================
# 環境：.venv
# 預估時間：< 1 分鐘
#
# 對應 project_plan.md §12.3 的五個報告必答問題。
# 這格會根據你實際跑出來的數據，自動填入答案。
# ============================================================
import json
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "project_plan.md").exists() and ROOT.parent != ROOT:
    ROOT = ROOT.parent
RESULTS = ROOT / "results"

def jload(p, default=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

EVAL   = {}
for f in RESULTS.glob("eval_*.json"):
    d = jload(f, {})
    s = d.get("summary")
    if s and "accuracy" in s:
        EVAL[s["tag"]] = s
PAIRED = jload(RESULTS / "paired_tests.json", {})
E7     = jload(RESULTS / "E7_outcome.json", {})
TRAIN  = {}
for d in (ROOT / "outputs").glob("*"):
    r = jload(d / "vram_report.json")
    if r:
        TRAIN[r.get("experiment", d.name)] = r

def acc(tag, mode="think"):
    s = EVAL.get(tag) or EVAL.get(f"{tag}")
    return f"{s['accuracy']:.1%} (n={s['n']})" if s else "（尚未評估）"

print("█" * 78)
print("█ 研究問題總結（對應 project_plan.md §12.3）")
print("█" * 78)

print("""
【問題 1】一個 LLM 到底需要改變多少才能學會新任務？
  → 用 D 系列（層選擇）與 B 系列（rank）的數據回答
""")
for exp in ["E1_2b_bf16_r16", "B1_2b_bf16_r4", "B3_2b_bf16_r64", "B4_2b_bf16_r128",
            "D1_2b_attn_only", "D2_2b_mlp_only"]:
    t = TRAIN.get(exp)
    if t:
        print(f"   {exp:22s} 可訓練 {100*t.get('trainable_pct',0):.4f}%  "
              f"VRAM {t.get('peak_memory_allocated_gb','?')} GB")
print("""
【問題 2】格式學習 vs 能力學習：微調帶來的是真本事還是只會寫格式？
  → 比較「格式合規率」與「正確率」的上升幅度
""")
for tag in ["2b-base-f16", "2b-base-fewshot", "2b-math-f16"]:
    s = EVAL.get(tag)
    if s:
        print(f"   {tag:18s} 準確率 {s['accuracy']:6.1%}   "
              f"think {s['think_rate']:6.1%}   閉合 {s['closed_think_rate']:6.1%}   "
              f"boxed {s['boxed_rate']:6.1%}   截斷 {s['truncation_rate']:6.1%}")
print("""
   ★ 解讀關鍵：若「正確率」幾乎沒動、但「boxed 合規率」大幅上升，
     結論就應該是「微調只教會了格式」，而不是「模型變聰明了」。
""")
print("【問題 3】規模效應：4B 是否明顯優於 2B？差距是否值得成本？")
print(f"   2B 微調後：{acc('2b-math-f16')}")
print(f"   4B 微調後：{acc('4b-math-f16')}")
t2, t4 = TRAIN.get("E1_2b_bf16_r16", {}), TRAIN.get("E2_4b_bf16_r16", {})
if t2 and t4:
    v2, v4 = t2.get("peak_memory_allocated_gb"), t4.get("peak_memory_allocated_gb")
    if v2 and v4:
        print(f"   VRAM 成本：2B {v2:.2f} GB → 4B {v4:.2f} GB（增加 {v4-v2:+.2f} GB）")

print("""
【問題 4】★ 訓練方法的效率：在固定 16 GB VRAM 下，各方法的記憶體與準確率？
  → 這是本專案對課程「Efficient Fine-Tuning」主題最直接的回應
""")
print("   方法 × VRAM × 可訓練% × 準確率：")
for exp in ["E1_2b_bf16_r16", "E2_4b_bf16_r16", "E3_2b_dora_r16", "E4_4b_dora_r16",
            "E5_2b_qlora_fixed", "E6_4b_qlora_fixed"]:
    t = TRAIN.get(exp)
    if t:
        tag = {"E1_2b_bf16_r16": "2b-math-f16", "E2_4b_bf16_r16": "4b-math-f16",
               "E3_2b_dora_r16": "2b-math-dora", "E4_4b_dora_r16": "4b-math-dora",
               "E5_2b_qlora_fixed": "2b-math-qlora"}.get(exp, "")
        print(f"   {exp:22s} {str(t.get('quant')):6s} "
              f"VRAM {t.get('peak_memory_allocated_gb','?'):>6} GB  "
              f"可訓練 {100*t.get('trainable_pct',0):.4f}%  準確率 {acc(tag) if tag else '—'}")

print("\n   ★ E7：QLoRA 不跳過 linear_attn（受控失敗實驗）")
if E7:
    outcome = E7.get("outcome")
    print(f"     outcome = {outcome}")
    print(f"     解讀：{E7.get('interpretation','')}")
    if outcome == "crashed":
        print("     → 完整 traceback 已存於 results/qlora_crash_traceback.txt（可放進報告附錄）")
    elif outcome in ("survived", "survived_trained"):
        print("     → 請在報告中明確寫出：崩潰的觸發條件是 Unsloth 預量化鏡像，")
        print("       而非 linear_attn 被量化本身。這是比崩潰更有價值的發現。")
else:
    print("     （尚未執行 §9）")

print("""
【問題 5】部署量化的代價：Q4 相對於 bf16/F16 損失多少準確率？值得嗎？
  → 比較同一個模型的不同量化格式（不可跨模型）
""")
def paired_diff(key_a, key_b):
    r = PAIRED.get(f"{key_a}__vs__{key_b}")
    if not r:
        return "（尚未配對）"
    sig = "顯著" if r.get("significant") else "不顯著"
    return (f"差異 {r['diff']:+.1%}（bootstrap 95% CI "
            f"{r['boot_ci95'][0]:+.1%} ~ {r['boot_ci95'][1]:+.1%}），"
            f"McNemar p={r['p_value']:.4f} → {sig}")

print(f"   微調效果（base F16 vs 微調 F16）: {paired_diff('2b-base-f16','2b-math-f16')}")
print(f"   Q8_0 的量化代價              : {paired_diff('2b-math-f16','2b-math-q8')}")
print(f"   Q4_K_M 的量化代價            : {paired_diff('2b-math-f16','2b-math-q4km')}")
print(f"   規模效應（2B vs 4B）         : {paired_diff('2b-math-f16','4b-math-f16')}")

print("\n" + "█" * 78)
print("""
★ 報告中務必寫清楚的兩件事（方法論誠信）

 1. 評估管線是 GGUF，不是 bf16：
    「本專案以 GGUF 格式評估，故所報數字皆為量化後表現；
      F16 為最接近訓練權重的版本，Q8/Q4 的差異即為量化代價。」

 2. 同一列才能互相比較：
      • 「微調有沒有效」→ 比較【同一種量化格式】下的 base vs 微調
      • 「量化掉多少」  → 比較【同一個模型】的不同量化
     拿 base 的 F16 去比微調後的 Q4，會把量化損失誤算成微調的效果。
""")
print("█" * 78)
''')

md(r"""
---
## §14 疑難排解速查

### 環境問題

| 症狀 | 原因 | 解法 |
| --- | --- | --- |
| `CUDA available: False` | 裝到 CPU 版 PyTorch | 用 `--index-url https://download.pytorch.org/whl/cu128` 重裝 |
| `KeyError: 'qwen3_5'` | transformers 太舊 | `pip install "transformers>=5.0.0" --upgrade` |
| `fla` 匯入失敗 | 沒裝 `[cuda]` extra，或 triton 沒先裝 | 先 `pip install triton-windows`，再 `pip install --no-build-isolation "flash-linear-attention[cuda]"` |
| Triton 核心編譯很久 | Qwen3.5 用自訂 Mamba 核心 | **正常現象**，第一次較慢（5–20 分鐘） |
| `No module named 'bitsandbytes'` | 沒裝量化套件 | `pip install bitsandbytes` |

### 訓練問題

| 症狀 | 原因 | 解法 |
| --- | --- | --- |
| Loss 完全不動 | 資料缺 `text` 欄位 | 檢查 JSONL 欄位名稱（§5.2） |
| 輸出沒有 `<think>` | 訓練資料缺思考標籤 | 檢查 §5.2 的格式合規率 |
| 模型一直重複 | 學習率過高或訓練過度 | 降到 `1e-4`，減少 epoch |
| 思考被截斷 | `max_seq_length` 太小 | 提高（依 §5.3 的統計），或過濾過長樣本 |
| 4B 訓練 OOM | 序列長度太大 | 降到 1536 或 1024（§7.4 的順序） |

### GGUF 問題

| 症狀 | 原因 | 解法 |
| --- | --- | --- |
| `missing tensor 'blk.32...'` | 轉換時打包了 MTP 層 | **加 `--no-mtp`**，並用最新版 llama.cpp |
| `missing tensor 'blk.40.ssm_conv1d'` | MTP 區塊被誤判為 recurrent 層 | 更新 llama.cpp |
| `Bad layer N ... Must be in [0, N)` | 舊版量化器層編號錯誤 | 更新 llama.cpp |
| `Unsupported architecture` | transformers 或 llama.cpp 太舊 | 兩者都更新 |
| 轉換時 `_reorder_v_heads` 形狀錯誤 | issue #27019（未修） | 改用**路線 A（Unsloth 匯出）** |
| 載入後輸出亂碼 | 聊天模板不符 | 加 `--jinja`，或指定訓練時的模板 |

### 評估問題（LM Studio）

| 症狀 | 原因 | 解法 |
| --- | --- | --- |
| 連線被拒 | LM Studio 沒開 | 開啟 LM Studio 並保持執行 |
| **推論很慢、GPU 使用率 0%** | **runtime 選成 CPU/Vulkan** | Runtime 面板改成 **CUDA** 並重新載入 |
| 只有部分層在 GPU | `gpu.ratio` 不是 1.0 或 VRAM 不足 | 已設 `{"ratio": 1.0}`；不足請改用 Q4_K_M |
| 找不到模型 | 沒匯入或 model-key 打錯 | `lms ls` 查看；或直接傳 `.gguf` 完整路徑 |
| **所有答案都被判錯** | **忘了傳 `parsing_timeout=None`** | §8.3 會自動偵測；請確認 `MATH_VERIFY_MODE` |
| 評估中途卡住 | 思考太長超過 `max_tokens` | 這是正常的（會被截斷），請看**截斷率** |
| SDK 60 秒逾時 | 同步 API 預設 60 秒無活動逾時 | §8.5 已設 `lms.set_sync_api_timeout(600)` |

---

## §15 交付物清單

- [ ] **LoRA adapter**（至少 6 組，以 bf16 LoRA 與 DoRA 為主）→ `outputs/*/`
- [ ] **每組的 `vram_report.json`**（VRAM 峰值 + 可訓練參數比例）→ `outputs/*/`
- [ ] **訓練 log 與 loss 曲線圖** → `outputs/*/train_log.json`、`results/figures/fig3_*.png`
- [ ] **QLoRA 崩潰的完整錯誤訊息與重現步驟** → `results/qlora_crash_traceback.txt`
- [ ] **評估結果 CSV** → `results/table_eval.csv`
- [ ] **「方法 × VRAM × 準確率」對照總表** → `results/table_training.csv`
- [ ] **配對檢定結果**（McNemar + bootstrap）→ `results/paired_tests.json`
- [ ] **汙染檢查報告** → `results/contamination_report.json`
- [ ] **合併後的 safetensors 權重** → `outputs/merged/`
- [ ] **GGUF F16 / Q8_0 / Q4_K_M** → `gguf/`
- [ ] **效率前緣圖與量化代價圖** → `results/figures/fig1_*.png`, `fig2_*.png`
- [ ] **最終書面報告**

---

## §16 時程建議（四週）

| 週次 | 工作內容 | 對應章節 |
| --- | --- | --- |
| **第 1 週** | 建立 venv 與 CUDA 環境；下載模型與資料；資料前處理；bf16 路徑煙霧測試 | §1–§7.1 |
| **第 2 週** | 完成 2B 與 4B 的 bf16 LoRA；合併、轉 GGUF、匯入 LM Studio；第一組評估數字 | §7.3–§8、§10–§11 |
| **第 3 週** | F16/Q8/Q4 完整評估與配對檢定；base 三條基準線；擴充實驗（DoRA、rank、資料量、層選擇）；重現 QLoRA | §7.4、§8.6–§8.7、§9 |
| **第 4 週** | 整理兩張對照表；繪製效率前緣圖；撰寫報告 | §12–§13 |

> 💡 **最務實的整體建議**（`project_plan.md` §5.6.7）：
> **2B 與 4B 都用 bf16 LoRA，不要用 QLoRA。**
> 理由：兩者都放得進 16 GB；QLoRA 在此架構上有已知問題，修正後也只省約 1.4 GB。
> **QLoRA 的價值在於「當對照組」，而不是「當主力」。**
""")

# ============================================================
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (.venv DSAI4207)",
            "language": "python",
            "name": "dsai4207",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10",
            "mimetype": "text/x-python",
            "file_extension": ".py",
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "codemirror_mode": {"name": "ipython", "version": 3},
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = "full-work-though.ipynb"

# nbformat 4.5 要求 source 是字串、且每個 cell 要有唯一 id
for c in CELLS:
    assert isinstance(c["source"], str), "source must be a string"
    assert c.get("id"), "cell id missing"

with io.open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print(f"wrote {OUT}: {len(CELLS)} cells "
      f"({sum(1 for c in CELLS if c['cell_type']=='markdown')} markdown, "
      f"{sum(1 for c in CELLS if c['cell_type']=='code')} code)")
