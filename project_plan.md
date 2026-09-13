# 課程專案操作手冊：用 SFT 微調小型 LLM 成為「數學高手」

> 本文件是一份**從零到完成**的逐步操作手冊，目標是讓你在自己的電腦上，
> 以 Qwen3.5-2B-Base 與 Qwen3.5-4B-Base 為基礎模型，透過監督式微調（SFT）
> 搭配 LoRA，訓練出可以「先思考再作答」的數學解題模型，並完成評估、
> 合併權重、轉成 GGUF、量化成 Q8 與 Q4 的完整流程。
>
> 閱讀順序：先看第 0 章確認你的電腦能不能做，再看第 1 章的前置作業，
> 之後照著章節順序做即可。每一章都有「驗收標準」，做完請自我檢查。

---

## 0.0 本手冊的執行環境（重要，先讀）

本專案**全程在 Windows 上執行**，並使用**兩套不同的環境**，
各自負責它擅長的事。請先建立這個心智模型，否則後面會混亂：

| 環境 | 用途 | 為什麼是它 |
| --- | --- | --- |
| **Python venv（第 2 章建立）** | **訓練**：資料前處理、LoRA 微調、合併權重、GGUF 轉換與量化 | 需要 PyTorch 的 CUDA 反向傳播。**LM Studio 不能訓練**，這是硬限制 |
| **LM Studio（第 7 章設定）** | **評估**：載入 GGUF 模型、產生解答、計算準確率 | 你要的評估後端。它用 **CUDA runtime** 在 GPU 上做推論 |

**完整流程與使用的環境：**

```
[venv]      第 4 章   資料前處理
[venv]      第 6 章   訓練 LoRA / QLoRA
[venv]      第 8 章   合併 adapter → bf16 safetensors
[venv]      第 9 章   轉成 GGUF（F16）
[venv]      第 10 章  量化成 Q8_0 與 Q4_K_M
     ↓
[LM Studio] 第 7 章   載入上述 GGUF，用 CUDA runtime 評估
     ↓
[venv]      第 11 章  整理結果、量化前後對照
```

> ⚠️ **一個必須先講清楚的後果：**
> 因為 **LM Studio 只能載入 GGUF**，用 LM Studio 評估就代表
> **你評估的是「量化後」的模型，而不是訓練出來的 bf16 原版**。
>
> 這不是缺陷，而是一個**必須在報告中誠實說明的方法論限制**。
> 本手冊的處理方式是：
> 1. 用 **F16 GGUF** 當作「最接近 bf16 原版」的評估版本
> 2. 同時比較 **F16 / Q8_0 / Q4_K_M**，這樣你可以**分離**
>    「微調帶來的效果」與「量化帶來的損失」——
>    後者本身就是課程要的量化主題
> 3. 在報告中明確寫出「評估管線是 GGUF，而非 bf16」

---

## 目錄

- [0.0 本手冊的執行環境（重要，先讀）](#00-本手冊的執行環境重要先讀)
- [0. 先確認可行性：硬體與現實檢查](#0-先確認可行性硬體與現實檢查)
  - [0.2 VRAM 預算：實測數字與方法的選擇](#02-vram-預算實測數字與方法的選擇)
  - [0.2.1 ⚠️ QLoRA 在 Qwen3.5 上目前是壞的](#021-️-qlora-在-qwen35-上目前是壞的)
- [1. 前置作業：帳號、工具、硬碟規劃](#1-前置作業帳號工具硬碟規劃)
- [2. 建立 Python 虛擬環境（venv）](#2-建立-python-虛擬環境venv)
- [3. 基礎模型：Qwen3.5-2B-Base 與 4B-Base 的真實規格](#3-基礎模型qwen35-2b-base-與-4b-base-的真實規格)
- [4. 資料集選擇與資料前處理](#4-資料集選擇與資料前處理)
- [5. 訓練設定：LoRA / QLoRA 超參數與實驗矩陣](#5-訓練設定lora--qlora-超參數與實驗矩陣)
  - [5.3 怎麼選方法：決策樹](#53-怎麼選方法決策樹)
  - [5.6 ⚠️ QLoRA 對 Qwen3.5 目前是壞的（附解法）](#56-️-重要警訊qlora-對-qwen35-目前是壞的附解法)
- [6. 開始訓練](#6-開始訓練)
- [7. 評估：使用 LM Studio（CUDA runtime）](#7-評估使用-lm-studio-cuda-runtime)
  - [7.0 ⚠️ 評估的順序與限制](#70-️-評估的順序以及一個必須承認的限制)
  - [7.5 評估環境設定：LM Studio + CUDA](#75-評估環境設定lm-studio--cuda)
  - [7.6 評估腳本（透過 LM Studio）](#76-評估腳本透過-lm-studio)
  - [7.7 跑評估](#77-跑評估)
- [8. 合併權重與輸出](#8-合併權重與輸出)
- [9. 轉成 GGUF](#9-轉成-gguf)
- [10. 量化成 Q8_0 與 Q4_K_M](#10-量化成-q8_0-與-q4_k_m)
- [11. 量化後驗證與推論](#11-量化後驗證與推論)
- [12. 時程建議與交付物清單](#12-時程建議與交付物清單)
- [13. 疑難排解](#13-疑難排解)
- [附錄 A：常用指令速查](#附錄-a常用指令速查)
- [附錄 B：名詞對照表](#附錄-b名詞對照表)
- [附錄 C：參考資料](#附錄-c參考資料)

---

## 0. 先確認可行性：硬體與現實檢查

### 0.1 你的機器規格

| 項目 | 你的規格 | 對本專案的意義 |
| --- | --- | --- |
| CPU | 8 核心 | 夠用。CPU 主要負責資料前處理與 GGUF 轉換，不是瓶頸。 |
| RAM | 64 GB | 非常充足。合併權重、轉 GGUF、量化都在 RAM 中進行，64 GB 綽綽有餘。 |
| GPU | 16 GB VRAM、支援 CUDA | **這是唯一的關鍵限制**。決定你能開多大的模型、序列長度與 batch size。 |
| 儲存 | 建議預留 **150 GB** 以上可用空間 | 見下方 0.3 的容量估算。 |

> ⚠️ **你的 GPU 會被兩套環境分別使用，兩邊都要驗證：**
>
> | 環境 | 驗證方式 | 章節 |
> | --- | --- | --- |
> | **venv（訓練）** | `torch.cuda.is_available()` 必須是 `True` | 第 2.3 節 |
> | **LM Studio（評估）** | Runtime 選 **CUDA**，且 GPU offload 為全部層數 | 第 7.5.1 節 |
>
> **只驗證一邊不算完成。** 常見的錯誤是訓練環境正常，
> 但 LM Studio 預設用了 CPU 或 Vulkan，導致評估慢上 10 倍。

### 0.2 VRAM 預算：實測數字與方法的選擇

**先講結論，因為它和很多網路教學相反：**

> **本專案的預設方法是 bf16 LoRA，不是 QLoRA。**
>
> 原因有兩個，都是硬事實：
> 1. **QLoRA 在 Qwen3.5 上目前是壞的**（見 5.6 節，有兩個未修的官方 issue）。
> 2. **4B 的 bf16 LoRA 實測只需要約 9.5 GB，你的 16 GB 放得下。**

訓練時的 VRAM 大致是：

```
VRAM ≈ 凍結的基礎權重 + 梯度 + 優化器狀態 + 活化值（activations）
```

#### 各方法的「每參數」記憶體（實測公式，不是估算）

| 方法 | 基礎權重 | 說明 |
| --- | --- | --- |
| **bf16 LoRA** | **2 bytes/參數** | 官方 bf16 檢查點就是 4.66e9 × 2 = 9.32 GB |
| **QLoRA（4-bit NF4 + 雙重量化）** | **約 0.516 bytes/參數** | 4 bits + 雙重量化後的常數 0.127 bits |
| QLoRA（關掉雙重量化） | 約 0.5625 bytes/參數 | 省下的是 **0.373 bits/參數** |
| 8-bit（LLM.int8()） | 1 byte/參數 | 另有每列 fp32 縮放常數 |

**優化器：對 LoRA 而言幾乎不重要。** 因為 LoRA 只訓練 2,000 萬～4,000 萬個參數：

| 優化器 | 每參數 | 對 4B 的 r=16 adapter（2,120 萬參數）的影響 |
| --- | --- | --- |
| `adamw_torch`（預設，fp32） | 8 bytes | 約 0.17 GB |
| **`adamw_8bit`** | 2.031 bytes | 約 0.04 GB |
| `adafactor` | 幾乎 0（2D 矩陣） | 約 0 GB |

**所以優化器的差別在 0.3 GB 以內，是「捨入誤差」。**
真正吃記憶體的是**凍結的基礎權重**。這也意味著：
**不要花時間糾結優化器，要糾結「基礎權重要不要量化」。**

#### 實測總量（weights + adapter + 梯度 + 優化器，未含活化值）

| 方法 | 2B（2.27B） | 4B（4.66B） | 16 GB 可行？ |
| --- | --- | --- | --- |
| **全參數微調**（`adamw_torch`） | 27.2 GB | **55.9 GB** | ❌ 不可行 |
| 全參數 + `adamw_8bit` | 13.7 GB | 28.1 GB | ❌ 4B 不可行 |
| 全參數 + `adafactor` | ~9.3 GB | ~19.1 GB | ❌ 4B 不可行 |
| **★ bf16 LoRA（r=16）** | **4.61 GB** | **9.45 GB** | ✅ **兩者都可行** |
| bf16 LoRA（r=32） | 4.72 GB | 9.66 GB | ✅ 可行 |
| QLoRA（r=16）**若可用** | ~2.96 GB | ~4.64 GB | ✅ 但**目前在此架構上是壞的** |
| 8-bit LoRA（r=16） | ~3.8 GB | ~6.6 GB | ⚠️ 見下方註解 |
| **DoRA（r=16）** | ≈ bf16 LoRA + 0.002 GB | ≈ bf16 LoRA + 0.003 GB | ✅ 成本幾乎為零 |

> ⚠️ **關於 8-bit LoRA**：理論上可行（基礎權重 1 byte/參數），
> 但 **Unsloth 只提供 4-bit 與 16-bit 兩條路徑，沒有 8-bit**。
> 要真正做 8-bit LoRA 必須自行建立 `BitsAndBytesConfig` 並傳入，
> 屬未經測試的領域。**本專案不把它列為正式實驗組。**

> **活化值（activations）**
> 開啟 gradient checkpointing 後，只需保存層邊界的活化值：
> 2B 在 seq 2048 時約 **0.2 GB**、4B 約 **0.34 GB**。
> 不開的話會膨脹 10–20 倍（2B 約 3.4 GB、4B 約 5.7 GB）。
> **所以 gradient checkpointing 一定要開**，但請注意：
> **在你這個規模，活化值不是瓶頸，凍結權重才是。**
> 序列長度拉到 4096 時，4B 的活化值也只約 0.67 GB——仍然放得下。

#### 三個結論

**結論一：全參數微調不可行。** 4B 需要約 56 GB，2B 需要約 27 GB。
這是數量級差距，不是調參數能解決的。

**結論二：4B 的 bf16 LoRA 可行，而且這是最好的選擇。**
9.45 GB 的權重 + 約 0.34 GB 的活化值 ≈ 10 GB，在 16 GB 內**還有 6 GB 餘裕**。
官方自己的 Qwen3.5-4B 筆記本就是在**一張 14.56 GB 的免費 T4 上跑 16-bit LoRA**。
**你不需要為了 4B 而冒 QLoRA 的風險。**

**結論三：QLoRA 目前不該作為主力。** 見下一節。

### 0.2.1 ⚠️ 重要更正：QLoRA 在 Qwen3.5 上目前是壞的

如果你在網路上看到「用 QLoRA 省 VRAM 跑 Qwen3.5」，請注意：
**那個做法目前在這個模型上是行不通的，而且不是設定問題，是程式碼 bug。**
完整說明與解法在 **5.6 節**。這裡先給你三句話版本：

1. Qwen3.5 的線性注意力層（Gated DeltaNet）在 4-bit 路徑上會把
   **打包好的 4-bit 權重直接送進 `F.linear`**，導致
   `mat1 and mat2 shapes cannot be multiplied` 而當場崩潰。
2. 官方（Unsloth）文件本身就寫明**不建議**對 Qwen3.5 做 QLoRA，
   並說它們自己的 4B 範例是 **16-bit LoRA**。
3. **有解法**（把線性注意力層排除在量化之外），但代價不小、且需要繞過
   Unsloth 的預量化模型鏡像。所以本專案把它列為**進階可選實驗**，
   而不是預設方法。

**本專案的預設配置：**

| 模型 | 預設方法 | 實測 VRAM | 備註 |
| --- | --- | --- | --- |
| **2B** | **bf16 LoRA** | 約 4.6 GB | 非常寬鬆，可拉長序列 |
| **4B** | **bf16 LoRA** | 約 9.5 GB | 官方在 14.5 GB T4 上實測可行 |
| （進階） | DoRA | ≈ 同上 | 幾乎零額外成本，值得加做 |
| （進階） | QLoRA | 約 3.0 / 4.6 GB | **需套用 5.6.3 的修正才可能運作** |
| （進階） | HQQ 4-bit | 未實測 | 真正支援 LoRA 的 4-bit 路線，混合架構未驗證 |

### 0.2.2 五種方法的一句話總結

| 方法 | 一句話 | 本專案的定位 |
| --- | --- | --- |
| **FFT** 全參數微調 | 更新全部權重，效果上限最高，但記憶體需求最大 | 你的硬體不適用 |
| **LoRA** | 凍結原權重，只訓練小的低秩矩陣 | **★ 預設方法（2B 與 4B）** |
| **DoRA** | LoRA 的改良版，把權重拆成「大小」與「方向」分別調整 | 幾乎免費的進階對照 |
| **QLoRA** | 把凍結的原權重壓到 4-bit，再掛 LoRA | **進階對照；標準版本會崩潰，需修正** |
| **HQQ** | 另一種 4-bit 量化，支援逐層位元數 | 探索性替代路線（未經混合架構驗證） |

### 0.2.3 其他省 VRAM 的技巧（可疊加）

| 技巧 | 省下的東西 | 代價 | 建議 |
| --- | --- | --- | --- |
| **Gradient checkpointing** | 活化值（省 10–20 倍） | 訓練慢約 20% | **一定要開** |
| **8-bit 優化器**（`adamw_8bit`） | 優化器狀態（8→2 bytes/參數） | 幾乎無 | 開（但對 LoRA 只省 0.1 GB） |
| **paged 優化器**（`paged_adamw_8bit`） | OOM 時把狀態分頁到 CPU RAM | 略慢 | OOM 時改用 |
| **較小的序列長度** | 活化值（與序列長度成正比） | 長思維鏈被截斷 | 先 2048，有餘裕再拉到 4096 |
| **梯度累積**（不用大 batch） | 活化值 | 訓練變慢 | 用 `accum` 補等效 batch size |
| **關閉視覺層微調** | 視覺塔的權重與活化值 | 無（數學不需要圖片） | **一定要關** |

> 💡 **你有 64 GB RAM，這是很大的優勢。**
> 若真的 OOM，改用 `paged_adamw_8bit` 可把優化器狀態分頁到系統記憶體。
>
> ⚠️ **但要注意：對 LoRA 而言優化器只佔 0.04–0.17 GB，分頁救不了多少。**
> 真正有效的手段是**降低序列長度**或**減少可訓練模組**。

### 0.3 硬碟容量估算

| 檔案 | 大小（估計） |
| --- | --- |
| Qwen3.5-2B-Base 權重（bf16） | 約 4.6 GB |
| Qwen3.5-4B-Base 權重（bf16） | 約 9.4 GB |
| 訓練資料集（SFT，含思維鏈） | 約 5–40 GB（依你選的子集而定） |
| 每個實驗的 LoRA adapter | 約 50–400 MB |
| QLoRA 的量化權重（不會額外存檔） | 0（只存在於訓練時的 VRAM） |
| 合併後的 bf16 權重 | 每個約 5–10 GB |
| GGUF F16 中繼檔 | 每個約 5–10 GB |
| GGUF Q8_0 | 每個約 2.5–5 GB |
| GGUF Q4_K_M | 每個約 1.4–2.8 GB |
| HF 快取（兩個模型 + tokenizer） | 約 15 GB |

**建議：準備 150 GB 以上可用空間。** 若空間吃緊，可在轉檔後刪除 F16 中繼檔
（但建議至少保留一份，方便日後重新量化成其他精度）。

### 0.4 時間估算

以單張 16 GB 顯卡、8 核心 CPU 為基準：

| 階段 | 2B 模型 | 4B 模型 |
| --- | --- | --- |
| 下載模型與資料 | 20–60 分鐘 | 40–90 分鐘 |
| 資料前處理（tokenize） | 10–30 分鐘 | 20–60 分鐘 |
| 一輪 LoRA 訓練（5k 樣本、1 epoch） | 30–90 分鐘 | 1.5–3 小時 |
| 一輪 QLoRA 訓練（同上） | 40–110 分鐘 | 2–4 小時（4-bit 解量化較慢） |
| 評估（3 個 benchmark × 2 個模型） | 1–3 小時 | 2–5 小時 |
| 合併 + 轉 GGUF + 量化 | 20–40 分鐘 | 30–60 分鐘 |

> 提醒：**第一次**執行時，Qwen3.5 需要編譯自訂的 Mamba/Triton 核心，
> 會額外花 5–20 分鐘。這是正常的，之後就不會了。

### 0.5 驗收標準

- [ ] 你已知道自己的 GPU 型號與 VRAM 大小（用 `nvidia-smi` 查）
- [ ] 你已理解「全參數微調不可行，本專案改用參數高效微調（LoRA / QLoRA / 8-bit / DoRA）」
- [ ] 你已決定 **2B 與 4B 都用 bf16 LoRA**（4B 實測約 9.5 GB，放得下）
- [ ] 你的硬碟有 150 GB 以上可用空間

---

## 1. 前置作業：帳號、工具、硬碟規劃

### 1.1 需要安裝的軟體

| 軟體 | 用途 | 備註 |
| --- | --- | --- |
| Python 3.11 或 3.12 | 執行訓練與轉檔腳本 | **不要用 3.13**，部分套件尚未支援 |
| Git | 下載 llama.cpp | 轉 GGUF 用 |
| NVIDIA 驅動程式 | CUDA 支援 | 請更新到支援 CUDA 12.x 以上的版本 |
| Visual C++ Build Tools（可選） | 編譯 bitsandbytes 或 llama.cpp | 若 pip 安裝失敗才需要 |
| CMake（可選） | 自行編譯 llama.cpp | 若使用預編譯版本則免 |
| 7-Zip 或同等工具 | 解壓縮 | 可選 |

**你目前的環境**：已偵測到 Python 3.12.10，可直接使用。

### 1.2 需要申請的帳號

| 帳號 | 為什麼需要 | 是否強制 |
| --- | --- | --- |
| Hugging Face 帳號 | 下載模型與資料集；若要把成果上傳 | 建議申請（免費） |
| Hugging Face Access Token | 部分資料集需同意條款後才能下載 | 依資料集而定 |

申請後，在本機登入：

```powershell
# 在已啟用 venv 的環境下執行
hf auth login
# 或舊版指令：huggingface-cli login
```

貼上你的 Access Token（需勾選 `read` 權限；若要上傳模型再加 `write`）。

> 💡 **中國大陸網路環境提醒**：若下載 Hugging Face 很慢，可設定鏡像端點：
> ```powershell
> $env:HF_ENDPOINT = "https://hf-mirror.com"
> ```
> 這只影響下載，不影響訓練結果。

### 1.3 建立專案目錄結構

建議建立以下結構，之後所有指令的路徑都以此為準：

```
DSAI4207\
├── project_plan.md            # 本文件
├── Proposal.md                # 提案書
├── requirements.txt           # 套件清單
├── .venv\                     # Python 虛擬環境（第 2 章建立）
├── data\
│   ├── raw\                   # 原始下載的資料集
│   └── processed\             # 清理並轉成訓練格式後的資料
├── scripts\
│   ├── prepare_data.py        # 資料前處理
│   ├── train_lora.py          # 訓練腳本
│   ├── evaluate.py            # 評估腳本
│   ├── merge_adapter.py       # 合併 LoRA 權重
│   └── make_gguf.ps1          # 轉 GGUF 與量化
├── outputs\
│   ├── qwen35-2b-lora-r16\    # 每個實驗一個資料夾
│   ├── qwen35-2b-lora-r64\
│   └── merged\                # 合併後的完整權重
├── gguf\                      # GGUF 與量化檔
└── results\                   # 評估結果（CSV / JSON / 圖表）
```

建立指令（PowerShell）：

```powershell
cd D:\DSAI\DSAI4207
New-Item -ItemType Directory -Force -Path data\raw, data\processed, scripts, outputs, gguf, results
```

### 1.4 驗收標準

- [ ] 已安裝 Python 3.11/3.12 與 Git
- [ ] 已完成 `hf auth login`
- [ ] 已建立上述目錄結構

---

## 2. 建立 Python 虛擬環境（venv）

> ⚠️ **本章建立的 venv 只用於「訓練與轉檔」。**
> **評估**會透過 LM Studio（第 7 章），那是完全獨立的另一套環境。
> 兩者不衝突，但請記住誰負責什麼（見 0.0 節）。
>
> 所有指令都是 **Windows PowerShell**。
> 若你使用的是 CMD，請把行尾的 `` ` ``（續行符號）改成 `^`。

**為什麼一定要用 venv？** 因為 PyTorch、CUDA 版本、transformers、unsloth 之間的
相容性非常脆弱。用 venv 可以讓這個專案的環境與你系統上其他 Python 專案隔離，
出事時直接刪掉重建即可。

### 2.1 建立與啟用虛擬環境

```powershell
cd D:\DSAI\DSAI4207

# 建立虛擬環境（會產生 .venv 資料夾）
python -m venv .venv

# 啟用（PowerShell）
.\.venv\Scripts\Activate.ps1

# 若出現「執行原則禁止指令碼」錯誤，先執行這行再重試：
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

啟用成功後，命令提示字元前方會出現 `(.venv)`。

> 每次開新終端機都要重新執行一次 `Activate.ps1`。

### 2.2 升級基礎工具

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 2.3 安裝 PyTorch（CUDA 版本）

**這是最關鍵的一步**：預設的 `pip install torch` 可能裝到 CPU 版本，
導致訓練時完全用不到 GPU。

先到 [PyTorch 官方網站](https://pytorch.org/get-started/locally/) 確認目前建議的
CUDA 版本與安裝指令（會隨時間變動）。以 CUDA 12.4 為例：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**安裝後務必驗證**：

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1) if torch.cuda.is_available() else 0)"
```

**驗收標準：必須印出 `CUDA available: True` 以及你的 GPU 名稱。**
若印出 `False`，請不要繼續，先解決驅動或 CUDA 版本問題。

### 2.4 安裝訓練框架

本手冊以 **Unsloth** 為主，原因是官方明確支援 Qwen3.5 全系列，且針對這個
架構做過最佳化（比標準 FA2 設定快約 1.5 倍、省約 50% VRAM），並內建 GGUF 匯出。

```powershell
pip install --upgrade unsloth unsloth_zoo
```

> ⚠️ **Qwen3.5 必須使用 `transformers v5`**，舊版無法辨識 `qwen3_5` 架構。
> Unsloth 現在預設會自動使用 v5。若你遇到架構辨識錯誤，手動安裝：
> ```powershell
> pip install "transformers>=5.0.0" --upgrade
> ```

接著安裝訓練與評估相關套件：

```powershell
pip install trl peft accelerate datasets bitsandbytes
pip install "math-verify[antlr4_13_2]"   # 數學答案等價性判定（第 7 章使用）
pip install pandas matplotlib             # 結果整理與繪圖
```

> 💡 `math-verify` 官方建議**固定 antlr 執行環境版本**（`[antlr4_13_2]`），
> 避免 LaTeX 解析器版本飄移造成結果不一致。

### 2.5 凍結環境（重要！）

確認一切正常後，把版本鎖定下來，避免之後重裝時版本漂移：

```powershell
pip freeze > requirements.txt
```

### 2.6 備份這個環境的方法

若環境搞爛了想重來：

```powershell
deactivate                  # 先離開
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2.7 驗收標準

- [ ] `(.venv)` 出現在終端機提示字元前
- [ ] `torch.cuda.is_available()` 回傳 `True`
- [ ] `python -c "import unsloth; print(unsloth.__version__)"` 沒有錯誤
- [ ] `requirements.txt` 已產生

---

## 3. 基礎模型：Qwen3.5-2B-Base 與 4B-Base 的真實規格

> 本章的數字來自 Hugging Face 上的 `config.json`、`tokenizer_config.json` 與
> `model.safetensors.index.json` 實際內容，不是猜測。

### 3.1 模型基本資訊

| 項目 | Qwen3.5-2B-Base | Qwen3.5-4B-Base |
| --- | --- | --- |
| Hugging Face ID | `Qwen/Qwen3.5-2B-Base` | `Qwen/Qwen3.5-4B-Base` |
| 架構（architectures） | `Qwen3_5ForConditionalGeneration` | 同左 |
| 模型類型 | `qwen3_5` | `qwen3_5` |
| 實際參數量 | 約 2.27 B | 約 4.66 B |
| 權重檔案大小（bf16） | 約 4.55 GB（單檔） | 約 9.32 GB（兩檔） |
| 語言模型層數 | 24 | 32 |
| 隱藏維度 | 2048 | 2560 |
| 詞表大小 | 248,320 | 248,320 |
| 原生上下文長度 | 262,144 tokens | 262,144 tokens |
| 授權 | Apache-2.0 | Apache-2.0 |
| 是否需同意條款 | 否（公開） | 否（公開） |

### 3.2 這是「混合架構」，不是標準 Transformer

**這是本專案最重要的技術細節，會直接影響你選擇工具與解讀結果。**

Qwen3.5 的語言模型採用了 3:1 的混合注意力配置：

- **每 4 層中有 3 層是「線性注意力」**（Gated DeltaNet，一種狀態空間／線性注意力機制）
- **每 4 層中有 1 層是「完整注意力」**（Gated Attention，標準 softmax 注意力）

以 2B 為例，`config.json` 中的 `layer_types` 是：

```
linear_attention, linear_attention, linear_attention, full_attention,
linear_attention, linear_attention, linear_attention, full_attention,
...（共 24 層，最後一層是 full_attention）
```

其他關鍵設定：

| 設定 | 2B | 4B | 說明 |
| --- | --- | --- | --- |
| `full_attention_interval` | 4 | 4 | 每隔 4 層一個完整注意力層 |
| `intermediate_size` | 6144 | 9216 | FFN 中間維度 |
| `num_attention_heads` | 8 | 16 | 完整注意力的 Q 頭數 |
| `num_key_value_heads` | 2 | 4 | GQA 的 KV 頭數 |
| `head_dim` | 256 | 256 | 注意力頭維度 |
| `linear_num_key_heads` | 16 | 16 | 線性注意力的 K 頭數 |
| `linear_num_value_heads` | 16 | 32 | 線性注意力的 V 頭數 |
| `mamba_ssm_dtype` | float32 | float32 | SSM 狀態用 fp32 |
| `mtp_num_hidden_layers` | 1 | 1 | 多 token 預測層（見 3.4） |
| `tie_word_embeddings` | true | true | 輸入嵌入與輸出層共用權重 |

**對你的影響：**

1. **這不是 MoE 模型。** 2B 與 4B 的 FFN 是標準 dense 結構
   （`intermediate_size` 為單一數值、`config.json` 中沒有 expert 相關欄位）。
   網路上說「Qwen3.5 是 MoE」指的是 35B-A3B、122B-A10B 等大型號，不是你要用的這兩個。
2. **`mamba_ssm_dtype` 是 float32**，代表線性注意力層的狀態需要較高精度。
   這就是為什麼官方**不建議對 Qwen3.5 做 4-bit 量化訓練**（見 5.3 節）。
3. 工具鏈相對新，遇到套件不相容時要優先懷疑「是否支援混合架構」。

### 3.3 內建視覺編碼器（比你想像的重要）

這兩個 Base 模型都是**視覺語言模型（VLM）**，`pipeline_tag` 是
`image-text-to-text`，權重中包含 `model.visual.*` 的視覺塔（vision tower）。

**這對純文字數學微調的影響：**

- 你**不需要**視覺能力，但視覺編碼器的權重會一起被下載（2B 約 0.4 GB、4B 更多）。
- 訓練時**應明確關閉視覺層的微調**，只調整語言層，理由：
  - 節省 VRAM 與時間
  - 數學資料沒有圖片，訓練視覺層毫無意義且可能造成災難性遺忘
  - 之後匯出 GGUF 時，純文字用途不需要 `mmproj` 檔案
- 若你不小心讓視覺層參與訓練，模型可能忘記原本的視覺能力。

### 3.4 MTP 層（Multi-Token Prediction）

`config.json` 中的 `mtp_num_hidden_layers: 1`，權重檔中也有 `mtp.*` 開頭的張量。
這是用來做「投機解碼」（speculative decoding）的額外預測頭，可以加速推論。

**注意**：這個額外的 MTP 層，正是後面 GGUF 轉換出問題的根源之一
（llama.cpp 的載入器會把層數算成「主幹層數 + MTP 層數」，
但轉換腳本不一定會寫出對應的張量）。詳見第 9 章。

### 3.5 Base 模型已經「看得懂」思考標籤

這是本專案最幸運的一點：**Qwen3.5 的 Base 模型 tokenizer 已經內建思考模式的
聊天模板**。

從 `tokenizer_config.json` 可以確認：

- 詞表中已有 `<think>` 與 `</think>` 兩個特殊 token（id 248068、248069）
- 聊天模板支援 `enable_thinking` 參數：
  - `enable_thinking=True` → 產生 `<|im_start|>assistant\n<think>\n`，模型開始思考
  - `enable_thinking=False` → 產生 `<|im_start|>assistant\n<think>\n\n</think>\n\n`，直接作答
- 模板會自動把 assistant 訊息中的 `</think>` 之前內容抽出來當作 `reasoning_content`
- 控制 token `<|im_start|>`、`<|im_end|>` 在預訓練時已被訓練過，
  官方明說這是為了「讓 LoRA 風格的 PEFT 更有效率，不需要微調嵌入層」

**實務結論：**

- 你的訓練資料應該格式化為：**使用者提問 → `<think>推理過程</think>` → 最終答案**
- 這正好對應「有 thinking tag 的資料集」的需求
- 官方建議：若你想**保留**推理能力，訓練資料中至少要有 **75% 是推理風格**
  的樣本；其餘可以混入直接作答的樣本

> ⚠️ **踩雷警告**：不同 Qwen3.5 版本的聊天模板對 `enable_thinking` 的判斷邏輯
> 不一致。例如 2B 的模板寫 `enable_thinking is true`，4B 的模板寫
> `enable_thinking is false`（兩者行為剛好相反）。
> **因此本手冊一律不依賴 `enable_thinking` 參數**，而是**直接手動拼出完整字串**
> 再送去 tokenize（見 4.3 節）。這樣可以完全避開模板差異帶來的意外。

### 3.6 下載模型

```powershell
# 只下載 2B（先做小規模驗證）
hf download Qwen/Qwen3.5-2B-Base --local-dir D:\DSAI\DSAI4207\models\Qwen3.5-2B-Base

# 4B（較大，約 9.4 GB）
hf download Qwen/Qwen3.5-4B-Base --local-dir D:\DSAI\DSAI4207\models\Qwen3.5-4B-Base
```

若不想佔用兩份空間，可改用 Hugging Face 快取（不指定 `--local-dir`，
直接在程式碼中用模型 ID，快取會放在 `~\.cache\huggingface`）。

### 3.7 驗收標準

- [ ] 能用程式載入模型並看到正確的層數（2B 為 24、4B 為 32）
- [ ] 確認 tokenizer 中存在 `<think>` 這個 token
- [ ] 知道自己的模型是 VLM，且打算只微調語言層

驗證程式碼：

```python
from transformers import AutoTokenizer, AutoConfig

for name in ["Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-4B-Base"]:
    cfg = AutoConfig.from_pretrained(name)
    tok = AutoTokenizer.from_pretrained(name)
    print(name)
    print("  語言層數:", cfg.text_config.num_hidden_layers)
    print("  layer_types 前 8 個:", cfg.text_config.layer_types[:8])
    print("  MTP 層數:", cfg.text_config.mtp_num_hidden_layers)
    print("  <think>  token id:", tok.convert_tokens_to_ids("<think>"))
    print("  </think> token id:", tok.convert_tokens_to_ids("</think>"))
    # 這兩個應該回傳 unk（未定義）以確認你真的用的是 Qwen3.5、不是 Qwen3
    print("  Qwen3 舊格式  thinking id:", tok.convert_tokens_to_ids(" thinking"))
```

**驗收重點：**

- `num_hidden_layers` 應為 24（2B）/ 32（4B）
- `mtp_num_hidden_layers` 應為 1（這就是 GGUF 轉換要用 `--no-mtp` 的原因）
- `<think>` 與 `</think>` 應該要能對應到有效 id（248068 / 248069）
- **若你的程式碼裡看到 ` thinking` 這個 token，那是 Qwen3 的舊格式，不是 Qwen3.5。**
  Qwen3.5 用的是 `<think>`（沒有那個 `|` 直線符號）。

---

## 4. 資料集選擇與資料前處理

### 4.1 選資料集的三個硬性條件

你不想要自己建資料集，所以必須從公開資料集中挑。請用以下條件篩選：

| 條件 | 為什麼 |
| --- | --- |
| **必須已包含推理過程（thinking / CoT 痕跡）** | 你要訓練的是「會思考的數學模型」。只有「題目+答案」的資料集（如原始 GSM8K）無法教會模型思考。 |
| **最終答案必須可自動驗證抽取** | 評估時要自動比對答案，通常要求推理結尾有 `\boxed{...}` 或明確的答案欄位。 |
| **大小可負擔** | 你的 16 GB VRAM 單卡，實用範圍是數千到數萬筆樣本。 |

### 4.2 候選資料集

以下皆出自你提供的
[awesome-AI-Math-Datasets](https://github.com/amao0o0/awesome-AI-Math-Datasets) 清單，
且都**已包含推理痕跡**：

| 資料集 | HF ID | 內容 | 是否可驗證答案 | 備註 |
| --- | --- | --- | --- | --- |
| **OpenR1-Math-220k** | `open-r1/OpenR1-Math-220k` | 有 3 個 config：`default` = 93,733 筆、`extended` = 131,396 筆、`all` = 225,129 筆。每題附 2–4 條 DeepSeek-R1 推理痕跡 | 是（`\boxed{}`，且附官方逐條驗證結果） | **最推薦**。由 NuminaMath 1.5 衍生，SFT 導向，且提供 `correctness_math_verify` 可過濾品質 |
| **OpenMathReasoning** | `nvidia/OpenMathReasoning` | 306k 題（AoPS 論壇），含 CoT 與工具整合推理（TIR） | 是 | 品質高但檔案很大，需挑子集 |
| **DeepMath-103K** | `zwhe99/DeepMath-103K` | 約 103k 高難度題（難度 5–9），每題 3 條 R1 解 | 是 | 偏難，適合當「進階」實驗 |
| **NuminaMath-CoT** | `AI-MO/NuminaMath-CoT` | 約 860k 題，含 CoT 解 | 部分是 | 量大但格式較雜，需清理 |
| **Bespoke-Stratos-17k** | `bespokelabs/Bespoke-Stratos-17k` | 17k 題，DeepSeek-R1 痕跡蒸餾 | 是 | 小、乾淨，適合第一次跑通流程 |

> 📌 **建議的選擇策略**
>
> - **第一次跑通流程（先求有）**：用 `Bespoke-Stratos-17k`，
>   或 `OpenR1-Math-220k` 的 `default` config 取前 5,000 筆。
> - **正式實驗（求好）**：用 `OpenR1-Math-220k`，這是目前社群最廣泛用於
>   數學 SFT 的「帶思考」資料集，而且它**內建逐條驗證結果**，
>   讓你可以只保留「答案確實正確」的樣本——這對小模型特別重要。
> - **進階對照（求深）**：再用 `DeepMath-103K` 做一次，觀察「難題訓練」是否
>   反而傷害簡單題表現。
>
> ⚠️ **下載容量提醒**：`OpenR1-Math-220k` 的 `all` config 下載約 4.2 GB、
> 解壓後約 10 GB。若硬碟吃緊，先用 `default`（下載約 2.1 GB、解壓約 5 GB）。

**各資料集的實際欄位（已核實，方便你寫轉換程式）**

| 資料集 | 筆數 | 實際欄位 |
| --- | --- | --- |
| `OpenR1-Math-220k`（`default`） | 93,733 | `problem`, `solution`, `answer`, `problem_type`, `question_type`, `source`, `uuid`, `is_reasoning_complete`(list), `generations`(list), `correctness_math_verify`(list), `correctness_llama`(list), `finish_reasons`(list), `correctness_count`, `messages` |
| `DeepMath-103K` | 103,022 | `question`, `final_answer`, `difficulty`(float), `topic`, `r1_solution_1`, `r1_solution_2`, `r1_solution_3` |
| `Bespoke-Stratos-17k` | 16,710 | `system`, `conversations`（`[{"from": ..., "value": ...}]`） |

`Bespoke-Stratos-17k` 的格式很適合快速跑通流程，因為它已經接近聊天格式：

```python
from datasets import load_dataset
ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")

def format_row(ex):
    """把 conversations 轉成我們要的 <think> 格式。"""
    parts = []
    if ex.get("system"):
        parts.append(f"<|im_start|>system\n{ex['system']}<|im_end|>\n")
    for turn in ex["conversations"]:
        role = "user" if turn["from"] in ("user", "human") else "assistant"
        parts.append(f"<|im_start|>{role}\n{turn['value']}<|im_end|>\n")
    return {"text": "".join(parts)}

ds = ds.map(format_row, remove_columns=ds.column_names)
```

> 💡 **注意**：`Bespoke-Stratos-17k` 的 assistant 內容裡可能已經是
> ` thinking...` 或 `<think>...</think>` 格式，也可能沒有。
> **轉換後一定要抽樣檢查**，確認思考標籤是 Qwen3.5 要的 `<think>`，
> 而不是 Qwen3 的 ` thinking`。
> 若是後者，請做字串替換：
> ```python
> text = text.replace(" thinking", "<think>").replace(" response", "</think>")
> ```

### 4.3 資料格式與前處理

**核心原則：不要讓資料集的欄位格式綁死你。** 先寫一個轉換腳本，
把所有候選資料集統一成下面這種「單一 `text` 欄位」的格式：

```
<|im_start|>user
{題目}
<|im_end|>
<|im_start|>assistant
<think>
{推理過程}
</think>

{最終答案，結尾用 \boxed{...}}
<|im_end|>
```

這種做法的好處：直接用 tokenizer 對 `text` 做 causal LM 訓練，
不需要處理掩碼（masking）的複雜度，也完全避開 3.5 節提到的模板差異問題。

`scripts/prepare_data.py` 骨架：

```python
"""把各種帶推理痕跡的數學資料集統一成訓練格式。"""
import json
import re
from datasets import load_dataset

# ===== 設定 =====
DATASET_ID = "open-r1/OpenR1-Math-220k"
CONFIG = "default"         # default=93,733 筆；extended=131,396 筆；all=225,129 筆
SPLIT = "train"            # 此資料集只有 train
N_SAMPLES = 5000           # 先用小量跑通，之後再放大
OUT_PATH = "data/processed/sft_5k.jsonl"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning expert. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# ===== 載入 =====
# 實際欄位（已核實）：
#   problem (str), solution (str), answer (str), problem_type, question_type,
#   source, uuid, is_reasoning_complete (list[bool]), generations (list[str]),
#   correctness_math_verify (list[bool]), correctness_llama (list[bool]),
#   finish_reasons (list[str]), correctness_count (int), messages (list[dict])
ds = load_dataset(DATASET_ID, CONFIG, split=SPLIT)
print("原始筆數:", len(ds))
print("欄位:", ds.column_names)

def extract_boxed(text):
    """抓出最後一個 \\boxed{...} 的內容（用括號計數，處理巢狀括號）。"""
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    depth, start = 0, None
    for i in range(idx, len(text)):
        if text[i] == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i]
    return None

def parse_generation(gen):
    """把 <think>...</think> 拆成 (reasoning, final_answer_part)。"""
    if not gen:
        return None, None
    if "</think>" in gen:
        reasoning, final = gen.rsplit("</think>", 1)
        reasoning = reasoning.replace("<think>", "", 1).strip()
        return reasoning, final.strip()
    # 沒有閉合標籤：整段當推理，沒有最終答案段
    return gen.replace("<think>", "", 1).strip(), ""

def pick_best_generation(example):
    """挑一條「通過驗證、且推理完整」的生成結果。

    這是這個資料集最關鍵的一步：generations 是候選清單，
    correctness_math_verify 是逐條的官方驗證結果。
    只挑 verify 通過的，可以大幅提升訓練訊號品質。
    """
    gens = example.get("generations") or []
    ok = example.get("correctness_math_verify") or []
    complete = example.get("is_reasoning_complete") or []

    # 優先：驗證通過 + 推理完整
    for i, g in enumerate(gens):
        if i < len(ok) and ok[i] and (i >= len(complete) or complete[i]):
            return g
    # 次選：只要驗證通過
    for i, g in enumerate(gens):
        if i < len(ok) and ok[i]:
            return g
    # 最後手段：第一條（之後靠 \boxed{} 檢查把關）
    return gens[0] if gens else None

# ===== 轉換 =====
written = skipped = 0
with open(OUT_PATH, "w", encoding="utf-8") as fout:
    for ex in ds:
        if written >= N_SAMPLES:
            break

        problem = ex.get("problem")
        gen = pick_best_generation(ex)
        if not problem or not gen:
            skipped += 1
            continue

        reasoning, final_part = parse_generation(gen)
        if not reasoning:
            skipped += 1
            continue

        # 最終答案：優先取模型生成中的 \boxed{}，其次取資料集的 answer 欄位
        boxed = extract_boxed(gen)
        final = boxed or ex.get("answer")
        if not final:
            skipped += 1
            continue

        # 若推理段裡沒有 \boxed{}，就在推理結尾補一句，確保格式一致
        if "\\boxed" not in reasoning:
            reasoning = reasoning.rstrip() + \
                f"\n\nTherefore, the answer is \\boxed{{{final}}}."

        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{reasoning.strip()}\n</think>\n\n"
            f"The final answer is \\boxed{{{final}}}.<|im_end|>\n"
        )
        fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        written += 1

print(f"已寫出 {written} 筆到 {OUT_PATH}（跳過 {skipped} 筆）")
```

**處理這個資料集時務必注意的四件事：**

1. **`generations` 是清單，不是單一字串。** 每一題有 2–4 條候選推理，
   要搭配 `correctness_math_verify`（逐條的官方驗證結果）挑選。
   **只挑驗證通過的樣本，是提升訓練品質最有效的一步。**
   官方也提供 `correctness_count` 讓你快速看出這題有幾條是對的。
2. **`is_reasoning_complete` 也是清單。** 有些推理是被截斷的（`False`），
   訓練時應優先排除。
3. **`answer` 欄位的格式不統一。** 觀察實際資料可以看到像是
   `v_{R}=4\mathrm{~}/\mathrm{},v_{B}=10...` 這種殘缺的 LaTeX，
   也有 `D`（多選題選項）或 `\frac{1}{4}`。
   所以**優先使用生成內容中的 `\boxed{}`**，`answer` 只當備援。
4. **有 `source` 欄位**（`olympiads`、`aops_forum`、`cn_contest` 等），
   以及 `problem_type`（Algebra、Geometry、Number Theory 等）。
   這讓你**可以按領域分析**，例如「只微調注意力層是否對幾何題特別有效」。

**另外兩件一定要做的事：**

5. **過濾長度。** 思維鏈動輒數千 token（實際觀察可達 3,000+ token）。
   先統計 token 長度分佈，把超過 `max_seq_length` 的樣本截斷或丟棄。
   若你有 50% 的樣本被截斷，模型的思考會被硬生生切掉，訓練效果會很差。
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base")
   lens = [len(tok.encode(json.loads(l)["text"])) for l in open(OUT_PATH, encoding="utf-8")]
   import numpy as np
   print("中位數:", np.median(lens), "p90:", np.percentile(lens, 90), "最大:", max(lens))
   print("超過 2048 的比例:", sum(1 for L in lens if L > 2048) / len(lens))
   ```
6. **去重。** 不同來源常有重複題目，可用 `problem` 正規化後的字串去重。

### 4.4 建議的資料量實驗設計

「不同資料量」是本專案指定的比較軸之一，建議固定其他變數，只動資料量：

| 實驗代號 | 樣本數 | 用途 |
| --- | --- | --- |
| D1 | 1,000 | 觀察「極少量資料能否學會格式」 |
| D2 | 5,000 | 主要基準 |
| D3 | 20,000 | 觀察報酬遞減 |
| D4 | 全部（可能 5 萬+） | 觀察是否飽和或退化 |

> 💡 **重要觀念**：小模型（2B）在少量資料下**很快就能學會「格式」**
> （輸出 `<think>` 標籤並以 `\boxed{}` 收尾），但**「數學正確率」的進步會慢很多**。
> 這兩件事一定要分開報告，否則你會誤判模型真的變聰明了。

### 4.5 驗收標準

- [ ] `data/processed/sft_5k.jsonl` 已產生，且每行都是合法 JSON
- [ ] 抽樣 3 筆人工檢查格式正確（有 `<think>`、有 `\boxed{}`）
- [ ] 已統計 token 長度分佈，並決定 `max_seq_length`

---

## 5. 訓練設定：LoRA / QLoRA 超參數與實驗矩陣

### 5.1 為什麼用參數高效微調（PEFT）

參數高效微調（PEFT）凍結原始權重，只訓練少量額外參數。
以 4B 模型、rank 16 為例，可訓練參數通常不到總參數的 1%，
**優化器狀態與梯度幾乎消失**，VRAM 需求從 FFT 的 56 GB 降到 11 GB 左右。

再加上 **QLoRA** 把凍結的基礎權重也壓到 4-bit，
就能再降到 4.3 GB，這就是 16 GB 顯卡能從容訓練 4B 模型的關鍵。

三者的關係可以這樣理解：

```
FFT        ：訓練「全部」權重                  → 56 GB   ❌
LoRA       ：凍結權重，只訓練小的低秩矩陣        → 11 GB   ⚠️
QLoRA      ：凍結權重（且壓成 4-bit），再掛 LoRA → 4.3 GB  ✅
DoRA       ：同 LoRA，但把權重拆成「大小」與「方向」分別調 → 約 6 GB
```

### 5.2 建議的超參數起點

| 參數 | 建議值 | 說明 |
| --- | --- | --- |
| **`quant_mode`** | **`qlora`（4B）／`bf16`（2B）** | **本專案最重要的設定，見 0.2 節結論三** |
| `max_seq_length` | 2048（起點）→ 4096（QLoRA 時可更大） | 思維鏈很長，這個值直接決定多少推理被保留 |
| `lora_rank (r)` | 16 | 實驗軸之一，見 5.4 |
| `lora_alpha` | 16 | 慣例上設為等於 r |
| `lora_dropout` | 0 | Unsloth 建議 |
| `target_modules` | `"all-linear"` | 涵蓋注意力與 FFN，也含線性注意力層 |
| `per_device_train_batch_size` | 1 | 16 GB 顯卡的現實值 |
| `gradient_accumulation_steps` | 8–16 | 用來湊出等效 batch size 8–16 |
| `learning_rate` | 2e-4（bf16）／1e-4 ~ 2e-4（QLoRA） | QLoRA 有時需要略低的學習率 |
| `lr_scheduler_type` | cosine | |
| `warmup_ratio` | 0.03 | |
| `num_train_epochs` | 1–3 | 先跑 1 epoch 看曲線 |
| `optim` | `adamw_8bit`（OOM 時改 `paged_adamw_8bit`） | 省 VRAM 的關鍵 |
| `gradient_checkpointing` | `"unsloth"` | 大幅省 VRAM，代價是慢一些 |
| `bf16` | True | 不要用 fp16 |
| `bnb_4bit_quant_type` | `"nf4"` | 只在使用 QLoRA 時生效 |
| `bnb_4bit_use_double_quant` | **True** | **務必開啟**，額外省約 0.37 bytes/參數且品質損失小 |
| `bnb_4bit_compute_dtype` | `torch.bfloat16` | 計算時的解量化精度 |

**VRAM 實測參考值：**

| 模型 | bf16 LoRA（官方數據） | QLoRA 4-bit（估算） |
| --- | --- | --- |
| Qwen3.5-0.8B | 約 3 GB | 約 1.5 GB |
| **Qwen3.5-2B** | **約 5 GB** | **約 2.5 GB** |
| **Qwen3.5-4B** | **約 10 GB** | **約 4.3 GB** |
| Qwen3.5-9B | 約 22 GB | 約 8–9 GB |

> 💡 **注意 QLoRA 帶來的「解鎖效果」**：
> 官方數據顯示 9B 的 bf16 LoRA 需要 22 GB，**你的 16 GB 完全做不到**；
> 但 9B 的 QLoRA 只需要約 8–9 GB，**你的顯卡可以跑**。
> 這意味著 QLoRA 不只是「省記憶體」，它**改變了你能選擇的模型規模**。
> 如果你時間充裕，**9B + QLoRA 是一個很有價值的額外實驗組**。

### 5.3 怎麼選方法：決策樹

不要憑感覺選。請照下面的順序決定：

```
你的模型有多大？
│
├─ 2B（2.27B）
│   └─ ★ 用 bf16 LoRA（約 4.6 GB，非常寬鬆）
│       └─ 想加一組幾乎零成本的對照？ → DoRA（VRAM 幾乎不變）
│
└─ 4B（4.66B）
    └─ ★ 也用 bf16 LoRA（約 9.5 GB，16 GB 內還有約 6 GB 餘裕）
        └─ 官方就是在 14.5 GB 的 T4 上跑 16-bit LoRA，這是可行路線
│
└─ 你非用 4-bit 不可嗎？（例如想跑 9B）
    └─ 標準 QLoRA 在 Qwen3.5 上會崩潰（見 5.6 節）
        ├─ 想研究這個 bug → E7 實驗組，重現崩潰並記錄
        ├─ 想真的跑起來 → 套用 5.6.3 的修正（只省約 1.4 GB，不值得）
        └─ 想找真正的 4-bit 路線 → HQQ（未經驗證，屬探索性）
```

**選擇的三個原則：**

1. **不要為了省 VRAM 而量化，因為你不需要。**
   2B 的 bf16 LoRA 只佔 4.6 GB、4B 只佔 9.5 GB，兩者都在 16 GB 內。
   **量化在此硬體上換不到關鍵好處，卻要付出品質與 bug 的風險。**
2. **先建立 bf16 基準線。** 任何量化實驗都必須跟 bf16 對照才有意義。
   bf16 是你的「正確答案版本」。
3. **量化是實驗變因，不是妥協。**
   本專案把 QLoRA 當成**正式的研究對照**（而且要誠實報告它會崩潰），
   而不是「不得已的替代方案」。**這才是課程要的深度。**

> ⚠️ **一個常見的錯誤期待**：很多人以為「4-bit 可以讓我用更大的模型」。
> 在 Qwen3.5 上是**反過來**的——因為 4-bit 會崩潰，
> 你反而**不能**用 QLoRA 去塞 9B。想跑 9B 請看 5.6.6 的 HQQ，
> 或老實用 4B 的 bf16 LoRA。
   這樣你的報告才有課程要的深度。

### 5.4 完整實驗矩陣

以下是建議矩陣。**總共約 20 組，請依實際時間彈性刪減**，
並優先保證兩件事：
（1）「base vs 微調後」的對照完整；
（2）**E 組（方法對照）至少完成 4 組**，因為那是課程主題的核心。
（好消息：主力方法都是 bf16 LoRA，所以 E 組跑起來比原估計更快、更穩。）

**A. 主軸：模型規模（必做）**

| 編號 | 模型 | 方法 | rank | 資料量 | 備註 |
| --- | --- | --- | --- | --- | --- |
| A1 | 2B | 無（base 基準線） | – | – | 評估用，不訓練 |
| A2 | 2B | **bf16 LoRA** | 16 | 5k | 2B 的品質基準 |
| A3 | 4B | 無（base 基準線） | – | – | 評估用，不訓練 |
| A4 | 4B | **bf16 LoRA** | 16 | 5k | **4B 的主力方法**（實測約 9.5 GB，可行） |

**B. 主軸：LoRA rank（在 2B、bf16、5k 上做）**

| 編號 | rank | 目的 |
| --- | --- | --- |
| B1 | 4 | 極低秩，觀察是否不足 |
| B2 | 16 | 基準（= A2） |
| B3 | 64 | 較高秩，觀察是否過擬合 |
| B4 | 128 | 高秩，觀察報酬遞減與 VRAM 壓力 |

> 💡 **進階**：在 4B 上重跑一次 rank 16 vs 64，
> 觀察「較大的模型是否對 rank 更不敏感」。這是很好的原創對照。

**C. 主軸：資料量（在 2B、bf16、rank 16 上做）**

| 編號 | 資料量 |
| --- | --- |
| C1 | 1,000 |
| C2 | 5,000（= A2） |
| C3 | 20,000 |
| C4 | 全部 |

**D. 主軸：微調哪些層（在 2B、bf16、rank 16、5k 上做）**

| 編號 | 微調範圍 | 目的 |
| --- | --- | --- |
| D1 | 只微調注意力層（`--attention-only`） | 測試注意力是否為主因 |
| D2 | 只微調 FFN 層（`--mlp-only`） | 測試 MLP 是否為主因 |
| D3 | 兩者都調（= A2） | 基準 |
| D4 | 只微調較高 rank 的注意力層（rank 64 + `--attention-only`） | 測試「少量但更強」能否勝過「多而弱」 |

> 這就是回應課程核心問題「**一個 LLM 到底需要改變多少才能學會新任務？**」
> 的關鍵實驗組。
>
> 💡 **進階延伸（強烈建議）**：Qwen3.5 是混合架構，
> 每 4 層中有 3 層是線性注意力（Gated DeltaNet）、1 層是完整注意力。
> 你可以進一步設計實驗，**只微調那 1/4 的完整注意力層**，
> 或**只微調線性注意力層**。這是一個既有原創性、
> 又直接呼應這個模型架構特性的實驗，很可能成為報告的亮點。
> 實作方式：用 `FastVisionModel.get_peft_model(..., target_modules=[...])`
> 傳入自訂的模組名稱清單（依層索引篩選）。

**E. 主軸：微調方法與量化（★ 本專案的核心對照軸）**

這一組直接對應課程的「Efficient Fine-Tuning」主題，
也是回答「怎麼在有限 VRAM 下微調」的關鍵。

| 編號 | 方法 | 模型 | 實測 VRAM | 目的 |
| --- | --- | --- | --- | --- |
| **E1** | **bf16 LoRA** | 2B | **約 4.6 GB** | **★ 品質基準線，最先跑這個** |
| **E2** | **bf16 LoRA** | 4B | **約 9.5 GB** | **★ 4B 的主力方法** |
| E3 | DoRA | 2B | ≈ 4.6 GB | 與 LoRA 比品質（成本幾乎為零） |
| E4 | DoRA | 4B | ≈ 9.5 GB | 同上，在 4B 上驗證 |
| E5 | QLoRA（4-bit NF4） | 2B | 約 3.0 GB | **進階**：需先套用 5.6 節修正 |
| E6 | QLoRA（4-bit NF4） | 4B | 約 4.6 GB | **進階**：需先套用 5.6 節修正 |
| E7 | QLoRA **不跳過 linear_attn** | 2B | – | **故意重現 issue #10010 的崩潰**，記錄錯誤訊息 |

> ⚠️ **E7 是一個「刻意的失敗實驗」，但它很有價值。**
> 它會證明「QLoRA 在 Qwen3.5 上會崩潰」這個事實，
> 讓你可以在報告中引用**自己重現的錯誤訊息**，而不是只引用別人的 issue。
> **請把完整的 traceback 貼進報告附錄。**
>
> ⚠️ **E2（4B 的 bf16 LoRA）不是「測試極限」，而是正式方法。**
> 9.5 GB 在 16 GB 內還有約 6 GB 餘裕。
> 若真的 OOM，請依 6.4 節的順序處理（先降序列長度，最後才考慮量化）。

**F. 進階對照（時間允許才做）**

| 編號 | 內容 |
| --- | --- |
| F1 | 指令微調風格（短答案）vs 領域適應（長推理），觀察能力取捨 |
| F2 | 4B 高秩（rank 64）是否勝過 2B 低秩 |
| F3 | 序列長度 2048 vs 4096，對正確率的影響 |
| F4 | 只微調線性注意力層 vs 只微調完整注意力層（見 5.4 D 組的延伸） |
| F5 | HQQ 4-bit + LoRA（真正的 4-bit 訓練替代路線，見 5.6 節） |

**執行順序建議**：

```
第一批（先求跑通）：E1 → A1 → A2 → E2
第二批（核心對照）：A3 → A4 → C1 → C3 → B1 → B3
第三批（擴大範圍）：D1 → D2 → D4 → E3 → E4
第四批（進階／有時間）：E7 → E5 → E6 → B4 → F 系列
```

> 💡 **務必先跑 E1（2B bf16 LoRA）**。它是你的「已知正確」基準線：
> 如果 E1 都跑不出合理結果，那問題在你的資料或流程，不在量化。
> **先建立基準線，再引入變因**，這是除錯的基本原則。

### 5.5 驗收標準

- [ ] 已確定第一批要跑的實驗編號（建議 E1 → A1 → A2 → E2）
- [ ] 已為每組實驗想好 `outputs/` 下的資料夾命名規則
- [ ] **已確認 2B 與 4B 都使用 bf16 LoRA 作為主力**（見 0.2 節結論二）
- [ ] 已讀完 5.6 節的 QLoRA 警訊，並知道如何驗證品質

### 5.6 ⚠️ 重要警訊：QLoRA 對 Qwen3.5 目前是壞的（附解法）

**這是本專案最需要謹慎處理的技術判斷，請完整閱讀。**

#### 結論先講

**不要在 Qwen3.5 上用 QLoRA，除非你願意套用下面 5.6.3 的修正。**
這不是「品質會差一點」的問題，而是**會直接崩潰**的問題。

#### 5.6.1 官方怎麼說

Unsloth 的 Qwen3.5 微調文件明確寫著：

> **不建議對 Qwen3.5 模型做 QLoRA（4-bit）訓練，無論是 MoE 或 dense 版本，
> 因為量化誤差比一般模型大（higher than normal quantization differences）。**

而且他們**自己的 Qwen3.5-4B 官方筆記本用的是 16-bit LoRA**
（`load_in_4bit = False`），不是 QLoRA。

#### 5.6.2 真正的原因（兩個機制）

**機制 A：量化敏感度**

Qwen3.5 的線性注意力層（Gated DeltaNet）與一般注意力層不同。
官方的逐張量 KLD 研究（超過 150 個基準、121 種設定）指出：

> **「有些張量對量化非常敏感」**
> **「`ssm_out` 會劇烈惡化 KLD，而且省下的空間微不足道」**
> **「量化任何 `attn_*` 對混合架構都特別敏感」**

對應到 Qwen3.5 的張量名稱：
- `ssm_out` = Gated DeltaNet 的 `out_proj`
- `attn_q` / `attn_gate` = `q_proj` 與它的 gate 部分（`attn_output_gate: true`）

**機制 B：程式碼 bug（這才是會讓你當場崩潰的原因）**

這是最關鍵的發現：**在 4-bit 路徑下，Qwen3.5 的 Gated DeltaNet
會把「打包好的 4-bit 權重」直接送進 `F.linear`**，導致形狀不符而崩潰：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
              (258x5120 and 1x15728640)
```

前面通常還會有一行警告：

```
[bitsandbytes] FP4 quantization state not initialized.
Please call .cuda() or .to(device) on the LinearFP4 layer first.
```

**為什麼會這樣？** 因為 Unsloth 內部的「不量化清單」用的是 `'mamba'` 這個字串，
而 **Qwen3.5 的模組名稱是 `linear_attn`**。比對邏輯是**前綴／後綴**，
不是子字串，所以 `'mamba'` **比對不到** `linear_attn`。結果就是：
**Gated DeltaNet 的 `in_proj_qkv`、`in_proj_z`、`in_proj_a`、`in_proj_b`、`out_proj`
全部被 4-bit 量化，然後其中一條路徑沒有正確解量化。**

**證據強度**：這個問題有兩個**目前仍開啟**的官方 issue，
而且有人做過隔離測試：

| 測試對象 | 架構 | QLoRA 結果 |
| --- | --- | --- |
| Qwen2.5-3B | 純注意力 | ✅ 正常訓練 |
| gemma-2-27b | 純注意力 | ✅ 正常訓練（16.66 GB 峰值） |
| **Qwen3.5 / Qwen3.8（GatedDeltaNet）** | **混合** | ❌ **第一次 forward 就崩潰** |

**同樣的技術堆疊，只有混合架構的模型失敗。** 這證明這是架構特定的 bug，
不是你的設定問題。

相關 issue：
- https://github.com/unslothai/unsloth/issues/10010
- https://github.com/unslothai/unsloth/issues/9867

#### 5.6.3 如果你真的想用 QLoRA：修正方法

**做法：把 Gated DeltaNet 的線性注意力層排除在量化之外。**

關鍵是 `llm_int8_skip_modules`。這個參數**確實會被 4-bit 路徑採用**
（雖然名稱看起來只屬於 8-bit），比對方式支援前綴、後綴與從頭開始的正規表達式。

```python
import torch
from transformers import BitsAndBytesConfig
from unsloth import FastVisionModel

N_LAYERS = 32   # 2B 用 24；4B 用 32

# ★ 一定要自己加上 "lm_head"
#   因為一旦你提供了自訂清單，Unsloth 的自動保護（lm_head、
#   綁定權重、輸出嵌入）就會全部失效，這會破壞 tie_word_embeddings。
skip_modules = ["lm_head"] + [
    f"model.language_model.layers.{i}.linear_attn"
    for i in range(N_LAYERS)
]

custom_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # ★ 一定要明確指定！
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=skip_modules,
)

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="Qwen/Qwen3.5-2B-Base",
    max_seq_length=2048,
    load_in_4bit=False,               # 關閉自動 4-bit
    quantization_config=custom_bnb,   # 改用自己的設定
    use_exact_model_name=True,        # ★ 關鍵：避免被導向預量化鏡像
)
```

**三個必須知道的細節：**

1. **`use_exact_model_name=True` 是關鍵。**
   Unsloth 在 `load_in_4bit=True` 時會把你導向它的「預量化鏡像模型」，
   而那個鏡像上的 `linear_attn` 量化狀態是壞的。
   若你載入的是預量化模型，**自訂的 skip 清單完全不會生效**
   （已有人實測確認）。用 `use_exact_model_name=True` 強制載入原始檢查點，
   讓 bitsandbytes 在載入時即時量化。

2. **`bnb_4bit_quant_type` 的預設值是 `"fp4"`，不是 `"nf4"`。**
   這和一般人的直覺相反。**訓練一定要明確寫 `"nf4"`**
   （NF4 是 QLoRA 論文提出、針對常態分布權重最佳化的格式）。

3. **`modules_to_not_convert` 在 `BitsAndBytesConfig` 上是無效的。**
   它會被丟進 `**kwargs` 然後**靜默忽略**，連警告都不會有。
   正確的參數名是 **`llm_int8_skip_modules`**。

**代價：記憶體會增加。**

PEFT 的 `prepare_model_for_kbit_training` 會把**所有非 4-bit 的參數升到 fp32**，
所以被你跳過的線性注意力層會變成 4 bytes/參數：

| 模型 | 線性注意力層參數量 | 4-bit | bf16 | **fp32（PEFT 會這樣做）** |
| --- | --- | --- | --- | --- |
| 2B | 約 378M | 0.2 GB | 0.76 GB | **1.5 GB** |
| 4B | 約 1,010M | 0.52 GB | 2.02 GB | **4.04 GB** |

修正後的 QLoRA 大約會多花 **+1.3 GB（2B）／+3.5 GB（4B）**，
實際用量約為 2B：4.3 GB、4B：8.1 GB
——**已經接近 bf16 LoRA 的 4.6／9.5 GB，省下的空間大幅縮水。**

> 💡 **這就是為什麼本專案的結論是「直接用 bf16 LoRA」。**
> 修正後的 QLoRA 既麻煩、又只省一點點，還要承擔品質風險。

#### 5.6.4 為什麼仍保留 QLoRA 作為實驗組

因為**「官方不建議」與「實際掉多少分」是兩件事**，而你正好有工具可以測量：

| 面向 | 事實 |
| --- | --- |
| **技術現實** | 標準 QLoRA 在此架構上會崩潰（bug）；修正後只省約 1.4 GB |
| **品質** | 官方警告 4-bit 對此架構誤差較大；`ssm_out` 與 `attn_*` 是敏感張量 |
| **現實** | 業界絕大多數消費級微調都在用 QLoRA，讀者會想知道這裡行不行 |

**你的 E5/E6/E7 實驗組正好能回答這個問題**，而且 E7（故意重現崩潰）
讓你可以在報告中拿出**自己跑出來的錯誤訊息**，比引用別人的 issue 有力得多。

**判斷標準**（比較 E1 vs E5、E2 vs E6）：

- 修正後 QLoRA 與 bf16 差距在 **±3% 以內** → 值得用，因為省了記憶體
- 掉超過 **8%** → 這是**寶貴的發現**，寫進報告，並用 bf16
- QLoRA **意外更好** → 先懷疑評估流程有問題，重複驗證再下結論

#### 5.6.5 ⚠️ 另一個獨立的陷阱：絕對不要用 fp16 訓練

這與量化無關，但同樣重要：

**Qwen3.5 的 Gated DeltaNet 層在 fp16 下會產生 NaN 梯度。**
這件事嚴重到 Unsloth 直接把 `qwen3_5` 放進它的 `FORCE_FLOAT32` 清單，
原始碼註解寫得很直白：

```
"qwen3_5",  # Qwen3.5 GDN layers produce NaN grad norms in float16 training
```

**所以你必須用 `bf16=True`，絕對不要用 `fp16=True`。**
本手冊的訓練腳本已固定 `bf16=True`，請不要改成 fp16。

> 📌 **還有一個相關的上游 bug（未修）**：
> transformers 的 `Qwen3_5GatedDeltaNet` 在 bf16 下初始化 `A_log` 時，
> 約有 0.4% 的機率抽到 `0.0`，使得 `log(0) = -inf`，
> 該注意力頭會**永久失效且無法恢復**（因為梯度恰好是 0）。
> 在 35B 設定下每次初始化約有 3–6 個頭死亡。
> 修正 PR 已被關閉但**未合併**。
> 對你的小模型影響有限（頭數少、且可重跑），但這是此架構的真實脆弱點，
> **值得在報告中提一句**。
> 參考：https://github.com/huggingface/transformers/issues/47831

#### 5.6.6 真正的 4-bit 替代路線：HQQ

如果你**就是想要 4-bit 訓練**（例如想把 9B 塞進 16 GB），
除了修正版 QLoRA 之外，還有一條路：**HQQ（Half-Quadratic Quantization）**。

| 特性 | HQQ |
| --- | --- |
| PEFT LoRA 訓練 | ✅ 官方支援（PEFT 文件明載） |
| 每層不同位元數 | ✅ `HqqConfig.dynamic_config`（**bitsandbytes 做不到**） |
| 跳過特定模組 | ✅ `HqqConfig.skip_modules`（預設已含 `lm_head`） |
| fp32 升級懲罰 | ✅ **沒有**（PEFT 會跳過 HQQ 模型的 fp32 升級） |
| 混合架構驗證 | ❓ **沒有公開證據**，屬未知領域 |

```powershell
pip install hqq
```

> ⚠️ HQQ 對混合 SSM 架構**沒有任何公開的驗證案例**。
> 若你要試（F5 實驗組），請務必先做煙霧測試，
> 並預期這是「探索性」實驗，而不是可靠的工作流程。

**另外兩個常見選項，為什麼不推薦：**

- **torchao**：透過 transformers 的 `TorchAoConfig` 搭配 LoRA
  **只支援 8-bit**，4-bit 會被 transformers 以 `ValueError` 直接擋下來
  （原始碼層級的硬性限制，不是文件沒寫）。
  它的 QAT 路徑可用，但 **QAT 不會降低訓練時的 VRAM**（權重仍是高精度）。
- **quanto**：沒有 PEFT 整合，且已進入維護模式
  （官方建議改用 bitsandbytes 或 torchao）。

#### 5.6.7 一句話總結

> **Qwen3.5 + 16 GB 的最佳解是 bf16 LoRA（2B 約 4.6 GB、4B 約 9.5 GB）。
> QLoRA 在此架構上會崩潰，修正後也只省約 1.4 GB，不值得作為主力。
> 如果你想要真正的 4-bit 訓練，HQQ 是更有希望但未經驗證的方向。**

**3. 把 bf16 LoRA 當作「答案的正確版本」**

本專案的最終結論應該以 **bf16 LoRA 的成績**為準，
QLoRA 的成績則是「用多少 VRAM 換多少品質」的量化答案。
這樣的報告才站得住腳。

#### 如果 QLoRA 修正後品質還是不行：三個備援方案

| 備援 | 做法 | 效果 |
| --- | --- | --- |
| **★ 直接用 bf16（推薦）** | `--quant bf16` | 2B 約 4.6 GB、4B 約 9.5 GB，**兩者都放得下**。這本來就是本專案的預設，所以「備援」其實是回到主線 |
| **降低序列長度** | `--max-seq-len 1024` | 活化值減半，但思維鏈會被截斷，傷害數學推理 |
| **改用 HQQ** | 見 5.6.6 節 | 真正支援 LoRA 訓練的 4-bit 路線，且無 fp32 升級懲罰；但**混合架構未經驗證** |
| **只做 2B** | 全部實驗集中在 2B | 2B bf16 LoRA 只要 4.6 GB，實驗空間最寬裕，可跑最多組對照 |

> 💡 **最務實的整體建議**：
> **2B 與 4B 都用 bf16 LoRA，不要用 QLoRA。**
> 理由：兩者都放得進 16 GB；QLoRA 在此架構上會崩潰，修正後也只省約 1.4 GB。
>
> **QLoRA 的價值在於「當對照組」，而不是「當主力」。**
> 用它來證明「傳統的省記憶體手段在這個架構上不適用」，
> 這本身就是一個紮實、可引用的研究結論。

---

## 6. 開始訓練

### 6.1 訓練腳本

`scripts/train_lora.py`：

```python
"""以 Unsloth 對 Qwen3.5 Base 模型做數學推理 SFT。

支援三種微調方法，用 --quant 切換：
    bf16   : bf16 全精度 + LoRA      ★ 預設，2B 與 4B 都可行
    dora   : bf16 + DoRA             （LoRA 改良版，幾乎不增加 VRAM）
    qlora  : 4-bit NF4 + LoRA        ⚠️ Qwen3.5 上目前有 bug，見 5.6 節

注意：Unsloth 只提供「4-bit」與「16-bit」兩條路徑，沒有 8-bit。
      若你需要真正的 8-bit LoRA，請自行建立 BitsAndBytesConfig（見 5.6 節）。

本腳本使用 Unsloth 官方的 FastVisionModel 介面，
因為 Qwen3.5 是 VLM，只有這個介面能精確控制「只微調語言層」。
"""
import argparse
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--data", default="data/processed/sft_5k.jsonl")
    ap.add_argument("--out", default="outputs/qwen35-2b-lora-r16")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--attention-only", action="store_true",
                    help="D1：只微調注意力層")
    ap.add_argument("--mlp-only", action="store_true",
                    help="D2：只微調 FFN 層")
    # ★★★ 本專案最重要的參數 ★★★
    ap.add_argument("--quant", default="bf16",
                    choices=["bf16", "dora", "qlora"],
                    help="微調方法：bf16（預設）/ dora / qlora（見 5.6 節警告）")
    ap.add_argument("--optim", default="adamw_8bit",
                    help="優化器：adamw_8bit / paged_adamw_8bit / adafactor")
    ap.add_argument("--dry-run", action="store_true",
                    help="只載入模型並印出 VRAM，不訓練")
    args = ap.parse_args()

    # ============================================================
    # 1. 依 --quant 決定量化設定
    # ============================================================
    if args.quant == "qlora":
        # QLoRA：把凍結的基礎權重壓到 4-bit
        # ⚠️ Unsloth 的內部 skip 清單用的是 'mamba' 這個字串，
        #    而 Qwen3.5 的模組叫 'linear_attn'，兩者對不上（比對是前綴/後綴，
        #    不是子字串）。所以 Unsloth 的 QLoRA「會」把 Gated DeltaNet
        #    的投影層量化掉 —— 這正是 issue #10010 / #9867 的崩潰原因。
        #    若要用 QLoRA，必須套用 5.6 節的自訂 BitsAndBytesConfig。
        load_in_4bit, load_in_16bit = True, False
    else:
        # bf16 / dora：全精度基礎權重（本專案的預設與建議）
        load_in_4bit, load_in_16bit = False, True

    print(f"=== 微調方法：{args.quant} ===")
    print(f"  load_in_4bit  = {load_in_4bit}")
    print(f"  load_in_16bit = {load_in_16bit}")

    if args.quant == "qlora":
        print("\n⚠️  警告：Qwen3.5 的 QLoRA 目前有已知 bug（見 project_plan.md 5.6 節）。")
        print("    若這裡載入後第一次 forward 就出現")
        print("    'mat1 and mat2 shapes cannot be multiplied'，")
        print("    代表你踩到了 issue #10010 / #9867。")
        print("    請改用 --quant bf16，或套用 5.6 節的自訂 quantization_config。")

    # ============================================================
    # 2. 載入模型
    # ============================================================
    print("\n載入模型中…")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=load_in_4bit,
        load_in_16bit=load_in_16bit,
        full_finetuning=False,
    )

    if torch.cuda.is_available():
        print(f"  載入後 VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ============================================================
    # 3. 實驗軸：微調哪些模組
    # ============================================================
    # 這組旗標是 Unsloth 官方為 Qwen3.5 這類 VLM 提供的，
    # 比分模組名稱的正則表達式可靠得多。
    finetune_attn = not args.mlp_only
    finetune_mlp = not args.attention_only

    # ---------- 4. 掛上 LoRA / DoRA adapter ----------
    use_dora = (args.quant == "dora")
    print(f"\n掛上 {'DoRA' if use_dora else 'LoRA'} adapter (r={args.rank})…")

    model = FastVisionModel.get_peft_model(
        model,
        # ★ 關鍵：數學資料沒有圖片，務必關閉視覺層微調
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=finetune_attn,
        finetune_mlp_modules=finetune_mlp,

        r=args.rank,
        lora_alpha=args.rank,          # 慣例：alpha == r
        lora_dropout=0,
        bias="none",
        use_dora=use_dora,             # ★ DoRA 開關
        use_rslora=False,
        random_state=3407,
        target_modules="all-linear",
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_seq_len,
    )

    # 印出實際可訓練參數比例（報告要用）
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n可訓練參數: {trainable:,} / {total:,} = {100*trainable/total:.3f}%")

    # 檢查參數 dtype 分佈 —— 可用來確認量化是否真的生效
    dtypes = {}
    for p in model.parameters():
        dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
    print(f"參數 dtype 分佈: {dtypes}")

    # ---------- 4. 資料 ----------
    ds = load_dataset("json", data_files=args.data, split="train")
    print(f"訓練樣本數: {len(ds)}")

    # ---------- 5. 訓練 ----------
    # Qwen3.5 是 VLM，所以要用 Unsloth 專用的 collator。
    # train_on_responses_only=True 是關鍵：它會把「使用者問題」部分的 loss 遮掉，
    # 只對 assistant 的回答計算 loss。這是 SFT 的正確做法，
    # 可以避免模型浪費容量去背誦使用者的提問。
    from unsloth.trainer import UnslothVisionDataCollator

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(
            model, tokenizer,
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
            force_match=True,
        ),
        args=SFTConfig(
            output_dir=args.out,
            max_seq_length=args.max_seq_len,
            dataset_text_field="text",
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            warmup_ratio=0.03,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            optim=args.optim,
            bf16=True,
            logging_steps=5,
            save_steps=200,
            save_total_limit=2,
            seed=3407,
            report_to="none",
        ),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    trainer.train()

    # ============================================================
    # 5. 記錄 VRAM 峰值（★ 這是本專案的重要實驗數據）
    # ============================================================
    if torch.cuda.is_available():
        import json, os
        peak = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"\n=== VRAM 峰值 ===")
        print(f"  max_memory_allocated : {peak:.2f} GB")
        print(f"  max_memory_reserved  : {reserved:.2f} GB")
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "vram_report.json"), "w") as f:
            json.dump({
                "quant_mode": args.quant,
                "rank": args.rank,
                "max_seq_length": args.max_seq_len,
                "batch": args.batch,
                "accum": args.accum,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_pct": trainable / total,
                "peak_allocated_gb": peak,
                "peak_reserved_gb": reserved,
            }, f, indent=2)

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"\nLoRA adapter 已存到 {args.out}")

if __name__ == "__main__":
    main()
```

> 📌 **務必記下三組數字，它們都是回答研究問題的關鍵證據：**
>
> | 數據 | 回答什麼問題 |
> | --- | --- |
> | **可訓練參數比例** | 「一個 LLM 到底需要改變多少？」 |
> | **VRAM 峰值**（已自動存成 `vram_report.json`） | 「不同方法省下多少記憶體？」 |
> | **準確率** | 「省下來的代價是什麼？」 |
>
> 這三者的關係就是本專案的核心貢獻：**用多少資源，換到多少能力**。
> 請務必在報告中把它們並列成一張表。
>
> ⚠️ **`finetune_vision_layers=False` 是關鍵。** Qwen3.5 是 VLM，
> 若不關閉視覺層，你會浪費大量 VRAM 訓練一個對數學毫無用處的視覺塔，
> 而且可能造成災難性遺忘。
>
> 📌 **`train_on_responses_only=True` 也很重要。** 它讓 loss 只計算
> assistant 的回答部分（遮掉使用者問題）。這是 SFT 的標準做法。
> 若你發現 `instruction_part` / `response_part` 沒有正確匹配
> （訓練 loss 異常低或模型學會複述問題），請先印出一筆
> tokenize 後的樣本檢查，確認這兩個字串確實出現在你的資料中。

### 6.2 先做煙霧測試（Smoke Test）

**永遠先用極小資料量確認整條流程能跑完**，再去跑正式實驗。
這可以省下你好幾個小時的無效等待。

```powershell
# 先產生 50 筆的小資料集
python -c "import json,itertools; lines=open('data/processed/sft_5k.jsonl',encoding='utf-8').readlines()[:50]; open('data/processed/smoke.jsonl','w',encoding='utf-8').writelines(lines)"

# 跑 10 步看看
python scripts/train_lora.py --model Qwen/Qwen3.5-2B-Base `
  --data data/processed/smoke.jsonl --out outputs/smoke `
  --quant bf16 --epochs 1 --max-seq-len 1024
```

若這一步能跑完並存出 adapter，就代表環境沒問題，**可以開始正式實驗了**。

**（選用）用同樣的 50 筆資料重現 QLoRA 的崩潰**

這一步不是為了「跑通」，而是為了**取得第一手證據**。
它應該會在幾秒內失敗，而那個失敗正是你要寫進報告的內容：

```powershell
python scripts/train_lora.py --model Qwen/Qwen3.5-2B-Base `
  --data data/processed/smoke.jsonl --out outputs/smoke_qlora `
  --quant qlora --max-seq-len 512
```

> ⚠️ **預期會出現的錯誤**（見 5.6.2 節）：
> ```
> RuntimeError: mat1 and mat2 shapes cannot be multiplied
>               (258x5120 and 1x15728640)
> ```
> 或先出現 `FP4 quantization state not initialized` 的警告。
>
> **請把完整的 traceback 複製到一個檔案保存起來**（例如
> `results/qlora_crash_traceback.txt`），這是報告附錄的材料。
>
> 💡 **這裡有個重要的心態調整**：
> 一般的教學會說「跑通了才能繼續」。但你的專案裡，
> **「證明某個主流方法在這個架構上不可用」本身就是有價值的結果。**
> 前提是你要有**完整、可重現的證據**——所以請保留原始錯誤訊息，
> 不要只寫「QLoRA 失敗了」。

### 6.3 正式訓練指令範例

```powershell
# ========== E1：2B，bf16 LoRA（★ 品質基準線，最先跑這個）==========
python scripts/train_lora.py --model Qwen/Qwen3.5-2B-Base `
  --data data/processed/sft_5k.jsonl `
  --out outputs/qwen35-2b-bf16-r16 --quant bf16 --rank 16 --max-seq-len 2048

# ========== A4/E2：4B，bf16 LoRA（4B 的主力方法，約 9.5 GB）==========
python scripts/train_lora.py --model Qwen/Qwen3.5-4B-Base `
  --data data/processed/sft_5k.jsonl `
  --out outputs/qwen35-4b-bf16-r16 --quant bf16 --rank 16 --max-seq-len 2048

# ========== E3/E4：DoRA（成本幾乎為零，值得做）==========
python scripts/train_lora.py --model Qwen/Qwen3.5-2B-Base `
  --data data/processed/sft_5k.jsonl `
  --out outputs/qwen35-2b-dora-r16 --quant dora --rank 16 --max-seq-len 2048

# ========== E7：故意重現 QLoRA 的崩潰（進階，用於報告）==========
# 預期會出現 "mat1 and mat2 shapes cannot be multiplied"
python scripts/train_lora.py --model Qwen/Qwen3.5-2B-Base `
  --data data/processed/smoke.jsonl `
  --out outputs/qwen35-2b-qlora-crash --quant qlora --max-seq-len 512

# ========== B3：2B，rank 64 ==========
python scripts/train_lora.py --model Qwen/Qwen3.5-2B-Base `
  --data data/processed/sft_5k.jsonl `
  --out outputs/qwen35-2b-bf16-r64 --quant bf16 --rank 64 --max-seq-len 2048

# ========== D1：只微調注意力層 ==========
python scripts/train_lora.py --model Qwen/Qwen3.5-2B-Base `
  --data data/processed/sft_5k.jsonl `
  --out outputs/qwen35-2b-attn-only --quant bf16 --attention-only

# ========== D2：只微調 FFN 層 ==========
python scripts/train_lora.py --model Qwen/Qwen3.5-2B-Base `
  --data data/processed/sft_5k.jsonl `
  --out outputs/qwen35-2b-mlp-only --quant bf16 --mlp-only
```

> 💡 **4B 的序列長度可以這樣估**：
> 權重 9.5 GB + seq 2048 的活化值約 0.34 GB ≈ 10 GB，還有 6 GB 餘裕。
> 拉到 seq 4096 時活化值約 0.67 GB，仍然安全。
> **不需要為了 4B 的序列長度而改用量化。**

> ⚠️ **關於 QLoRA（E5/E6）：請先讀 5.6 節。**
> 直接跑 `--quant qlora` **會崩潰**，因為 Unsloth 的內部清單
> 用 `'mamba'` 比對不到 Qwen3.5 的 `linear_attn` 模組名。
> 要用 QLoRA 必須先套用 5.6.3 節的自訂 `BitsAndBytesConfig`。
> **建議先用 E7 跑一次崩潰，把錯誤訊息記錄下來**，
> 這對報告很有價值，而且只需幾秒鐘。

### 6.4 訓練時該看什麼

| 指標 | 健康訊號 | 危險訊號 |
| --- | --- | --- |
| `loss` | 穩定下降，最後落在 0.5–1.5 附近 | 完全不動（學習率太低或資料格式錯誤） |
| `grad_norm` | 平穩 | 劇烈震盪或爆衝到很大 |
| 產出樣本 | 有 `<think>`、有 `\boxed{}` | 沒有標籤、重複語句、亂碼 |
| VRAM | 穩定 | 逐步上升（記憶體洩漏）或 OOM |

**OOM（Out of Memory）的處理順序：**

**先做「零成本」的三步（不會犧牲模型品質，但對 LoRA 效果有限）：**

1. 確認 `per_device_train_batch_size` 是 1
2. 提高 `gradient_accumulation_steps` 補回等效 batch size
3. 改用 `--optim paged_adamw_8bit`（把優化器狀態分頁到你的 64 GB RAM）

> ⚠️ **注意**：對 LoRA 而言優化器只佔 0.04–0.17 GB，
> **所以第 3 步幾乎救不了什麼**。真正佔記憶體的是凍結的基礎權重（2 bytes/參數）
> 與活化值。別在優化器上花太多時間。

**若還是 OOM，依序犧牲（由代價小到大）：**

4. **降低 `max_seq_length`**（2048 → 1536 → 1024）
   —— 這是最有效的一步，因為活化值與序列長度成正比
5. 縮小 rank（16 → 8）
6. 改用只微調注意力層（`--attention-only`），減少可訓練模組與其活化值
7. 最後才考慮：**修正版 QLoRA**（見 5.6.3 節）

> ⚠️ **不要直接跑 `--quant qlora` 來解決 OOM。**
> 標準 QLoRA 在 Qwen3.5 上會**崩潰**（見 5.6 節），
> 你不會得到「省記憶體」，只會得到一個 `RuntimeError`。
> 若真的需要量化，必須先套用 5.6.3 節的自訂設定，
> 而那只省約 1.4 GB。
>
> 💡 **實際上你大概率不會 OOM。** 4B 的 bf16 LoRA 約 9.5 GB，
> 在 16 GB 內還有約 6 GB 餘裕。

### 6.5 驗收標準

- [ ] 煙霧測試成功跑完（bf16 路徑）
- [ ] **至少完成 E1（2B + bf16 LoRA + rank 16）作為品質基準線**
- [ ] 至少完成 A4（4B + bf16 LoRA），確認 4B 在 16 GB 上可行
- [ ] （進階）用 E7 重現 QLoRA 崩潰，並保存完整錯誤訊息
- [ ] `vram_report.json` 已產生，且記錄了每組的 VRAM 峰值
- [ ] 訓練曲線有存下來（截圖或 log 檔）
- [ ] 用訓練後的模型生成一段輸出，確認格式正確

---

## 7. 評估：使用 LM Studio（CUDA runtime）

> 🎯 **這一章是整個專案的靈魂。** 課程問的是「改多少才夠」，
> 而沒有可信的評估，前面所有訓練都只是感覺。

### 7.0 ⚠️ 評估的順序，以及一個必須承認的限制

**評估排在後面。** 因為評估後端是 LM Studio，而 LM Studio 只吃 GGUF，
所以你必須**先完成第 8–10 章**（合併 → 轉 GGUF → 量化），
才有東西可以評估：

```
訓練（第 6 章）→ 合併（第 8 章）→ GGUF F16（第 9 章）→ Q8/Q4（第 10 章）
                                                              ↓
                                                LM Studio 評估（本章）
```

**你已經選擇接受這個限制，這是正確的決定**，因為它換來一個更有價值的東西：
**你可以同時比較 F16 / Q8_0 / Q4_K_M，把「微調的效果」與「量化的代價」分開。**

| 你要回答的問題 | 比較哪兩者 |
| --- | --- |
| 微調有沒有用？ | base 模型 F16 **vs** 微調後模型 F16（同一種量化，公平） |
| 量化掉多少分？ | F16 **vs** Q8_0 **vs** Q4_K_M（同一個模型，不同量化） |
| 整體部署代價？ | base F16 **vs** 微調後 Q4_K_M（你實際上會部署的版本） |

> **報告中務必寫清楚**：「本專案以 GGUF 格式評估，故所報數字皆為量化後表現；
> F16 為最接近訓練權重的版本，Q8/Q4 的差異即為量化代價。」
> 這一句話會讓你的方法論站得住腳，而不是被人質疑「為什麼不用 bf16 評估」。

### 7.1 最重要的觀念：基礎模型不能直接跟微調模型比

`Qwen3.5-2B-Base` 是**預訓練模型（pre-trained only）**，不是指令微調模型。
它沒有被教過「要遵守對話格式」。

如果你直接問它一道數學題，它可能：
- 繼續接龍而不是回答
- 不產生 `<think>` 標籤
- 不輸出 `\boxed{}`

**這不代表它數學比較差，只是它沒被教過格式。** 若你直接比較，
你會得到「微調後大幅進步」的假象——但其實你只教會了它格式。

**正確做法：為基礎模型加上公平的起跑點。** 至少要做以下三種基準線：

| 基準線 | 說明 | 目的 |
| --- | --- | --- |
| **B0：chat template + thinking 開啟** | 用官方的聊天模板，`enable_thinking=True` | 誠實呈現 base 模型「被允許思考」時的原始狀態 |
| **B1：chat template + thinking 關閉** | `enable_thinking=False`（模板會直接關閉思考區塊） | 隔離出「思考」這件事本身的貢獻 |
| **B2：few-shot / completion 風格** | 在提示中給 4–8 個訓練集格式的完整範例，不用聊天模板 | **最公平的對照**：證明微調不是只贏在格式 |

**主要結論請以「微調後模型」對比「B0/B1/B2 中最好的那一個」為準。**
若微調模型只贏過最差的基準線，那個進步就沒有說服力。

**必須保持完全一致的變因**（base 與微調模型都要相同）：

- 提示文字（建議使用 Qwen 官方推薦的
  `"Please reason step by step, and put your final answer within \boxed{}."`）
- 取樣參數（本手冊固定 `temperature=0`，即 greedy 解碼）
- **`--max-tokens`**（LM Studio 端的生成上限）
- **`--context`**（載入時的 context length）
- **`--mode` 與 `--limit`**
- 評分器（同一個 `math-verify` 設定）

> ⚠️ **這些參數只要有一個不同，兩份結果就不能放進同一張表。**
> 本手冊的評估腳本會把這些參數寫進結果 JSON 的 `summary`，
> 方便你日後核對。

**另外一定要同時報告「格式調整後的正確率」**，也就是
「有產生可解析 `\boxed{}` 的比例」。因為對 base 模型而言，
微調帶來的進步有很大一部分可能只是格式合規，而不是數學能力。

> 📌 **重要事實**：Qwen **沒有**公布 Qwen3.5-2B-Base / 4B-Base 的數學成績，
> 所以沒有官方的 base 基準線可以引用。你在模型卡上看到的數字
> （例如 4B 的 HMMT 74.0）都是**後訓練版本**的成績，不能當作 base 的基準。

### 7.2 評估指標（三層，缺一不可）

**第一層：正確率（Accuracy）**

最終答案是否正確。使用 `math-verify`（`huggingface/Math-Verify`）做數學等價性判定
（不是字串比對，所以 `1/2` 與 `0.5` 會被視為相同）。

> ⚠️ **Windows 使用者必讀：`math-verify` 有一個會「靜默失敗」的嚴重問題。**
>
> `math-verify` 在 Windows 上使用 `multiprocessing` 來實作逾時機制，
> 這會導致 `parse()` 與 `verify()` 拋出
> `AttributeError: Can't get local object ... run_func`。
> 更糟的是，預設的 `raise_on_error=False` 會**把這個錯誤吞掉**，
> 讓 `parse()` 回傳空清單、`verify()` 回傳 `False`。
> 結果就是：**你的模型明明答對了，卻被計為答錯，而且完全沒有任何錯誤訊息。**
> （對應 issue：https://github.com/huggingface/Math-Verify/issues/79）
>
> **解法：呼叫時明確傳入 `None` 關閉內建逾時。**
> ```python
> gold = parse(gold_str,     parsing_timeout=None)
> pred = parse(gen_text,     parsing_timeout=None)
> ok   = verify(gold, pred,  timeout_seconds=None)
> ```
> 關閉逾時後，請自行在迴圈外層用 `try/except` 與整體時間上限保護，
> 避免某一道題的 LaTeX 讓程式卡死。
>
> 另外建議先跑一次「除錯模式」：把 `raise_on_error=True` 打開，
> 用 20 筆已知答案的樣本確認判分邏輯正確，再開始大規模評估。

**正確的 API 用法（官方文件明載：順序很重要）**

```python
from math_verify import parse, verify
from math_verify.metric import math_metric          # 另一種高階用法

gold = parse(gold_answer_str)     # 標準答案放第一個參數
pred = parse(model_output_str)    # 模型輸出放第二個參數
is_correct = verify(gold, pred)   # verify(gold, prediction)
```

`verify` **刻意設計成非對稱**（不是可交換的），順序寫反會得到錯誤的結果。

**抽取 `\boxed{}` 的建議設定**

`math-verify` 內建支援 `\boxed{}` 抽取，但預設的優先序可能會被
「the final answer is 42」這類句型搶先。評估模型輸出時建議改成：

```python
from math_verify import LatexExtractionConfig, ExprExtractionConfig
from math_verify import parse as mv_parse

pred = mv_parse(
    gen_text,
    extraction_config=[
        LatexExtractionConfig(boxed_match_priority=0),  # 讓 \boxed{} 優先
        ExprExtractionConfig(),
    ],
    parsing_timeout=None,
)
```

> **其他已知陷阱**（都會造成誤判，請一併注意）：
> - 抽取邏輯偏好**最後一個**符合的數字。若模型輸出
>   `"20+20=40. My favorite number is 50."`，會被抽成 `50`。
>   **所以一定要在提示中要求模型用 `\boxed{}`。**
> - 純文字 `x = 1` 抽不出東西，必須包在 LaTeX 環境中
>   （`$x=1$`、`\[x=1\]`、`\boxed{x=1}`）。
> - 用 `**粗體**` 包住答案會破壞解析。
> - 帶分數（如 `11 5/6`）可能解析錯誤並產生偽陽性。
> - **多選題（MCQ）**請改用
>   `StringExtractionConfig(strings=("A","B","C","D"))`，
>   且**不要**與其他抽取設定混用（會誤抓內文中的 A/B/C/D）。

**第二層：格式合規率（Format Compliance）**

模型是否遵守我們要的輸出格式。這是小模型微調最容易進步、也最容易被誤認成
「變聰明」的地方。指標：

- 有產生 `<think>` 的比例
- 有**閉合** `</think>` 的比例（很重要！很多模型會忘記收尾）
- 有產生 `\boxed{}` 的比例
- 答案抽取成功率
- **被 `--max-tokens` 截斷的比例**（見下方警示）

> ⚠️ **「截斷」是評估推理模型時最大的隱藏混淆因子。**
> Qwen 官方建議數學難題要給到 **81,920** 個 token 的思考預算，
> 這在 16 GB 顯卡上根本不可能。若你的 `--max-tokens` 設得太小，
> 基礎模型會因為「還沒想完就被切斷」而拿不到分，
> 你就會誤以為微調帶來了巨大進步。
>
> **做法：對 base 與微調模型使用完全相同的 `--max-tokens`，
> 並且一定要把「截斷率」跟準確率一起報告。**
>
> 實用建議：設 4096（2B）到 8192（4B），
> 並在報告中註明「截斷率 X%」。若截斷率超過 20%，
> 這個基準的數字就不可信，應該提高預算或改用較短的基準。
>
> 💡 **注意 `--max-tokens` 與 `--context` 的關係**：
> `context` 必須至少等於「prompt 長度 + max-tokens」，
> 否則 LM Studio 會因為放不下而報錯，或提前截斷。
> 本手冊預設 `context=8192`、`max-tokens=4096`，是安全的組合。

**第三層：效率指標**

- 平均思考 token 數（思考變長不代表變好，常常只是變囉唆）
- 平均生成時間

**建議的最終報告表格：**

| 模型 | 訓練方法 | GGUF 量化 | VRAM 峰值(訓練) | 可訓練% | GSM8K acc | MATH-500 acc | 截斷率 | `<think>` 合規率 | `\boxed{}` 合規率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2B-Base | 未微調 | F16 | – | – | | | | | |
| **2B** | **bf16 LoRA** | **F16** | | | | | | | |
| 2B | bf16 LoRA | Q8_0 | | | | | | | |
| 2B | bf16 LoRA | Q4_K_M | | | | | | | |
| 2B | DoRA | F16 | | | | | | | |
| 2B | QLoRA（修正版） | F16 | | | | | | | |
| **4B-Base** | 未微調 | F16 | – | – | | | | | |
| **4B** | **bf16 LoRA** | **F16** | | | | | | | |
| 4B | bf16 LoRA | Q4_K_M | | | | | | | |

> 💡 **這張表就是你的論文核心。** 把「訓練方法」「量化格式」「VRAM」「可訓練%」
> 全部放在準確率旁邊，讀者一眼就能看出**兩層代價**：
> 訓練時的方法代價（VRAM）、部署時的量化代價（準確率）。
>
> **建議畫兩張圖：**
> 1. **訓練效率**：橫軸 VRAM 峰值、縱軸準確率（比較訓練方法）
> 2. **量化代價**：橫軸 GGUF 檔案大小、縱軸準確率（比較 F16/Q8/Q4）
>
> 第二張圖直接對應課程的量化主題，而且**只有你能產生**——
> 因為你有自己微調出來的模型。
>
> ⚠️ **注意「同一列才能互相比較」**：
> 要判斷「微調有沒有效」，請比較**同一種量化格式**下的 base vs 微調
> （例如 2B-Base F16 vs 2B bf16 LoRA F16）。
> 拿 base 的 F16 去比微調後的 Q4，會把量化損失誤算成微調的效果。

### 7.3 建議的評估基準

| 基準 | HF ID | 題數 | 標準誤（p=0.5 時） | 為什麼用它 |
| --- | --- | --- | --- | --- |
| GSM8K（test） | `openai/gsm8k`（config `main`） | 1,319 | ±2.7% | 小學數學，靈敏度高；汙染嚴重，只當地板值 |
| **MATH-500** | `HuggingFaceH4/MATH-500` | 500 | ±4.4% | 最划算的單一指標，學術可比性高 |
| MinervaMath | `math-ai/minervamath` | 272 | ±5.9% | 較不易飽和，品質不錯的補充 |
| AIME 2024 | `Maxwell-Jia/AIME_2024` | 30 | **±18%** | 極難，只能當趨勢觀察 |
| AIME 2025 | `math-ai/aime25` | 30 | **±18%** | 同上 |
| BeyondAIME | `ByteDance-Seed/BeyondAIME` | 100 | ±9.8% | 較新、設計上更難且更乾淨 |

> 💡 **建議策略**：以 **MATH-500（500 題）** 為主要指標，
> **MinervaMath（272 題）** 為次要指標。先用前 200 題快速迭代，
> 確認流程無誤後再跑完整集合。
>
> **AIME 只有 30 題，標準誤高達 ±18%，單獨用它無法支撐任何結論。**
> 2B 模型很可能全部答錯——這不是 bug，而是重要的基線資訊
> （「此規模模型無法處理競賽級題目」）。若要用 AIME，
> 請務必標註 `n=30, ±18%`。

> ⚠️ **命名陷阱**：`lm-evaluation-harness` 裡的任務叫 `minerva_math`，
> 那是「MATH 資料集的 Minerva 4-shot 格式」，**不是**這裡說的
> MinervaMath（`math-ai/minervamath`，272 題）。兩者完全不同，別搞混。
> 同樣地，harness 中 MATH-500 的任務名是 `hendrycks_math500`，
> 沒有 `math_500` 這個任務。

### 7.4 ⚠️ 資料汙染（Contamination）問題

**這件事你一定要知道，否則你的結論可能是假的。**

許多數學 SFT 資料集（特別是 `OpenR1-Math-220k`、`NuminaMath`）
本身就是**從 MATH 和 GSM8K 建構出來的**。這代表：

> 你用來訓練的題目，很可能**就包含在你要評估的測試集裡**。

這樣「微調後 GSM8K 從 20% 進步到 60%」可能只是**背題**，不是學會推理。

**至少要做的事：**

1. **做去汙染（decontamination）**：用 n-gram 或正規化後的題目字串，
   比對訓練集與測試集，把重疊的訓練樣本剔除。
2. **使用有去汙染設計的基準**：例如 `DeepMath-103K` 這類資料集
   有針對常見 benchmark 做過去汙染。
3. **報告時誠實揭露**：若無法完全去汙染，就在報告中明確註明這個限制。
4. **最強的證據是「同資料同題」的比較**：即 base（few-shot）與微調模型
   在**完全相同的題目與提示**下比較。這個比較是無條件有效的，
   因為汙染對兩者影響相同。

簡易去汙染腳本骨架：

```python
import json, re
from datasets import load_dataset

def normalize(s):
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s.strip()

# 建立訓練題目的指紋集合
train_fps = set()
with open("data/processed/sft_5k.jsonl", encoding="utf-8") as f:
    for line in f:
        text = json.loads(line)["text"]
        # 取 user 區塊當題目指紋
        m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.S)
        if m:
            train_fps.add(normalize(m.group(1)))

# 比對測試集
test = load_dataset("openai/gsm8k", "main", split="test")
overlap = sum(1 for ex in test if normalize(ex["question"]) in train_fps)
print(f"GSM8K 測試集與訓練集重疊: {overlap}/{len(test)} ({overlap/len(test):.1%})")
```

### 7.5 評估環境設定：LM Studio + CUDA

> 本節是本專案**唯一使用 LM Studio 的地方**。
> 訓練仍然在第 2 章建立的 venv 中進行。
> 設定只做一次，之後每次評估都直接跳到 7.6。

#### 7.5.1 安裝與確認 GPU 後端

1. 到 https://lmstudio.ai/ 下載 **Windows x64** 版本並安裝。
2. 啟動 LM Studio，打開右側的 **Runtime（執行階段）** 面板。
3. **確認你使用的是 CUDA 版的 llama.cpp runtime。**
   LM Studio 會依你的 GPU 提供多種 runtime；
   請選 **CUDA llama.cpp**，而**不要**選 CPU 或 Vulkan。

**如何確認 CUDA 真的生效（三處都要看）：**

| 檢查點 | 應該看到什麼 |
| --- | --- |
| Runtime 面板 | 選中的項目是 CUDA 版本，且標示你的 GPU 名稱 |
| 載入模型後的資訊列 | 顯示 GPU offload（例如 `GPU offload: 28/28 layers`） |
| 推論時的工作管理員 | **GPU（而非 CPU）的使用率明顯上升** |

> ⚠️ **若載入後 GPU 使用率為 0，代表你其實在用 CPU 跑。**
> 這會讓評估慢上 10 倍以上，而且會讓你誤判「模型很慢」。
> 請回到 Runtime 面板改成 CUDA 版並重新載入模型。

#### 7.5.2 安裝 LM Studio 的 Python SDK

評估腳本透過 LM Studio 的 Python SDK 與它溝通。
**這仍然安裝在你原本的 venv 裡**（SDK 只是客戶端，模型跑在 LM Studio 中）：

```powershell
# 在已啟用的 venv 中
pip install lmstudio
```

同時把這個套件補進環境清單：

```powershell
pip freeze > requirements.txt
```

> 💡 **SDK 與 LM Studio 的關係**：腳本負責「送題目、收答案、算分」，
> LM Studio 負責「用 CUDA 跑模型」。**兩者必須同時開著**——
> 關掉 LM Studio 的 GUI 之後，腳本會連不上。

#### 7.5.3 把 GGUF 匯入 LM Studio

LM Studio 只載入它自己模型目錄下的檔案，目錄結構必須是
`發佈者/模型名稱/檔案.gguf`。預設位置是 `~\.lmstudio\models\`。

**最簡單的做法：用 `lms import`。**

```powershell
# 先找到 lms CLI（LM Studio 安裝時會一併裝好）
Get-Command lms

# 匯入你的每一個 GGUF
lms import D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf
lms import D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q8_0.gguf
lms import D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q4_K_M.gguf
```

**或者手動放到正確的目錄結構**（適合一次匯入很多檔案）：

```powershell
$dest = "$env:USERPROFILE\.lmstudio\models\local\qwen35-2b-math"
New-Item -ItemType Directory -Force -Path $dest
Copy-Item D:\DSAI\DSAI4207\gguf\qwen35-2b-math-*.gguf $dest
```

**匯入後，你需要在 LM Studio 中確認模型名稱（model key）。**
本章腳本用 **`--model-key` 參數**指定，所以請先列出可用模型：

```powershell
lms ls
```

把輸出的識別字（通常長得像 `local/qwen35-2b-math-f16` 或檔案路徑形式）
記下來，之後傳給腳本。

> 💡 **`--model-key` 也可以直接給 `.gguf` 檔案的完整路徑**，
> 這樣可以省去匯入步驟，最不容易出錯。
> 兩種都試一次，看哪個在你機器上可用。

#### 7.5.4 基準線模型也要匯入

別忘了**未微調的 base 模型**也要匯入，否則你無法做最重要的對照。
從第 3.6 節下載的 base 權重先用第 9 章轉成 GGUF，或直接下載
社群已發布的版本（見 10.5 節），例如：

```powershell
lms import D:\DSAI\DSAI4207\gguf\qwen35-2b-base-f16.gguf
```

#### 7.5.5 驗收標準

- [ ] LM Studio 的 Runtime 面板選中 **CUDA** 版 llama.cpp
- [ ] 載入任一模型後，**GPU offload 顯示為全部層數**
- [ ] 推論時工作管理員確認 **GPU 使用率上升**
- [ ] `pip install lmstudio` 完成，且 `import lmstudio` 沒有錯誤
- [ ] `lms ls` 能看到你的模型
- [ ] **base 模型與微調後模型都已匯入**

---

### 7.6 評估腳本（透過 LM Studio）

`scripts/evaluate_lmstudio.py`：

```python
"""透過 LM Studio（CUDA runtime）評估數學推理能力。

流程：LM Studio 負責用 CUDA 跑模型，本腳本負責送題、收答案、算分。
使用前請確認：
  1. LM Studio 已開啟，且 runtime 是 CUDA 版
  2. 目標模型已匯入（lms import 或放到 ~/.lmstudio/models）
"""
import argparse, json, time
import lmstudio as lms
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

# ⚠️ Windows 必看：一定要傳 parsing_timeout=None / timeout_seconds=None
# 否則 math-verify 會靜默失敗，把所有答案判為錯（見 7.2 節）。
_PARSE_CFG = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]

def mv_parse(text):
    return parse(text, extraction_config=_PARSE_CFG, parsing_timeout=None)

def mv_verify(gold, pred):
    return verify(gold, pred, timeout_seconds=None)

def extract_boxed(text):
    """抓出最後一個 \\boxed{...}。用括號計數，因為巢狀括號會讓 regex 失效。"""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    depth, start = 0, None
    for i in range(idx, len(text)):
        if text[i] == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i]
    return None

def split_answer(gen):
    """最終答案在【最後一個】</think> 之後。注意是 <think> 不是  thinking。"""
    if "</think>" in gen:
        return gen.rsplit("</think>", 1)[1], True
    return gen, False

# ===== 提示模板（三種基準線，見 7.1 節）=====
SYS = ("You are a mathematical reasoning expert. "
       "Please reason step by step, and put your final answer within \\boxed{}.")

FEWSHOT = (
    "What is 12 * 7? Please reason step by step, and put your final answer "
    "within \\boxed{}.\n\n"
    "<think>\n12 * 7 = 84.\n</think>\n\n"
    "The final answer is \\boxed{84}."
)

def build_chat(mode, question):
    """mode: think | nothink | fewshot"""
    if mode == "fewshot":
        # 用 few-shot 條件時，把示範放在 system 訊息裡
        system = SYS + "\n\nExample:\n" + FEWSHOT
        return lms.Chat(system), question
    if mode == "nothink":
        # 要求直接作答，不要思考
        return lms.Chat(SYS + "\nAnswer directly without thinking."), question
    return lms.Chat(SYS), question

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True,
                    help="LM Studio 的模型識別字，或 .gguf 檔案的完整路徑")
    ap.add_argument("--tag", required=True,
                    help="這份結果的標籤，例如 2b-bf16-f16（會寫進結果檔）")
    ap.add_argument("--bench", default="HuggingFaceH4/MATH-500")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--mode", default="think",
                    choices=["think", "nothink", "fewshot"])
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--context", type=int, default=8192,
                    help="載入時的 context length")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or f"results/eval_{args.tag}_{args.mode}.json"

    # ===== 載入模型（★ 關鍵：gpu ratio 1.0 代表全部放 GPU）=====
    print(f"載入模型：{args.model_key}")
    model = lms.llm(args.model_key, config={
        "contextLength": args.context,
        "gpu": {"ratio": 1.0},      # ★ 必須是 1.0，否則會部分跑在 CPU
    })

    # 顯示實際載入資訊，方便確認 CUDA 生效
    try:
        info = model.get_model_info()
        print(f"  模型：{info.display_name}")
        print(f"  最大 context：{info.max_context_length}")
    except Exception as e:
        print(f"  （無法取得模型資訊：{e}）")

    # ===== 資料 =====
    from datasets import load_dataset
    ds = load_dataset(args.bench, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"評測題數：{len(ds)}（{args.bench} / {args.split}）")

    correct = total = 0
    think_ok = closed_ok = boxed_ok = truncated = 0
    think_word_counts, gen_tokens = [], []
    rows = []

    for i, ex in enumerate(ds, 1):
        question = ex.get("problem") or ex.get("question")
        gold = ex.get("answer") or ex.get("solution")

        chat, user_msg = build_chat(args.mode, question)
        chat.add_user_message(user_msg)

        # ★ temperature=0 → greedy，確保可重現
        t0 = time.time()
        try:
            result = model.respond(chat, config={
                "temperature": 0.0,
                "maxTokens": args.max_tokens,
            })
            gen = str(result)
            stop_reason = result.stats.stop_reason
            n_tokens = result.stats.predicted_tokens_count
        except Exception as e:
            print(f"  [{i}] 推論失敗：{e}")
            rows.append({"idx": i, "question": question, "gold": gold,
                         "error": str(e)})
            total += 1
            continue
        elapsed = time.time() - t0

        answer_part, has_close = split_answer(gen)
        boxed = extract_boxed(answer_part)

        # 判斷是否被長度上限截斷
        hit_cap = (n_tokens >= args.max_tokens
                   and "endOfText" not in str(stop_reason))

        total += 1
        think_ok += int(args.mode != "nothink")
        closed_ok += int(has_close)
        boxed_ok += int(boxed is not None)
        truncated += int(hit_cap)
        gen_tokens.append(n_tokens)
        if has_close:
            think_span = gen.rsplit("</think>", 1)[0]
            think_word_counts.append(len(think_span.split()))

        is_correct = False
        if boxed:
            try:
                is_correct = mv_verify(mv_parse(str(gold)), mv_parse(boxed))
            except Exception as e:
                print(f"  [{i}] verify 失敗：{e} | gold={gold} | pred={boxed}")
                is_correct = str(boxed).strip() == str(gold).strip()
        correct += int(is_correct)

        rows.append({
            "idx": i, "question": question, "gold": gold, "pred_boxed": boxed,
            "correct": is_correct, "has_close_think": has_close,
            "truncated": hit_cap, "gen_tokens": n_tokens,
            "elapsed_sec": round(elapsed, 2), "generation": gen,
        })

        if i % 10 == 0:
            acc = correct / total
            print(f"  [{i}/{len(ds)}] acc={acc:.1%} "
                  f"boxed={boxed_ok/total:.1%} trunc={truncated/total:.1%}")

    summary = {
        "tag": args.tag,
        "model_key": args.model_key,
        "benchmark": args.bench,
        "mode": args.mode,
        "n": total,
        "context": args.context,
        "max_tokens": args.max_tokens,
        "accuracy": correct / max(total, 1),
        "think_rate": think_ok / max(total, 1),
        "closed_think_rate": closed_ok / max(total, 1),
        "boxed_rate": boxed_ok / max(total, 1),
        "truncation_rate": truncated / max(total, 1),
        "avg_gen_tokens": (sum(gen_tokens) / len(gen_tokens)) if gen_tokens else 0,
        "avg_think_words": ((sum(think_word_counts) / len(think_word_counts))
                            if think_word_counts else 0),
    }

    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n已寫入 {out_path}")

if __name__ == "__main__":
    main()
```

> ⚠️ **幾個需要你實測確認的地方**（SDK 的細節可能隨版本變動）：
> - `result.stats.stop_reason` 的字串值（不同版本可能不同），
>   這只影響「截斷率」的判定，不影響準確率。
> - `model.get_model_info()` 的欄位名稱。
> - `fewshot` 模式放在 system 訊息是否被模型正確接受
>   （若不行，改成把示範串在使用者訊息前面）。
>
> **先用 `--limit 5` 跑一次，人工檢查 `rows` 裡的 `generation` 欄位**，
> 確認格式與計分都正確，再放大到 200 題。
>
> 📌 **`"gpu": {"ratio": 1.0}` 是 CUDA 生效的關鍵。**
> 若設成 0.5，模型會有一半跑在 CPU 上，速度會慢好幾倍。
>
> ⚠️ **`avg_think_words` 是「詞數」不是「token 數」**（用空白切分）。
> 若要精確 token 數，需另外用 LM Studio 的 tokenization API；
> 報告中請註明你用的是哪一種。
### 7.7 跑評估

> ⚠️ **執行前檢查清單（每次都要確認）：**
> 1. **LM Studio 已開啟**，runtime 是 **CUDA** 版
> 2. 模型已匯入（`lms ls` 看得到）
> 3. 前一批模型評估完後，**先 unload** 再載入下一個（避免 VRAM 不足）

**第一步：先跑 5 題的煙霧測試，人工檢查輸出格式。**

```powershell
python scripts/evaluate_lmstudio.py `
  --model-key "D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf" `
  --tag 2b-math-f16 --mode think --limit 5
```

打開產生的 JSON，人工確認 `generation` 欄位裡**真的有 `<think>` 推理過程**
而且結尾有 `\boxed{}`。**格式不對就不要往下跑 200 題。**

**第二步：跑完整的量化對照（本專案的核心實驗）。**

```powershell
# ===== 微調後模型：同一組題目，三種量化各跑一次 =====
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf"    --tag 2b-math-f16    --mode think --limit 200
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q8_0.gguf"   --tag 2b-math-q8     --mode think --limit 200
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q4_K_M.gguf" --tag 2b-math-q4km   --mode think --limit 200

# ===== base 模型對照（同一種量化才能公平比較）=====
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-2b-base-f16.gguf"    --tag 2b-base-f16    --mode think   --limit 200
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-2b-base-f16.gguf"    --tag 2b-base-nothink --mode nothink --limit 200
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-2b-base-f16.gguf"    --tag 2b-base-fewshot --mode fewshot --limit 200

# ===== 4B 模型（F16 與 Q4_K_M）=====
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-4b-math-f16.gguf"    --tag 4b-math-f16    --mode think --limit 200
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-4b-math-Q4_K_M.gguf" --tag 4b-math-q4km   --mode think --limit 200
```

> ⚠️ **一致性檢查：**
> 所有要互相比較的模型，**`--mode`、`--limit`、`--max-tokens`、
> `--context` 必須完全相同**。改動任何一個，那份結果就不能放進同一張表。
> 本手冊的檔名設計（`--tag` + `--mode`）已經把參數寫進檔名，方便你核對。

**第三步：跑配對比較。**

```powershell
# 微調 vs base（同一量化）→ 回答「微調有沒有用」
python scripts/compare_paired.py --a results/eval_2b-base-f16_think.json `
                                  --b results/eval_2b-math-f16_think.json

# F16 vs Q8 → 回答「Q8 的代價」
python scripts/compare_paired.py --a results/eval_2b-math-f16_think.json `
                                  --b results/eval_2b-math-q8_think.json

# F16 vs Q4_K_M → 回答「Q4 的代價」
python scripts/compare_paired.py --a results/eval_2b-math-f16_think.json `
                                  --b results/eval_2b-math-q4km_think.json
```

`scripts/compare_paired.py`（做 McNemar 檢定）：

```python
"""配對比較兩個模型在同一批題目上的表現。"""
import argparse, json
from scipy.stats import binomtest

def load_rows(path):
    data = json.load(open(path, encoding="utf-8"))
    return {r["question"]: r["correct"] for r in data["rows"] if "correct" in r}

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)   # 基準
ap.add_argument("--b", required=True)   # 對照
args = ap.parse_args()

a, b = load_rows(args.a), load_rows(args.b)
common = sorted(set(a) & set(b))
print(f"共同題數: {len(common)}")
if not common:
    raise SystemExit("兩份結果沒有共同題目，無法配對比較")

acc_a = sum(a[q] for q in common) / len(common)
acc_b = sum(b[q] for q in common) / len(common)

# McNemar：只看「兩者不一致」的題目
b_only = sum(1 for q in common if b[q] and not a[q])   # b 對 a 錯
a_only = sum(1 for q in common if a[q] and not b[q])   # a 對 b 錯
n = b_only + a_only

print(f"  基準準確率: {acc_a:.1%}")
print(f"  對照準確率: {acc_b:.1%}")
print(f"  差異      : {acc_b - acc_a:+.1%}")
print(f"  僅基準對  : {a_only}")
print(f"  僅對照對  : {b_only}")

if n > 0:
    p = binomtest(b_only, n, 0.5).pvalue
    print(f"  McNemar p = {p:.4f}  ->  "
          f"{'顯著差異' if p < 0.05 else '無顯著差異'}")
else:
    print("  兩者完全一致，無差異")
```

> 💡 若 `scipy` 未安裝：`pip install scipy`。
>
> **這一步的輸出就是你的核心結論**：
> - 「微調 vs base」的差異 → 微調到底有沒有效
> - 「F16 vs Q4」的差異 → 量化到底掉多少分
>
> 兩者都是**配對檢定**，所以即使只有 200 題，也能偵測到 3–5% 的真實差異。

### 7.8 統計顯著性：不要被雜訊騙了

**單一基準的標準誤：**

```
SE = sqrt(p(1-p)/n)
```

| 題數 | p=0.5 時的 SE | 95% 信賴區間 |
| --- | --- | --- |
| 30（AIME） | 9.1% | ±18% |
| 100 | 5.0% | ±9.8% |
| 272（MinervaMath） | 3.0% | ±5.9% |
| 500（MATH-500） | 2.2% | ±4.4% |
| 1,319（GSM8K） | 1.4% | ±2.7% |

也就是說，在 500 題上 **±4% 以內的差異可能只是隨機誤差**。
200 題的 SE 約 3.5%，更不穩定。

**✅ 更強的做法：用「配對檢定」（paired test）**

這是本節最重要的一點。base 模型與微調模型是在**完全相同的題目**上評估的，
這叫「配對資料」。配對比較遠比「比較兩個獨立的信賴區間」靈敏得多：

- **不要**用「base 是 28% ± 4%，微調是 32% ± 4%，區間重疊所以不顯著」這種推論。
- **要**用 **McNemar 檢定**，或對「每題是否答對」的差異做 **bootstrap**，
  算出「差異的信賴區間」。

在配對設計下，n=500 時 3% 的差異就可能是顯著的，
而用獨立信賴區間看則完全看不出來。

**報告建議格式：**

```
MATH-500 (n=500)
  base (few-shot)      : 28.4%  [95% CI 24.5–32.5]
  2B + LoRA r16        : 33.2%  [95% CI 29.1–37.5]
  配對差異             : +4.8%  [bootstrap 95% CI +1.2 ~ +8.4]  ← 顯著
```

**關於 pass@1 與 maj@k：**

- **pass@1（平均正確率）**：固定溫度下取 N 個樣本，取平均。這是標準做法。
- **maj@k（self-consistency）**：取 k 個樣本投票。**兩個都建議報告**，
  因為 pass@1 會掩蓋「小模型的推理不穩定」，而 maj@k 會掩蓋
  「它其實只是偶爾猜對」。
- 在 16 GB 上，k=4–8 是合理的設定。不要過度投入 pass@k——
  近期研究（arXiv:2510.04265）指出 pass@k 在小樣本數下排名不穩定。

### 7.9 若評估太慢：可以考慮的加速方式

200 題 × 長思考鏈，在 16 GB 卡上可能要跑 1–3 小時。若真的太慢，
**優先調整 LM Studio 本身的設定**，而不是換框架：

| 做法 | 效果 | 風險 |
| --- | --- | --- |
| **降低 `--max-tokens`**（4096 → 2048） | 最直接，可能快將近一倍 | 思考被截斷，記得看截斷率 |
| 降低 `--limit`（200 → 100） | 線性加速 | 統計檢定力下降 |
| **確認 GPU offload 是 100%** | 若目前只跑部分 GPU，可能有數倍加速 | 需足夠 VRAM（F16 的 4B 約 8.4 GB，放得下） |
| 開啟 LM Studio 的 flash attention | 通常快 10–20% | 少數模型可能出錯，需驗證輸出正確 |
| 縮短 `--context` | 減少 KV cache | context 太小會直接報錯，不要低於 prompt + max-tokens |

> ⚠️ **不建議切換到其他評估框架。**
> 你的專案要求評估後端一致，而且**換框架會引入新的變數**
> （不同的 tokenizer 行為、不同的推論參數預設值），
> 讓你的數字無法與其他結果比較。
> **寧可少跑幾題，也不要混用框架。**

### 7.10 驗收標準

- [ ] **LM Studio 的 runtime 確認為 CUDA，且 GPU offload 為全部層數**
- [ ] base 的三條基準線（think / nothink / fewshot）都已完成
- [ ] 微調模型的 F16、Q8_0、Q4_K_M **三個版本都已完成評估**
- [ ] 三個層次的指標（正確率／格式／效率）都有數據
- [ ] **已記錄截斷率**，且所有比較對象使用相同的 `--max-tokens`
- [ ] 每組實驗都記錄了「可訓練參數比例」
- [ ] 已完成汙染檢查並在報告中註明
- [ ] 每個數字都有標註題數與信賴區間
- [ ] 已保留逐題的對錯紀錄（供配對檢定使用）

---

## 8. 合併權重與輸出

LoRA adapter 只是一個小檔案，部署前通常需要「合併」回基礎模型，
成為一個獨立的完整模型。GGUF 轉換也必須用合併後的權重。

### 8.1 合併腳本

`scripts/merge_adapter.py`：

```python
"""把 LoRA adapter 合併回基礎模型，輸出 bf16 safetensors。"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基礎模型路徑或 HF ID")
    ap.add_argument("--adapter", required=True, help="LoRA adapter 資料夾")
    ap.add_argument("--out", required=True, help="輸出資料夾")
    args = ap.parse_args()

    print("載入基礎模型（CPU，避免 VRAM 不足）…")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print("載入並合併 adapter…")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    print(f"儲存到 {args.out} …")
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    print("完成。")

if __name__ == "__main__":
    main()
```

執行：

```powershell
python scripts/merge_adapter.py `
  --base Qwen/Qwen3.5-2B-Base `
  --adapter outputs/qwen35-2b-lora-r16 `
  --out outputs/merged/qwen35-2b-math-r16
```

### 8.2 檢查合併結果

```powershell
# 檔案大小應該跟原始 base 模型差不多
Get-ChildItem outputs\merged\qwen35-2b-math-r16 | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
```

**驗收標準**：合併後的 safetensors 總大小應與 base 模型相近
（2B 約 4.6 GB，4B 約 9.4 GB）。若只有幾百 MB，代表你只存到了 adapter。

### 8.3 常見問題

| 問題 | 原因 | 解法 |
| --- | --- | --- |
| 合併後模型回答品質變差 | 聊天模板／EOS token 不一致 | 確認推論時使用**訓練時相同的模板** |
| 合併時 CPU 記憶體不足 | 用了 float32 載入 | 保持 `torch_dtype=torch.bfloat16` |
| 只有 adapter 大小 | `merge_and_unload()` 沒被呼叫 | 檢查腳本 |

---

## 9. 轉成 GGUF

### 9.1 為什麼要轉 GGUF

GGUF 是 llama.cpp 生態系（llama.cpp、Ollama、LM Studio 等）使用的格式，
讓你的模型可以在各種裝置上以 CPU 或 GPU 高效推論。這也是「量化」的前提。

### 9.2 ⚠️ 最重要的一節：MTP 層與 `--no-mtp`

這是本專案**最容易卡住**的一步。請完整讀完本節，它會幫你省下好幾個小時。

#### 症狀

社群上與 Qwen3.5 GGUF 相關的錯誤訊息幾乎都是這個樣子：

```
missing tensor 'blk.32.attn_norm.weight'
```
或
```
check_tensor_dims: tensor 'blk.32.attn_norm.weight' not found
```

看到「32 層的模型卻要求 `blk.32`（第 33 塊）」，
很多人的第一反應是「這是 off-by-one 的 bug」。**但這個判斷是錯的。**

#### 真正的原因

回到第 3.4 節：Qwen3.5 的 Base 檢查點**真的包含完整的 MTP 層權重**。

我實際檢查 `model.safetensors.index.json` 後確認，`Qwen3.5-2B-Base` 內含：

```
mtp.fc.weight
mtp.norm.weight
mtp.pre_fc_norm_embedding.weight
mtp.pre_fc_norm_hidden.weight
mtp.layers.0.input_layernorm.weight
mtp.layers.0.post_attention_layernorm.weight
mtp.layers.0.mlp.{gate,up,down}_proj.weight
mtp.layers.0.self_attn.{q,k,v,o}_proj.weight
mtp.layers.0.self_attn.{q,k}_norm.weight
```

搭配 `config.json` 的 `mtp_num_hidden_layers: 1`，結論是：

> **2B-Base 實際上有 24 + 1 = 25 塊，4B-Base 有 32 + 1 = 33 塊。
> 「33 塊」是正確的設計，不是 bug。**

llama.cpp 的轉換腳本預設會把 MTP 層**一起打包進去**，
於是載入時就會去找不存在的 `blk.32`（因為 MTP 區塊的張量命名/內容與主幹不同，
在部分版本上無法對應），產生上面那些錯誤。

#### 解法：轉換時加上 `--no-mtp`

**對你的數學微調而言，MTP 層完全沒有用**（它只是用來加速推論的投機解碼頭）。
所以最乾淨的做法就是**不要打包它**：

```powershell
python convert_hf_to_gguf.py <merged_model_dir> `
  --outfile gguf\model-f16.gguf --outtype f16 --no-mtp
```

`--no-mtp`（別名 `--no-nextn`）會把 NextN/MTP 張量排除，得到 24 / 32 塊的 GGUF，
也就是載入器期待的形狀。這是最重要的一個參數，**請務必加上**。

#### 社群 issue 的實際狀態（避免你白等）

很多人會去找「修好了沒」，但這些 issue 的狀態容易誤導：

| Issue | 狀態 | 說明 |
| --- | --- | --- |
| [#24737](https://github.com/ggml-org/llama.cpp/issues/24737)（33 vs 32 塊） | **仍開啟**（stale） | 因為根本不是 off-by-one bug，而是 MTP 設計 |
| [#24211](https://github.com/ggml-org/llama.cpp/issues/24211) | **被機器人自動關閉為 not_planned** | 不是被修好，是過期自動關閉 |
| [#23033](https://github.com/ggml-org/llama.cpp/issues/23033)（缺 `blk.40.ssm_conv1d`） | **仍開啟**（stale） | 這是**不同的** bug：MTP 區塊被誤判為 recurrent 層 |
| [#24661](https://github.com/ggml-org/llama.cpp/issues/24661)（量化時 `Bad layer N`） | **已修復並關閉** | 舊版的 MTP GGUF 無法量化，已修 |

所以不要看到 issue 還開著就以為不能用——**關鍵是搞清楚你遇到的是哪一類，
以及要用哪個參數。**

#### 真正已合併的兩個關鍵修補

1. **`recurrent_layers` 的 MTP 補位**（commit `9a757058`，2026-09-07，PR #28208）
   轉換腳本現在會明確寫出 `recurrent_layers` 陣列，並且
   **對 MTP 區塊補上 `false`**。這正是防止「MTP 區塊被誤判為 recurrent 層、
   進而被要求提供不存在的 `ssm_conv1d` 張量」的關鍵。
2. **量化器的層編號修正**（#24986）
   修掉了 `llama-quantize` 在含 MTP 的 GGUF 上
   以 `Bad layer N ... Must be in [0, N)` 中止的問題。

**因此：請使用 2026-09-07 之後的 llama.cpp。** 實務上就是
**最新版的 master 分支**（撰寫當下的最新 release 是 `b10906`，2026-09-11）。
**不要**去釘選舊版本，因為上述修補都是 2026 年 4–9 月才進去的。

#### 其他你應該知道的事

- **不要依賴 PR #27132**：它至今仍是 **draft（草稿）未合併**狀態。
  它想修的行為其實已經由其他 commit 進入 master 了。
- **不要用社群的 fork 修補**：有人在 #24737 貼了修補 commit
  （CryptoJones 的 `57e8718`），但那是**被上游拒絕**的做法。
- **Windows 的 release 壓縮檔裡沒有 `convert_hf_to_gguf.py`。**
  轉換腳本只存在於原始碼倉庫。所以你會需要**兩者**：
  （a）`git clone` 下來的原始碼（取得轉換腳本），
  （b）編譯後（或下載的）`llama-quantize.exe`。

### 9.3 路線 A（推薦）：使用 Unsloth 內建的 GGUF 匯出

這是最省事、也最不容易踩雷的路線。官方文件明確支援 Qwen3.5 匯出 GGUF，
且已處理好架構細節。

在訓練腳本結尾直接匯出（或寫一個獨立腳本載入後匯出）：

```python
"""直接從已訓練的模型匯出 GGUF。"""
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/qwen35-2b-lora-r16",   # 訓練輸出的資料夾
    max_seq_length=2048,
    load_in_4bit=False,
    load_in_16bit=True,
)

# 先匯出 F16 中繼檔（後續再自行量化成 Q8/Q4）
model.save_pretrained_gguf("gguf/qwen35-2b-math", tokenizer,
                           quantization_method="f16")

# 也可以直接一次匯出你要的量化版本
model.save_pretrained_gguf("gguf/qwen35-2b-math", tokenizer,
                           quantization_method="q8_0")
model.save_pretrained_gguf("gguf/qwen35-2b-math", tokenizer,
                           quantization_method="q4_k_m")
```

> 💡 第一次執行時，Unsloth 會先 clone 並編譯 llama.cpp，需要幾分鐘到下載數 GB，
> 請保持網路暢通。

### 9.4 路線 B（備援）：手動用 llama.cpp 轉換

若路線 A 失敗，或你想理解底層流程（也才能使用 imatrix 提升量化品質），
可用官方工具。**前提是使用 2026-09-07 之後的 llama.cpp。**

```powershell
# 1. 取得 llama.cpp 原始碼
cd D:\DSAI\DSAI4207
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# 2. 安裝轉換腳本所需的 Python 套件
pip install -r requirements/requirements-convert_hf_to_gguf.txt

# 3. 若架構辨識失敗，安裝最新 transformers
pip install "transformers>=5.0.0" --upgrade

# 4. 轉成 F16 GGUF —— ★ --no-mtp 是關鍵，絕對不能漏 ★
python convert_hf_to_gguf.py D:\DSAI\DSAI4207\outputs\merged\qwen35-2b-math-r16 `
  --outtype f16 `
  --outfile D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf `
  --no-mtp
```

> 💡 若你在轉換時遇到 lazy transpose 相關的錯誤，可加上 `--no-lazy`。
> 代價是會**大幅增加 RAM 使用量**（你的 64 GB 應該還撐得住）。

**轉換完成後，先別急著量化，務必先驗證能不能載入：**

```powershell
cd D:\DSAI\DSAI4207\llama.cpp

# 這一步是「花 5 秒省 5 小時」的檢查
.\build\bin\Release\llama-cli.exe `
  -m D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf `
  -p "What is 2+2?" -n 32
```

- **能正常生成文字** → 恭喜，繼續往下做量化。
- **報 `missing tensor 'blk.32...'`** → 你的 `--no-mtp` 沒生效，
  或 llama.cpp 版本太舊。請更新後重轉。
- **報缺 `ssm_conv1d`** → 這是 issue #23033 那類問題，
  代表你的 llama.cpp 早於 2026-09-07 的 `recurrent_layers` 修補。

### 9.5 關於視覺編碼器與 mmproj

由於本專案是純文字數學微調，你**不需要** `mmproj`（視覺投影）檔案。

- **不要下載 mmproj**：官方發布的 mmproj 檔案每個約 672 MB，
  對純文字模型是純粹的浪費。
- **轉換時不要加 `--mmproj`**：不加的話，轉換腳本只會匯出文字模型
  （視覺權重會被丟棄），這正是你要的結果。
- **不要訓練視覺層**：數學資料沒有圖片，訓練視覺層只會浪費資源
  並可能造成災難性遺忘。

> 補充：視覺塔本身有 24 個 transformer block，
> 所以你在合併與轉換時仍需在 RAM 中載入這些權重，
> 這會增加轉換時間，但不會影響結果。

### 9.6 驗收標準

- [ ] `gguf/qwen35-2b-math-f16.gguf` 已產生，且檔案大小合理（約 4–5 GB）
- [ ] **已用 `llama-cli` 成功載入並生成文字**（最重要的驗收項）
- [ ] 確認轉換時有加上 `--no-mtp`

---

## 10. 量化成 Q8_0 與 Q4_K_M

### 10.1 量化的原理（為什麼可以變小又不爛掉）

神經網路的權重是浮點數（bf16 每個數字 2 bytes）。量化就是**用更少的位元來
表示這些數字**：

| 格式 | 每權重位元 | 相對大小 | 品質 | 用途 |
| --- | --- | --- | --- | --- |
| F16 | 16 | 100% | 基準 | 中繼檔，不直接使用 |
| **Q8_0** | 約 8.5 | 約 53% | 幾乎無損 | 品質優先、VRAM 夠 |
| **Q4_K_M** | 約 4.8 | 約 30% | 輕微下降 | 速度與大小平衡，最常用 |
| Q4_K_S | 約 4.6 | 約 29% | 比 Q4_K_M 略差 | 更極端壓縮 |
| Q3_K_M | 約 3.9 | 約 25% | 明顯下降 | 不建議用於數學推理 |
| Q2_K | 約 2.6 | 約 17% | 嚴重劣化 | 不建議 |

**Q4_K_M 的「K」代表 k-quant（先進的分塊量化），「M」代表 medium 混合精度**
（重要的層用較高精度，不重要的層用較低精度）。這是目前公認性價比最好的選擇。

> ⚠️ **對數學推理的特別提醒**：數學任務對量化誤差比一般聊天更敏感。
> 建議你的實驗中，**Q4 的準確率下降幅度本身就是一個值得報告的結果**
> ——這正好呼應課程「量化如何影響能力」的主題。

**混合架構的敏感層級（重要）**

Qwen3.5 的官方量化基準研究（在 35B-A3B 上做的 KLD 實驗，
對你的 dense 2B/4B 只能算方向性參考）指出：

- **`attn_*` 相關張量對量化特別敏感**——這對混合架構尤其明顯。
  把它們留在較高精度，效果會好很多。
- **`ssm_out`（線性注意力層的輸出投影）最好不要量化**——
  壓到 q2_k 會讓 KLD 大幅惡化，卻幾乎省不到空間。
- **MXFP4 比 Q4_K 差**（在 `attn_gate`、`attn_q`、`ssm_beta`、`ssm_alpha` 上）。
  所以 Q4 等級請**優先選 Q4_K_M，不要選 MXFP4**。
- **imatrix（重要性矩陣）有明顯幫助**，官方發布的 GGUF 都有使用。
  **強烈建議 Q4_K_M 一定要搭配 imatrix。**

這也解釋了為什麼 Unsloth 的「UD（Dynamic）」量化版本表現較好：
它們會把重要層自動升到 8-bit 或 16-bit，
所以 `UD-Q4_K_XL` 跟純 `Q4_K_M` 的張量組成並不相同。

### 10.2 執行量化

```powershell
cd D:\DSAI\DSAI4207\llama.cpp

# 確認 llama-quantize.exe 存在（若無，需先編譯，見 10.3）
Get-ChildItem build\bin\Release\llama-quantize.exe

# ★ 先做 dry-run，確認張量都能處理，避免跑很久才失敗
.\build\bin\Release\llama-quantize.exe --dry-run `
  D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf Q4_K_M

# 量化成 Q8_0
.\build\bin\Release\llama-quantize.exe `
  D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf `
  D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q8_0.gguf `
  Q8_0

# 量化成 Q4_K_M
.\build\bin\Release\llama-quantize.exe `
  D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf `
  D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q4_K_M.gguf `
  Q4_K_M
```

若你使用路線 A（Unsloth 匯出），可以跳過這一步，因為
`quantization_method="q8_0"` 與 `"q4_k_m"` 已經幫你做好了。

### 10.2.1 進階：用 imatrix 提升 Q4 品質（建議做）

imatrix 會先統計「哪些權重對模型輸出比較重要」，
再讓量化器對重要權重保留較高精度。對數學推理這種敏感任務特別有價值。

```powershell
# 1. 準備一小段校準文字（用你的訓練資料切片即可，約 200-500 筆）
#    從 data/processed/sft_5k.jsonl 取出前 300 筆的文字即可

# 2. 產生 imatrix
.\build\bin\Release\llama-imatrix.exe `
  -m D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf `
  -f D:\DSAI\DSAI4207\data\processed\calib.txt `
  -o D:\DSAI\DSAI4207\gguf\qwen35-2b-math.imatrix `
  --chunks 200 -ngl 99

# 3. 用量化時帶入 imatrix
.\build\bin\Release\llama-quantize.exe `
  --imatrix D:\DSAI\DSAI4207\gguf\qwen35-2b-math.imatrix `
  D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf `
  D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q4_K_M-im.gguf `
  Q4_K_M
```

> 💡 **把 `attn_*` 留在高精度**（進階技巧）：
> 如果你發現 Q4 掉分嚴重，可以用 `--tensor-type` 指定個別張量的精度，例如
> `--tensor-type attn_q=Q8_0 --tensor-type attn_k=Q8_0`，
> 或寫成一個 `--tensor-type-file`。
> 這會增加檔案大小，但可能救回關鍵的推理能力。
> **這本身就是一個很好的實驗組**：比較「純 Q4」與
> 「attn 層保持 Q8 的 Q4」，看準確率差多少。

### 10.3 若需要自行編譯 llama.cpp

Windows 上使用 CUDA 的編譯指令（需先安裝 CUDA Toolkit 與 CMake）：

```powershell
cd D:\DSAI\DSAI4207\llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8
```

若只想量化（不需要 GPU 推論加速），可以省掉 `-DGGML_CUDA=ON`：
量化本身是 CPU 工作。

### 10.4 驗證量化結果

```powershell
Get-ChildItem D:\DSAI\DSAI4207\gguf\*.gguf | Select-Object Name, @{n='GB';e={[math]::Round($_.Length/1GB,2)}}
```

**預期大小：**

| 檔案 | 2B 模型（估計） | 4B 模型（實測參考值） |
| --- | --- | --- |
| F16 | 約 4.5–5 GB | 約 8.4 GB（全部量化檔合計） |
| Q8_0 | 約 2.4–2.5 GB | **4.48 GB** |
| Q4_K_M | 約 1.4–1.5 GB | **2.74 GB** |

> 4B 的數字來自實際量測的公開 GGUF（`unsloth/Qwen3.5-4B-GGUF`），
> 是可信的參考值。2B 的數字是依比例推算。

若 Q4_K_M 只有幾百 MB，表示量化過程出錯或檔案被截斷。

**更重要的是：量化後一定要能載入。**

```powershell
.\build\bin\Release\llama-cli.exe `
  -m D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q4_K_M.gguf `
  -p "What is 17 * 23?" -n 64 -ngl 99
```

### 10.5 參考：已發布的現成 GGUF（可用來驗證你的流程）

如果你想確認「我的量化流程對不對」，可以下載社群已發布的 GGUF 做對照。
這些是**未微調的原始模型**，不能當成你的成果，但可以當作流程的對照組：

| 儲存庫 | 說明 |
| --- | --- |
| `unsloth/Qwen3.5-2B-GGUF` | 2B 的完整量化系列（含 Q8_0、Q4_K_M、UD 系列） |
| `unsloth/Qwen3.5-4B-GGUF` | 4B 同上 |
| `unsloth/Qwen3.5-2B-MTP-GGUF` | 保留 MTP 層的版本（與你的 `--no-mtp` 相反） |
| `bartowski/Qwen_Qwen3.5-2B-GGUF` | 另一個常見來源 |
| `lmstudio-community/Qwen3.5-2B-GGUF` | LM Studio 官方社群版 |

> 📌 **`Qwen` 官方並沒有發布任何 GGUF 檔案**，只發布 FP8 與 GPTQ-Int4。
> 網路上所有 Qwen3.5 GGUF 都是社群製作的。

**建議的驗證方法**：把現成的 Q4_K_M 下載下來用 llama.cpp 跑一次，
確認你的環境能正確執行 Qwen3.5 GGUF。這樣當你的自製模型出問題時，
你就能確定問題在「你的轉換」而不是「你的環境」。

> ⚠️ **關於 Ollama**：目前證據不一致。Unsloth 的文件仍寫著
> 「目前沒有任何 Qwen3.5 GGUF 能在 Ollama 上運作（因為分離的 mmproj 視覺檔案）」，
> 但 Ollama 的純文字 qwen35 相關 issue 已被標記為完成。
> **保守建議：先以 llama.cpp 與 LM Studio 為主，把 Ollama 視為未驗證。**

### 10.6 驗收標準

- [ ] Q8_0 與 Q4_K_M 兩個檔案都已產生
- [ ] 檔案大小符合上表預期
- [ ] **兩個量化檔都能用 `llama-cli` 成功載入並生成**
- [ ] 已記錄每個檔案的精確大小（報告要用）

---

## 11. 量化後驗證與推論

### 11.1 用 llama.cpp 測試

```powershell
cd D:\DSAI\DSAI4207\llama.cpp

.\build\bin\Release\llama-cli.exe `
  -m D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q4_K_M.gguf `
  -p "Solve: What is 17 * 23? Think step by step." `
  -n 512 --temp 0.7 -ngl 99
```

參數說明：
- `-ngl 99`：把所有層放到 GPU（99 = 「全部」）
- `-n 512`：最多生成 512 個 token
- `--temp 0.7`：取樣溫度

**你應該看到模型輸出 `<think>` 開頭的推理過程，最後以 `\boxed{}` 收尾。**
若沒有，可能是聊天模板沒套用（加上 `--jinja` 試試）。

### 11.2 在 Ollama / LM Studio 使用（可選）

**Ollama**：建立 `Modelfile`：

```
FROM ./gguf/qwen35-2b-math-Q4_K_M.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
<think>
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
```

```powershell
ollama create qwen35-math-2b -f Modelfile
ollama run qwen35-math-2b
```

**LM Studio**：直接把 `.gguf` 檔拖進模型資料夾即可。

> ⚠️ **Ollama 請視為「未驗證」**：Unsloth 文件目前仍寫著
> 「沒有任何 Qwen3.5 GGUF 能在 Ollama 上運作」，而 Ollama 的相關 issue
> 狀態不一致。**建議優先用 llama.cpp 或 LM Studio**，
> 把 Ollama 當作「有時間再試」的額外項目，不要讓它卡住你的進度。

### 11.3 ⚠️ 量化對準確率的影響（本專案的核心實驗之一）

> 📌 **這個實驗已經在第 7 章完成了。**
> 本節只是把結果整理成報告要用的表格，並說明如何解讀。
> 若你還沒跑，請回到 **7.7 節**。

**為什麼這個實驗特別有價值：** 因為評估後端是 LM Studio（只能載 GGUF），
你**本來就會**得到 F16 / Q8_0 / Q4_K_M 三個版本的分數。
換句話說，這個「量化代價」的實驗是**評估流程的自然產物**，不需要額外工作。

**你要產出的表格：**

| 版本 | 檔案大小 | MATH-500 準確率 | 相對 F16 的損失 | 與 base 的配對 p 值 |
| --- | --- | --- | --- | --- |
| base F16 | 約 4.6 GB | | – | – |
| **微調 F16** | 約 4.6 GB | | **基準** | （微調是否有效） |
| 微調 Q8_0 | 約 2.4 GB | | 應接近 0 | |
| 微調 Q4_K_M | 約 1.4 GB | | 通常 1–4% | |

> 💡 **三個要注意的解讀陷阱：**
>
> 1. **「微調有沒有用」要看同一種量化。**
>    用 base-F16 對比微調-Q4_K_M，會把量化損失混進微調效果裡。
>    正確做法：**base-F16 vs 微調-F16**。
> 2. **「量化掉多少」要在同一個模型內比。**
>    正確做法：**微調-F16 vs 微調-Q4_K_M**。
> 3. **小差距要用配對檢定。** 200 題時 3% 的差異可能是雜訊，
>    請用 7.7 節的 `compare_paired.py`（McNemar），不要只看兩條長條圖。

> ⚠️ 若 Q4_K_M 掉了超過 5%，可能是這個混合架構對量化特別敏感
> （回想 5.6.2 節提到的 `ssm_out` 與 `attn_*` 敏感張量）。
> 這是一個值得寫進報告的發現，而且可以回頭用 10.2.1 節的
> `--tensor-type` 技巧，把敏感層留在較高精度再測一次。

### 11.4 驗收標準

- [ ] 能用 llama.cpp 載入並生成合理的數學推理（環境健康檢查）
- [ ] **F16 / Q8_0 / Q4_K_M 三個版本都已完成 LM Studio 評估**
- [ ] 已完成量化前後的品質對照表，並使用配對檢定
- [ ] 已明確區分「微調效果」與「量化代價」兩個不同的比較

---

## 12. 時程建議與交付物清單

### 12.1 四週時程（你選擇「一個月以上」，此為最小可行版本）

**第 1 週：環境與基礎**
- 完成 venv、PyTorch CUDA 驗證、Unsloth 安裝
- 下載 2B 與 4B 模型
- 選定資料集並完成前處理，產出 `sft_5k.jsonl`
- 跑通煙霧測試（**bf16 路徑**）

**第 2 週：基準線與主力訓練**
- 完成 E1（2B bf16 LoRA）建立品質基準線
- 完成 A4/E2（4B bf16 LoRA），確認 16 GB 可行
- 完成 A1、A3（兩個 base 基準線）
- 建立評估流程，跑出第一組數字
- 完成汙染檢查
- （可選）用 E7 重現並記錄 QLoRA 崩潰

**第 3 週：擴充實驗**
- 完成 DoRA（E3、E4）
- 完成 rank 掃描（B1、B3）與資料量掃描（C1、C3）
- 完成層選擇實驗（D1、D2、D4）
- 整理 VRAM × 準確率的對照表

**第 4 週：量化與報告**
- 合併權重、轉 GGUF（記得 `--no-mtp`）
- 量化成 Q8_0 與 Q4_K_M
- 量化後評估
- （進階）嘗試修正版 QLoRA 或 HQQ
- 撰寫報告與繪圖

### 12.2 交付物清單

- [ ] LoRA adapter 檔案（至少 6 組實驗，**以 bf16 LoRA 與 DoRA 為主**）
- [ ] 每個實驗的 `vram_report.json`（VRAM 峰值與可訓練參數比例）
- [ ] **「方法 × VRAM × 準確率」對照總表**（本專案的核心產出）
- [ ] **QLoRA 崩潰的完整錯誤訊息與重現步驟**（有引用價值）
- [ ] 每個實驗的訓練 log 與 loss 曲線圖
- [ ] 評估結果 CSV（所有模型 × 所有 benchmark）
- [ ] 合併後的 safetensors 權重
- [ ] GGUF F16 / Q8_0 / Q4_K_M 三個檔案
- [ ] 量化前後準確率對照表
- [ ] 汙染檢查報告
- [ ] 最終書面報告

### 12.3 報告必答問題（對應課程核心問題）

1. **一個 LLM 到底需要改變多少才能學會新任務？**
   → 用 D 系列（層選擇）與 B 系列（rank）的數據回答
2. **格式學習 vs 能力學習**：微調帶來的是真本事還是只會寫格式？
   → 用第 7.2 節的三層指標回答
3. **規模效應**：4B 是否明顯優於 2B？差距是否值得成本？
4. **★ 訓練方法的效率**：bf16 LoRA 相對於 QLoRA / DoRA，
   在**固定 16 GB VRAM 的硬體條件下**，各自的記憶體與準確率是多少？
   QLoRA 在此架構上是否可用？
   → 用 E 系列（E1/E2 對比 E3/E4/E5/E6/E7）的數據回答
   → **這是本專案對課程「Efficient Fine-Tuning」主題最直接的回應**
5. **部署量化的代價**：Q4 相對於 bf16 損失多少準確率？值得嗎？

> 💡 **建議的報告核心圖表（一張圖講完整個專案）：**
> 橫軸為 VRAM 峰值，縱軸為 MATH-500 準確率，
> 把 E1（2B bf16）、E2（4B bf16）、E3/E4（DoRA）畫上去，
> 並把 QLoRA 標成「無法執行」的註記。
> 這張圖會直接呈現**「效率前緣」以及「哪些方法根本走不通」**。
> 這比任何文字敘述都有說服力。

---

## 13. 疑難排解

### 13.1 環境問題

| 症狀 | 原因 | 解法 |
| --- | --- | --- |
| `CUDA available: False` | 裝到 CPU 版 PyTorch | 重新用 `--index-url https://download.pytorch.org/whl/cuXXX` 安裝 |
| `CUDA out of memory`（載入時） | 其他程式佔用 VRAM | 關閉瀏覽器、遊戲；用 `nvidia-smi` 檢查 |
| `KeyError: 'qwen3_5'` | transformers 版本太舊 | `pip install "transformers>=5.0.0" --upgrade` |
| Unsloth 匯入失敗 | 版本不合 | `pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo` |
| Triton 核心編譯很久 | Qwen3.5 使用自訂 Mamba 核心 | 正常現象，第一次較慢，請耐心等待 |
| `No module named 'bitsandbytes'` | 沒安裝量化套件 | `pip install bitsandbytes` |
| QLoRA 載入時 `CUDA error` | bitsandbytes 與 CUDA 版本不符 | 更新 `bitsandbytes`；或先退回 `--quant bf16` 繼續其他實驗 |
| QLoRA 載入時抱怨 dtype | 4-bit 與 bf16 混用設定衝突 | 確認 `load_in_4bit=True` 時 `load_in_16bit=False`（腳本已自動處理） |

### 13.2 訓練問題

| 症狀 | 原因 | 解法 |
| --- | --- | --- |
| Loss 完全不動 | 資料格式錯誤（缺 `text` 欄位） | 檢查 JSONL 的欄位名稱 |
| 輸出沒有 `<think>` | 訓練資料缺少思考標籤 | 檢查 4.3 節的格式 |
| 模型一直重複 | 學習率過高或訓練過度 | 降低 lr 到 1e-4，減少 epoch |
| 思考被截斷 | `max_seq_length` 太小 | 提高至 4096，或過濾過長樣本 |
| 4B 訓練 OOM | 序列長度太大 | 降到 1536 或 1024 |

### 13.3 GGUF 問題

| 症狀 | 原因 | 解法 |
| --- | --- | --- |
| `missing tensor 'blk.32...'` | 轉換時打包了 MTP 層 | **轉換時加上 `--no-mtp`**（見 9.2），並使用 2026-09-07 之後的 llama.cpp |
| `missing tensor 'blk.40.ssm_conv1d'` | MTP 區塊被誤判為 recurrent 層 | 更新 llama.cpp 到含 `recurrent_layers` 補位的版本（commit `9a757058` 之後） |
| `Bad layer N ... Must be in [0, N)`（量化時） | 舊版量化器的層編號錯誤 | 更新 llama.cpp（#24986 已修） |
| 轉換時 `Unsupported architecture` | transformers 或 llama.cpp 太舊 | 兩者都更新到最新；`pip install "transformers>=5.0.0"` |
| lazy transpose 相關錯誤 | 轉換腳本的 lazy 路徑問題 | 加上 `--no-lazy`（代價是 RAM 用量大增） |
| 找到的 `llama-quantize.exe` 路徑不對 | 工具已搬到 `tools/quantize/` | 用 `Get-ChildItem -Recurse -Filter llama-quantize.exe` 找 |
| 載入後輸出亂碼 | 聊天模板不符 | 加上 `--jinja`，或手動指定你訓練時用的模板 |
| Q4 品質明顯變差 | 混合架構對量化敏感 | 用 imatrix；或用 `--tensor-type attn_q=Q8_0` 把注意力層留在高精度；或改用 Q5_K_M / Q8_0 |
| Ollama 載入失敗 | Qwen3.5 在 Ollama 上支援狀況未定 | 先改用 llama.cpp 或 LM Studio |

### 13.4 評估問題（LM Studio）

| 症狀 | 原因 | 解法 |
| --- | --- | --- |
| `Cannot connect to LM Studio` / 連線被拒 | LM Studio 沒開，或沒啟動 API server | 開啟 LM Studio；確認它正在執行 |
| **推論很慢、GPU 使用率 0%** | **runtime 選成 CPU 或 Vulkan** | 到 Runtime 面板改成 **CUDA** 版並重新載入模型 |
| 只有部分層在 GPU | `gpu ratio` 不是 1.0，或 VRAM 不足 | 腳本已設 `{"ratio": 1.0}`；若 VRAM 不足請改用 Q4_K_M |
| 模型載入時 VRAM 不足 | F16 的 4B 約 8.4 GB，加上 context 可能超標 | 降低 `--context`，或改用 Q8_0 / Q4_K_M |
| 找不到模型 | 沒匯入，或 model-key 打錯 | `lms ls` 查看正確名稱；或直接傳 `.gguf` 完整路徑 |
| `lms` 指令不存在 | LM Studio 未安裝或未加入 PATH | 重開終端機；或直接用 `.gguf` 路徑跳過匯入 |
| 答案抽取率很低 | 模型沒用 `\boxed{}` | 這是格式問題，檢查訓練資料與 few-shot 提示 |
| Base 模型評估結果極差 | Base 模型不會 follow 格式 | 使用 few-shot 基準線（見 7.1） |
| `math_verify` 報錯 | 參數順序錯誤或 LaTeX 異常 | 確認 `verify(gold, pred)` 順序；用 try/except 包裹 |
| **所有答案都被判錯** | **忘了傳 `parsing_timeout=None`** | 見 7.2 節的 Windows 靜默失敗說明 |
| 評估中途卡住 | 思考太長，超過 `--max-tokens` | 這是正常的（會被截斷），請確認截斷率而非當機 |

---

## 附錄 A：常用指令速查

```powershell
# ========== 環境（訓練用 venv）==========
.\.venv\Scripts\Activate.ps1              # 啟用虛擬環境
python -c "import torch; print(torch.cuda.is_available())"   # 驗證訓練用 GPU

# ===== 下載 =====
hf download Qwen/Qwen3.5-2B-Base --local-dir models\Qwen3.5-2B-Base
hf download Qwen/Qwen3.5-4B-Base --local-dir models\Qwen3.5-4B-Base

# ===== 資料 =====
python scripts/prepare_data.py

# ===== 訓練（在 venv 中執行）=====
# 2B bf16 LoRA（★ 預設，約 4.6 GB）
python scripts/train_lora.py --model Qwen/Qwen3.5-2B-Base --data data/processed/sft_5k.jsonl --out outputs/qwen35-2b-bf16-r16 --quant bf16 --rank 16
# 4B bf16 LoRA（★ 4B 主力，約 9.5 GB）
python scripts/train_lora.py --model Qwen/Qwen3.5-4B-Base --data data/processed/sft_5k.jsonl --out outputs/qwen35-4b-bf16-r16 --quant bf16 --rank 16
# DoRA（幾乎零成本，值得做）
python scripts/train_lora.py --model Qwen/Qwen3.5-2B-Base --data data/processed/sft_5k.jsonl --out outputs/qwen35-2b-dora-r16 --quant dora --rank 16
# QLoRA（⚠️ 預期會崩潰，見 5.6 節）
python scripts/train_lora.py --model Qwen/Qwen3.5-2B-Base --data data/processed/smoke.jsonl --out outputs/qwen35-2b-qlora-crash --quant qlora --max-seq-len 512

# ===== 合併（adapter 一律合併回「原始 bf16 基礎模型」）=====
python scripts/merge_adapter.py --base Qwen/Qwen3.5-2B-Base --adapter outputs/qwen35-2b-bf16-r16 --out outputs/merged/qwen35-2b-math-bf16

# ===== GGUF（在 venv 中執行）=====
python convert_hf_to_gguf.py <merged_model_dir> --outtype f16 --outfile gguf\model-f16.gguf --no-mtp
.\build\bin\Release\llama-quantize.exe --dry-run gguf\model-f16.gguf Q4_K_M
.\build\bin\Release\llama-quantize.exe gguf\model-f16.gguf gguf\model-Q8_0.gguf Q8_0
.\build\bin\Release\llama-quantize.exe gguf\model-f16.gguf gguf\model-Q4_K_M.gguf Q4_K_M

# ===== 匯入 LM Studio（評估前）=====
lms ls                                                    # 列出已匯入的模型
lms import D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf    # 匯入
lms import D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q8_0.gguf
lms import D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q4_K_M.gguf

# ===== 評估（★ 透過 LM Studio，需先開啟 LM Studio 且 runtime = CUDA）=====
# 煙霧測試：先跑 5 題，人工檢查輸出格式
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf" --tag 2b-math-f16 --mode think --limit 5
# 完整評估：三種量化各跑一次
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-2b-math-f16.gguf"    --tag 2b-math-f16  --mode think --limit 200
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q8_0.gguf"   --tag 2b-math-q8   --mode think --limit 200
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-2b-math-Q4_K_M.gguf" --tag 2b-math-q4km --mode think --limit 200
# base 對照
python scripts/evaluate_lmstudio.py --model-key "D:\DSAI\DSAI4207\gguf\qwen35-2b-base-f16.gguf" --tag 2b-base-f16 --mode think --limit 200

# ===== 配對比較 =====
python scripts/compare_paired.py --a results/eval_2b-math-f16_think.json --b results/eval_2b-math-q4km_think.json

# ===== llama.cpp 環境健康檢查（非評估路徑）=====
.\build\bin\Release\llama-cli.exe -m gguf\model-Q4_K_M.gguf -p "your question" -n 512 -ngl 99 --jinja
```

---

## 附錄 B：名詞對照表

| 英文 | 中文 | 說明 |
| --- | --- | --- |
| SFT (Supervised Fine-Tuning) | 監督式微調 | 用「輸入-輸出」配對資料訓練模型 |
| PEFT | 參數高效微調 | 只訓練少量參數的微調方法總稱 |
| LoRA (Low-Rank Adaptation) | 低秩適應 | 只訓練小的附加矩陣，大幅降低資源需求 |
| QLoRA | 量化 LoRA | 把**凍結的基礎權重**量化到 4-bit 再掛 LoRA |
| DoRA (Weight-Decomposed LoRA) | 權重分解 LoRA | 把權重拆成「大小」與「方向」分別調整，品質通常略優於 LoRA |
| FFT (Full Fine-Tuning) | 全參數微調 | 更新所有權重 |
| BitsAndBytes | － | 提供 4-bit / 8-bit 量化功能的 Python 套件 |
| NF4 (4-bit NormalFloat) | 4-bit 常態浮點 | QLoRA 預設的 4-bit 量化資料型別 |
| Double quantization | 雙重量化 | 連量化常數也一起量化，額外省約 0.37 bytes/參數 |
| Compute dtype | 計算精度 | 4-bit 權重在做矩陣乘法時解量化成的精度（通常 bf16） |
| rank (r) | 秩 | LoRA 矩陣的大小，越大容量越強 |
| adapter | 適配器 | LoRA 訓練出來的權重檔案 |
| GGUF | GPT-Generated Unified Format | llama.cpp 生態系使用的模型格式（**LM Studio 只吃這個**） |
| LM Studio | － | 本專案使用的**評估後端**；用 CUDA runtime 在 GPU 上跑 GGUF 推論 |
| `lmstudio` (SDK) | － | LM Studio 的 Python 客戶端套件，安裝在 venv 中 |
| `lms` | － | LM Studio 的命令列工具，用於匯入與列出模型 |
| GPU offload / `gpu ratio` | GPU 卸載比例 | 決定多少層放到 GPU。**設 1.0 才是全 GPU**，關係到 CUDA 是否真的生效 |
| Quantization | 量化 | 降低權重精度以縮小體積 |
| Q8_0 / Q4_K_M | － | 兩種 **GGUF 部署**量化等級（8-bit / 約 4-bit） |
| Gradient checkpointing | 梯度檢查點 | 用重算換記憶體的技巧，可省 50–70% 活化值 |
| Activation | 活化值 | 前向傳播過程中的中間張量，長序列時是主要記憶體消耗 |
| 8-bit optimizer | 8-bit 優化器 | 把優化器狀態從 8 bytes 降到 2 bytes/參數 |
| Paged optimizer | 分頁優化器 | 快 OOM 時把優化器狀態分頁到 CPU RAM |
| Thinking tag | 思考標籤 | `<think>...</think>`，包住推理過程 |
| CoT (Chain-of-Thought) | 思維鏈 | 逐步推理 |
| Base model | 基礎模型 | 只做過預訓練、未經指令微調 |
| Gated DeltaNet | 閘控 Delta 網路 | Qwen3.5 使用的線性注意力（SSM）機制 |
| MTP / NextN | 多 token 預測 | 額外的預測頭，用於加速推論 |
| Contamination | 資料汙染 | 測試題出現在訓練集中 |
| OOM | 記憶體不足 | Out of Memory |
| Epoch | 訓練輪次 | 完整看過訓練資料一次 |

---

## 附錄 C：參考資料

**模型與架構**
- Qwen3.5-2B-Base：https://huggingface.co/Qwen/Qwen3.5-2B-Base
- Qwen3.5-4B-Base：https://huggingface.co/Qwen/Qwen3.5-4B-Base
- Qwen3.5 官方部落格：https://qwen.ai/blog?id=qwen3.5

**訓練框架**
- Unsloth Qwen3.5 微調指南：https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Unsloth 匯出 GGUF：https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf
- PyTorch 安裝說明：https://pytorch.org/get-started/locally/

**資料集**
- 數學資料集總覽：https://github.com/amao0o0/awesome-AI-Math-Datasets
- OpenR1-Math-220k：https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- OpenMathReasoning：https://huggingface.co/datasets/nvidia/OpenMathReasoning
- DeepMath-103K：https://huggingface.co/datasets/zwhe99/DeepMath-103K
- Bespoke-Stratos-17k：https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k

**工具鏈（GGUF 相關，已核實）**
- llama.cpp 主專案：https://github.com/ggml-org/llama.cpp
- 最新 release（建議使用）：https://github.com/ggml-org/llama.cpp/releases/tag/b10906
- `recurrent_layers` MTP 補位（關鍵修補）：https://github.com/ggml-org/llama.cpp/commit/9a7570587ce908b0073a0458877205b80627f393
- MTP 層數問題（仍開啟）：https://github.com/ggml-org/llama.cpp/issues/24737
- 缺少 `ssm_conv1d`（仍開啟）：https://github.com/ggml-org/llama.cpp/issues/23033
- 量化器層編號問題（已修）：https://github.com/ggml-org/llama.cpp/issues/24661
- 混合架構載入失敗：https://github.com/ggml-org/llama.cpp/issues/26916
- Unsloth 量化基準（敏感層分析）：https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks
- Unsloth Qwen3.5 總覽：https://unsloth.ai/docs/models/qwen3.5
- 現成 GGUF：https://huggingface.co/unsloth/Qwen3.5-4B-GGUF

**評估（LM Studio）**
- LM Studio 官網：https://lmstudio.ai/
- LM Studio 系統需求：https://lmstudio.ai/docs/app/system-requirements
- **Python SDK 總覽**：https://lmstudio.ai/docs/python
- 載入模型與 GPU 設定：https://lmstudio.ai/docs/python/manage-models/loading
- 推論參數設定：https://lmstudio.ai/docs/python/llm-prediction/parameters
- 匯入模型：https://lmstudio.ai/docs/app/advanced/import-model
- math-verify：https://github.com/huggingface/Math-Verify
- math-verify Windows 靜默失敗 bug：https://github.com/huggingface/Math-Verify/issues/79
- math-verify 套件頁：https://pypi.org/project/math-verify/
- GSM8K：https://huggingface.co/datasets/openai/gsm8k
- MATH-500：https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- MinervaMath：https://huggingface.co/datasets/math-ai/minervamath
- AIME 2024：https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
- BeyondAIME：https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME
- lm-evaluation-harness：https://github.com/EleutherAI/lm-evaluation-harness

**論文**
- Qwen3 Technical Report：https://arxiv.org/abs/2505.09388
- LoRA: Low-Rank Adaptation of Large Language Models：https://arxiv.org/abs/2106.09685
- OpenMathReasoning：https://arxiv.org/abs/2504.16891
- Don't Pass@k（評估方法論）：https://arxiv.org/abs/2510.04265
