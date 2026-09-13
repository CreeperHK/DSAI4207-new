# Qwen3.5 (2B/4B-Base) + Unsloth + llama.cpp + LM Studio + math-verify — fact check

Research date: as fetched. Method: web search + page fetch of primary sources (HF raw files & API,
PyPI JSON API, download.pytorch.org index, GitHub raw source and GitHub REST API, vendor docs).
Direct byte-level network access from the sandbox is blocked (pwsh `Invoke-WebRequest` → connection
closed; `curl.exe` → `schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`), so all
evidence is via the fetch tooling.

Legend: **VERIFIED** = primary source quote/field observed. **UNVERIFIED** = could not confirm.

---

## Q1. Do the base repos exist, and with what metadata? — VERIFIED

Both exist and are **not gated**.

| field | `Qwen/Qwen3.5-2B-Base` | `Qwen/Qwen3.5-4B-Base` |
|---|---|---|
| repo id | `Qwen/Qwen3.5-2B-Base` | `Qwen/Qwen3.5-4B-Base` |
| `gated` | `false` | `false` |
| `architectures` | `["Qwen3_5ForConditionalGeneration"]` | same |
| `model_type` (top level) | `"qwen3_5"` | `"qwen3_5"` |
| `text_config.model_type` | `"qwen3_5_text"` | `"qwen3_5_text"` |
| parameter count (safetensors) | **2,274,069,824** (BF16 2,274,067,232 + F32 2,592) | **4,659,865,088** (BF16 4,659,861,248 + F32 3,840) |
| text layers | `num_hidden_layers: 24` (`layer_types` = 24 entries, 18 `linear_attention` / 6 `full_attention`) | `num_hidden_layers: 32` (24 / 8) |
| `vocab_size` | **248320** | **248320** |
| `full_attention_interval` | 4 | 4 |
| `mtp_num_hidden_layers` | **1** | **1** |
| license | apache-2.0 | apache-2.0 |
| `pipeline_tag` | `image-text-to-text` | `image-text-to-text` |
| HF `transformersInfo.auto_model` | `AutoModelForMultimodalLM` | same |
| created / lastModified | 2026-02-28 / 2026-04-23 | 2026-02-27 / 2026-04-23 |

Sources:
- https://huggingface.co/api/models/Qwen/Qwen3.5-2B-Base
- https://huggingface.co/api/models/Qwen/Qwen3.5-4B-Base
- https://huggingface.co/Qwen/Qwen3.5-2B-Base/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3.5-4B-Base/raw/main/config.json

Notable, not just trivia:
- These are **not text-only causal LMs**. `pipeline_tag` is `image-text-to-text`, the configs carry a
  full `vision_config` (depth 24, hidden 1024) plus `vision_start/end/pad`, `image_pad`, `video_pad`
  token ids, and the `architectures` entry is `...ForConditionalGeneration`. The HF docs describe
  Qwen3.5 as "Qwen's natively multimodal foundation model family".
- HF docs: "Use [`Qwen3_5ForCausalLM`] for text-only generation with [`Qwen3_5TextConfig`]; use
  [`Qwen3_5ForConditionalGeneration`] with the full [`Qwen3_5Config`] and a processor (...) to feed
  interleaved image/video + text".
  (https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/qwen3_5.md)
- `mtp_num_hidden_layers: 1` is why GGUF conversion has the extra-block hazard — see Q8.
- Chat-template inconsistency between the two base repos (as fetched): 2B-Base emits
  `{%- if enable_thinking is defined and enable_thinking is true %}` while 4B-Base emits
  `is false` — i.e. inverted default logic in the two cards. Flagging because the notebook will
  apply a chat template; worth testing per-model.

## Q2. `<think>` / `</think>` token ids — ids VERIFIED, pipe claim UNVERIFIED

`added_tokens_decoder` from raw `tokenizer_config.json`:

- **Qwen3.5**: `248068` = `<think>`, `248069` = `</think>`; both `"special": false`, and neither is
  in `additional_special_tokens`.
  (https://huggingface.co/Qwen/Qwen3.5-2B-Base/raw/main/tokenizer_config.json)
- **Qwen3** (`Qwen/Qwen3-0.6B` and `Qwen/Qwen3-4B-Base`): `151667` = `<think>`, `151668` = `</think>`;
  both `"special": false`.
  (https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/tokenizer_config.json,
  https://huggingface.co/Qwen/Qwen3-4B-Base/raw/main/tokenizer_config.json)

The Qwen3.5 chat template (as fetched) writes the tag with no pipe, e.g.
`{{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}`.

**UNVERIFIED — do not assert in the notebook:** the claim that the old Qwen3 tag is ` thinking`
(with pipes) while Qwen3.5 uses `<think>` without a pipe. Every copy I retrieved for *both* model
families shows the same no-pipe spelling, through two different fetch routes (`web_fetch` and
`read_page`/Firecrawl). Pipe characters elsewhere (`<|im_start|>`, `<|endoftext|>`, `<|fim_prefix|>`)
survive intact, so I cannot distinguish "Qwen3 genuinely has no pipe" from "the fetch pipeline
normalises the think tag". Byte-level confirmation was impossible:
- `tokenizer.json` is an LFS pointer (`size 12807196`, pointer content only);
- direct network egress from the sandbox is blocked (see header).

Independently corroborated only at the id level: Qwen's own Qwen3 blog parses
`# rindex finding 151668 (</think>)` (https://qwenlm.github.io/blog/qwen3/).

## Q3. Unsloth Qwen3.5 fine-tune recipe — VERIFIED (with a correction)

Source: https://unsloth.ai/docs/models/qwen3.5/fine-tune
(raw markdown: https://unsloth.ai/docs/models/qwen3.5/fine-tune.md)

**Correction: that page does not give a `FastVisionModel.from_pretrained(...)` call at all.** The
code-based SFT recipe uses `FastLanguageModel.from_pretrained(...)`:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3.5-27B",
    max_seq_length = max_seq_length,
    load_in_4bit = False,     # MoE QLoRA not recommended, dense 27B is fine
    load_in_16bit = True,     # bf16/16-bit LoRA
    full_finetuning = False,
)
```

So yes — `load_in_4bit`, `load_in_16bit`, `full_finetuning` are used. Separate MoE loader example
uses `FastModel.from_pretrained(model_name="unsloth/Qwen3.5-35B-A3B", load_in_4bit=False,
load_in_16bit=True, full_finetuning=False)`.

The four `finetune_*` flags are real and appear on **`FastVisionModel.get_peft_model`** (identical
block on the Qwen3.5 page and on https://unsloth.ai/docs/basics/vision-fine-tuning.md):

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,               # And LoftQ
    target_modules = "all-linear",    # Optional now! Can specify a list if needed
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)
```

The text-only SFT path instead uses `FastLanguageModel.get_peft_model(model, r=16,
target_modules=[...], lora_alpha=16, lora_dropout=0, bias="none",
use_gradient_checkpointing="unsloth", random_state=3407, max_seq_length=...)` — note the
`finetune_*` flags are not in that block.

Other verified doc statements:
- "**Please use `transformers v5` for Qwen3.5. Older versions will not work.**"
- bf16 LoRA VRAM: "**0.8B**: 3GB • **2B**: 5GB • **4B**: 10GB • **9B**: 22GB • **27B**: 56GB"
  (so 2B/4B bf16 LoRA are well within a 16 GB RTX 4070).
- "Qwen3.5 is "Causal Language Model with Vision Encoder" (it's a unified VLM), so ensure you have
  the usual vision deps installed (`torchvision`, `pillow`)".
- "If training seems **slower than usual**, it's because Qwen3.5 use custom Mamba Triton kernels."
- GGUF export: `model.save_pretrained_gguf("directory", tokenizer, quantization_method = "q4_k_m")`.

## Q4. Does Unsloth say QLoRA (4-bit) is not recommended for Qwen3.5? — VERIFIED, yes

Exact wording, in a warning callout on both the rendered page and the `.md` source:

> "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense,
> due to higher than normal quantization differences."

And in the MoE section:

> "**Best to use bf16 setups (e.g. LoRA or full fine-tuning)** (MoE QLoRA 4‑bit is not recommended due
> to BitsandBytes limitations)."

Internal inconsistency worth knowing: the inline comment in the same page's code sample reads
`load_in_4bit = False,     # MoE QLoRA not recommended, dense 27B is fine`, which contradicts
"no matter MoE or dense". The prose callout is the more explicit/most recent statement.

## Q5. Unsloth issues #10010 and #9867 — VERIFIED (both still OPEN)

**#10010** — https://github.com/unslothai/unsloth/issues/10010
Title: `[Bug] Qwen3.8-27B (Qwen3.5 hybrid linear-attention) QLoRA fails with "mat1 and mat2 shapes
cannot be multiplied" in linear_attn.in_proj_z — missing quant_state in
unsloth/qwen3.8-27b-unsloth-bnb-4bit`
State **open**, opened 2026-08-30, 0 comments, labels `bug` + `feature request`.
Reported error: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (258x5120 and 1x15728640)`
at `Qwen3_5GatedDeltaNet_forward → z = self.in_proj_z(hidden_states)`, preceded by
`[bitsandbytes.nn.modules|WARNING] FP4 quantization state not initialized.`
Reported root cause: `FastModel.from_pretrained(..., load_in_4bit=True)` silently redirects to the
pre-quantized mirror `unsloth/qwen3.8-27b-unsloth-bnb-4bit`, where `linear_attn.*` GatedDeltaNet
projections have no `quant_state` ("Skipping model.language_model.layers.0.linear_attn.in_proj_z:
no quant_state found"), so the packed uint8 buffer reaches `F.linear` undequantized.
Reported workaround: `use_exact_model_name=True` + explicit `BitsAndBytesConfig(load_in_4bit=True,
llm_int8_skip_modules=[...linear_attn...])` with `load_in_4bit=False`.

**#9867** — https://github.com/unslothai/unsloth/issues/9867
Title: `[Bug] Qwen3.5 GatedDeltaNet + bnb-4bit: packed 4-bit weight passed undequantized to F.linear
(mat1/mat2 shape error) — ROCm gfx1201, native Windows`
State **open**, opened 2026-08-27, **3 comments**, last update 2026-08-31.
Same crash: `mat1 and mat2 shapes cannot be multiplied (82x5120 and 1x15728640)` from
`unsloth_zoo/temporary_patches/bitsandbytes.py`, line 78
`torch.nn.functional.linear(x, weight, bias)`. Reporter notes `15,728,640 = 6144 x 5120 / 2`, the
packed nf4 shape of a (6144, 5120) weight. Environment: AMD Radeon AI PRO R9700 (gfx1201), native
Windows 11, unsloth 2026.8.21, torch 2.11.0+rocm7.13.0, transformers 5.5.0.
Comments: (a) independent repro on CUDA/NVIDIA NGC with `Qwen/Qwen3.8-27B` + `FastModel` + LoRA,
same signature and same pre-quantized-mirror root cause; (b) contributor `lonexreb` points to
proposed fix `unslothai/unsloth-zoo#1132` (in the patched `Linear4bit.forward`, re-derive quant
state and route through `matmul_4bit` instead of `F.linear`) and asks for an end-to-end run on real
hardware; (c) same contributor on 2026-08-31 repeating that merge is blocked on that hardware
verification. Still open.

Scope caveat: both reports are about **Qwen3.8-27B**, not Qwen3.5-2B/4B. Neither is direct evidence
about the two target checkpoints. But both sit in the same Qwen3.5-family hybrid GatedDeltaNet +
bnb-4bit code path — consistent with Unsloth's blanket "no QLoRA" guidance for Qwen3.5.

## Q6. Is `flash-linear-attention` required? PyPI name/version, Windows install — VERIFIED

**Not required** — it is optional, with a silent fallback. Verbatim from the transformers Qwen3.5
docs:
> "The DeltaNet path (`Qwen3NextGatedDeltaNet`) needs the optional `causal_conv1d` (from
> [Dao-AILab](https://github.com/Dao-AILab/causal-conv1d)) and `fla` packages for its fast kernels —
> without them, the model silently falls back to slower and more memory hungry PyTorch ops."

(https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/qwen3_5.md)

PyPI (JSON API):
- `flash-linear-attention`, **latest 0.5.2** (uploaded 2026-07-27), requires-python `>=3.10`,
  pure-python wheel `flash_linear_attention-0.5.2-py3-none-any.whl`.
  https://pypi.org/pypi/flash-linear-attention/json
- There is a **second package**: `fla-core` (kernels in `fla/ops`, `fla/modules`, `fla/utils`);
  `flash-linear-attention` = `fla/layers` + `fla/models` + `fla-core` dep.
- Behavior change vs pre-0.5, from INSTALL.md: "bare `pip install flash-linear-attention` no longer
  pulls `torch` / `triton`. Pick a backend extra." and "Bare `pip install flash-linear-attention`
  no longer imports." CUDA install: `pip install flash-linear-attention[cuda]`.
  https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/INSTALL.md

Windows: **fla's INSTALL.md documents no Windows path at all** (backend table covers CUDA/ROCm/XPU/
NPU/CPU only), and the CUDA extra takes `triton` from PyPI.
- Upstream `triton` on PyPI: latest **3.8.0**, requires-python `>=3.10,<3.15`.
  https://pypi.org/pypi/triton/json — in the metadata I retrieved (220 wheel entries across
  releases 0.4.x–3.5.x) there were **zero `win_amd64` wheels**. So on native Windows the upstream
  package does not install; the community fork is required.
- `triton-windows`: **latest 3.8.0.post28**, requires-python `>=3.10,<3.15`, classifiers
  Python 3.10–3.14. https://pypi.org/pypi/triton-windows/json , https://pypi.org/project/triton-windows/
- Fork README (https://github.com/woct0rdho/triton-windows): "Since `triton-windows 3.2.0.post11`,
  the wheels are published to PyPI, so you don't need to manually download a wheel from GitHub
  releases", and the documented sequence is `pip uninstall triton` then
  `pip install -U "triton-windows<3.7"`.

Unsloth's own Qwen3.5 MoE notebook installs these explicitly, which is a good template:
`pip install --no-build-isolation flash-linear-attention causal_conv1d==1.6.0`, with the comment
"causal_conv1d is supported only on torch==2.8.0", and sets `os.environ["FLA_TILELANG"] = "0"`.
(https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_MoE.ipynb)

Caveats flagged: (i) the "no win_amd64 wheels" check was against a truncated PyPI capture (releases
up to ~3.5.0); (ii) I could **not** verify a `cp312` wheel exists for `triton-windows 3.8.0.post28`
specifically — only that `requires_python` admits 3.12.

## Q7. PyTorch cu128 wheels — VERIFIED

- `https://download.pytorch.org/whl/cu128` exists (HTTP 200) and lists `torch/`, `torchvision/`,
  `torchaudio/`, `triton/`, `pytorch-triton/`, `flash-attn-3/`, etc.
- https://download.pytorch.org/whl/cu128/torch/ lists versions **2.7.0, 2.7.1, 2.8.0, 2.9.0, 2.9.1,
  2.10.0, 2.11.0** → highest/current is **torch 2.11.0+cu128**.
- A cp312 Windows wheel exists for it:
  `torch-2.11.0+cu128-cp312-cp312-win_amd64.whl`
  (also cp313/cp313t/cp314/cp314t win_amd64 for 2.11.0).

## Q8. llama.cpp `--no-mtp` / `--no-nextn` — VERIFIED (flag exists; documented issue exists)

Source `convert_hf_to_gguf.py` (master), verbatim:
```python
parser.add_argument(
    "--no-nextn", "--no-mtp", dest="no_mtp", action="store_true",
    help="Exclude NextN speculative draft tensors from the converted GGUF. Pair with --mtp or --dspark on a second run to publish target and draft as two files.",
)
```
Related flags/logic in the same file:
```python
parser.add_argument(
    "--mtp", action="store_true",
    help="Export only the multi-token prediction (MTP) head as a separate GGUF, suitable for use as a speculative draft. An 'mtp-' prefix will be added to the output file name.",
)
...
if sum((args.mtp, args.no_mtp, args.dspark)) > 1:
    logger.error("--mtp, --no-nextn, and --dspark are mutually exclusive")
...
if args.mtp or args.no_mtp:
    if not model_class.supports_mtp_export:
        logger.error("--mtp / --no-nextn are not supported for %s", model_architecture)
```
(https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py)

**Documented issue — yes.** PR #27132 (state `open`, `draft`) states verbatim:
> "Additional note (not a code change): Qwen3.5 checkpoints need `--no-mtp` - the MTP mixin can add a
> speculative block (33 blocks vs expected 32)."

and in its verification section: "`Qwen/Qwen3.5-9B` converted with `--no-mtp --no-lazy` and loaded
in llama-server (model load + generation OK)".
https://github.com/ggml-org/llama.cpp/pull/27132

Corroborating open issues (same 33-vs-32 block symptom):
- #26916 — "Eval bug: Qwen3.5-Hybrid model (qwen3_5, SSM+Attention) fails to load — "tensor
  'blk.32.attn_norm.weight' not found""; notes GGUF metadata contains
  `qwen35.nextn_predict_layers = 1` alongside 32 real blocks and guesses the loader expects
  `n_layer + 1`. https://github.com/ggml-org/llama.cpp/issues/26916
- #24737 — "Eval bug: Qwen3.5-4B: GGUF conversion/load expects 33 blocks, model only has 32"
  (open, label `stale`), `missing tensor 'blk.32.attn_norm.weight'`.
  https://github.com/ggml-org/llama.cpp/issues/24737

Also relevant to conversion of these exact repos: #27019 (open) reports
`RuntimeError: shape '[16, 2, 1, 1]' is invalid for input of size 65536` in `conversion/qwen.py`
`_reorder_v_heads` for qwen3_5 conversion (ssm_conv1d kernel dim + in_proj_a/b expansion), with
#27132 as the proposed (unmerged) fix. https://github.com/ggml-org/llama.cpp/issues/27019

## Q9. LM Studio Python SDK — VERIFIED except `get_model_info()`

Sources: https://lmstudio.ai/docs/python , https://lmstudio.ai/docs/python/llm-prediction/parameters ,
https://lmstudio.ai/docs/python/llm-prediction/chat-completion ,
https://lmstudio.ai/docs/python/model-info/get-model-info

| claimed API | verdict | evidence |
|---|---|---|
| `lms.llm(key, config={...})` | **VERIFIED** | `model = lms.llm("qwen2.5-7b-instruct", config={"contextLength": 8192, "gpu": {"ratio": 0.5}})` |
| GPU config key `{"gpu": {"ratio": 1.0}}` | **VERIFIED shape** | same example; docs: "Set load-time parameters such as the context length, GPU offload ratio". Docs show `0.5`; `1.0` is the same nested key. Note: "if the model is already loaded, the given configuration will be **ignored**." |
| `lms.Chat(system_prompt)` | **VERIFIED** | `chat = lms.Chat("You are a resident AI philosopher.")` |
| `chat.add_user_message(...)` | **VERIFIED** | `chat.add_user_message("What is the meaning of life?")` |
| `model.respond(chat, config={"temperature":0.6,"maxTokens":50})` | **VERIFIED** | docs verbatim: `result = model.respond(chat, config={"temperature": 0.6, "maxTokens": 50,})` |
| `result.stats.stop_reason` | **VERIFIED** | `print("Stop reason:", result.stats.stop_reason)` |
| `result.stats.predicted_tokens_count` | **VERIFIED** | `print("Predicted tokens:", result.stats.predicted_tokens_count)` |
| `result.model_info.display_name` | **VERIFIED** | `print("Model used:", result.model_info.display_name)` |
| `model.get_model_info()` | **UNVERIFIED / likely wrong** | Not found. The "Get Model Info" page documents **`model.get_info()`**: `print(model.get_info())`, output shown as `LlmInstanceInfo.from_dict({...})` (fields include `architecture`, `contextLength`, `displayName`, `identifier`, `maxContextLength`, `modelKey`, `paramsString`, `path`, `sizeBytes`, `trainedForToolUse`, `type`, `vision`). Use `get_info()`. |

Also verified for Jupyter use: SDK ≥1.5.0's synchronous API "defaults to timing out after 60 seconds
with no activity", adjustable via `lmstudio.set_sync_api_timeout()` / `get_sync_api_timeout()`.
Docs also show `lms.llm()` with no argument (uses currently loaded model) and
`result = model.respond("What is the meaning of life?")` (plain string accepted).
`get_load_config()` and `get_context_length()` pages exist under /docs/python/model-info/.

## Q10. math-verify — usage and `boxed_match_priority` VERIFIED; issue #79 open

Real signatures, read from source:

`src/math_verify/parser.py`:
```python
def parse(
    pred: str,
    extraction_config: Sequence[ExtractionTarget] = [
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ],
    fallback_mode: Literal["no_fallback", "first_match"] = "first_match",
    extraction_mode: Literal["first_match", "any_match"] = "any_match",
    parsing_timeout: int = 5,
    raise_on_error: bool = False,
):
```
`src/math_verify/grader.py`:
```python
def verify(
    gold: list[Basic | MatrixBase | str] | Basic | MatrixBase | str,
    target: list[Basic | MatrixBase | str] | Basic | MatrixBase | str,
    float_rounding: int = 6,
    numeric_precision: int = 15,
    strict: bool = True,
    allow_set_relation_comp: bool = False,
    timeout_seconds: int | None = 5,
    raise_on_error: bool = False,
) -> bool:
```

- `parse(text, extraction_config=[...], parsing_timeout=None)` — **valid**, but the **default is 5,
  not None**. Passing `None` is explicitly supported: the code warns "Timeout is disabled as
  parsing_timeout is None or <= 0, you must provide the logic for timeout interuption yourself".
  (Docstring says "Defaults to 3" while the signature says 5 — doc inconsistency, source wins.)
- `verify(gold, pred, timeout_seconds=None)` — **valid**, default is 5. Argument **order matters**:
  README: "# Order here is important!" and the grader docstring: "Function is not symmetric, gold
  answer should be passed as gold and prediction as pred."
- `LatexExtractionConfig(boxed_match_priority=0)` — **REAL parameter**, default 50. Docstring verbatim:
  "boxed_match_priority (int): Priority for matching boxed expressions (e.g., \boxed{}).
   - 0: Highest priority (matched first)
   - 50: Default priority (matched after final answer patterns)
   - -1: Disable boxed expression matching"
  README recommends exactly this usage: "we recommended instructing the model to output the answer
  in a `\boxed{}` environment and set `boxed_match_priority` to 0 in the latex extraction config."
- Issue **#79** — https://github.com/huggingface/Math-Verify/issues/79
  Title: `Windows: parse/verify fail with timeout wrapper (AttributeError on local function
  pickling)`. State **open**, 0 comments, opened 2026-04-03. Reported on Windows 11, Python 3.13.3,
  math-verify 0.9.0:
  - `math_verify.parse(r"\boxed{4}", raise_on_error=True)` →
    `AttributeError: Can't get local object 'timeout.<locals>.decorator.<locals>.wrapper.<locals>.run_func'`
  - `math_verify.verify("4", "4", strict=True, timeout_seconds=1, raise_on_error=True)` → same.
  - With default `raise_on_error=False`: `parse(...)` returns `[]` and `verify(...)` returns `False`
    — i.e. **silent degradation**; sometimes `OSError: [WinError 6] The handle is invalid` from
    multiprocessing teardown.
  - Reporter's analysis: timeout wrapper uses a local function that is not picklable under Windows
    `spawn` multiprocessing.
  Still open with no comments → on native Windows, expect this; passing `parsing_timeout=None` /
  `timeout_seconds=None` is the documented escape hatch (handle timeouts yourself).
- Also from the README: extracted LaTeX must be inside a LaTeX environment
  (`\[ ... \]`, `$$ ... $$`, `\boxed{...}`, `$...$`, `\( ... \)`); `parse(gold, extraction_config=
  [LatexExtractionConfig()])` etc. Installation pins an antlr4 runtime: `pip install
  math-verify[antlr4_13_2]`, and "We recommend always specifying the antlr4 runtime".

---

## Cross-cutting flags for the notebook

1. **Don't claim the ` thinking` pipe difference** (Q2) — unverifiable here; use token ids instead.
2. **Don't use QLoRA** — Unsloth explicitly advises against it for Qwen3.5 dense *and* MoE (Q4).
   bf16 LoRA at 2B (5 GB) / 4B (10 GB) fits 16 GB.
3. **`FastVisionModel.from_pretrained` is not in the Qwen3.5 fine-tune docs** (Q3) — the documented
   loader is `FastLanguageModel.from_pretrained`; only `get_peft_model` is shown with `FastVisionModel`.
4. **`model.get_model_info()` is undocumented; use `model.get_info()`** (Q9).
5. **llama.cpp conversion of these repos is a live minefield** — pass `--no-mtp`, and expect the
   `conversion/qwen.py` ssm tensor-layout failures (#27019, unmerged fix in #27132) since both target
   repos carry `mtp_num_hidden_layers: 1` (Q8).
6. **`transformers v5` required** for Qwen3.5 per Unsloth, and the checkpoints' own
   `transformers_version` is `4.57.0.dev0` — verify what the installed stack actually resolves to.
