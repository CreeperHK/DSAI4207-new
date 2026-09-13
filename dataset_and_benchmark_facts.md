# Verified dataset / benchmark / Qwen3.5 facts (research note)

Compiled from live Hugging Face pages, the `datasets-server` API, dataset READMEs, HF
docs sources, and Qwen model cards. Every claim below carries a source. Unverified points
are flagged **UNVERIFIED**.

---

## 1. `open-r1/OpenR1-Math-220k`

### Configs and row counts (all configs have exactly one split: `train`)

| config | rows | download_size | dataset_size (arrow) |
|---|---|---|---|
| `default` | 93,733 | 2,149,897,914 B (~2.15 GB) | 5,079,805,007 B per `/info`; card YAML says 4,964,543,659 B (~4.96 GB) |
| `extended` | 131,396 | 2,063,936,457 B (~2.06 GB) | 4,770,393,404 B per `/info`; card YAML says 4,769,566,550 B |
| `all` | 225,129 | 4,221,672,067 B (~4.22 GB) | 10,161,169,949 B per `/info`; card YAML says 9,734,110,026 B |

- `all` = `default` + `extended` (93,733 + 131,396 = 225,129).
- The dataset page reports **"Number of rows: 450,258"** and **"Total file size: 12.6 GB"** —
  that is the sum over all three configs, and the repo stores the data three times
  (`data/`, `extended/`, `all/` directories).
- **Practical disk math:** `load_dataset(..., "default")` pulls ~2.15 GB; loading without a
  config name pulls `default` + `extended` (~4.2 GB); `all` alone is ~4.22 GB; the whole repo
  is ~12.6 GB.
- Sources:
  - https://datasets-server.huggingface.co/info?dataset=open-r1%2FOpenR1-Math-220k
  - https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/raw/main/README.md
  - https://huggingface.co/datasets/open-r1/OpenR1-Math-220k (page shows "Number of rows: 450,258" / "Total file size: 12.6 GB")
  - https://huggingface.co/api/datasets/open-r1/OpenR1-Math-220k/tree/main?recursive=true

Note: the `/info` `dataset_size` and the card YAML `dataset_size` disagree slightly for all
three configs; treat `download_size` as the authoritative "size on the wire".

### Exact column names (identical for all three configs — 14 columns)

`problem`, `solution`, `answer`, `problem_type`, `question_type`, `source`, `uuid`,
`is_reasoning_complete`, `generations`, `correctness_math_verify`, `correctness_llama`,
`finish_reasons`, `correctness_count`, `messages`

**Every one of the names you asked about exists.** None are missing.

Declared types:
- `problem`, `solution`, `answer`, `problem_type`, `question_type`, `source`, `uuid` → `string`
- `is_reasoning_complete` → `Sequence(bool)` / `List[bool]`
- `generations` → `Sequence(string)` / `List[str]`
- `correctness_math_verify` → `Sequence(bool)` / `List[bool]`
- `correctness_llama` → `Sequence(bool)` / `List[bool]`
- `finish_reasons` → `Sequence(string)` / `List[str]`
- `correctness_count` → `int64`
- `messages` → `List[{role: string, content: string}]`

Source (verbatim, README YAML):
```
  - name: generations
    sequence: string
```
Source: https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/raw/main/README.md

### `generations` type — **it is a list of strings, NOT a list of dicts**

Verified four independent ways:

1. Card YAML: `generations: sequence: string`.
2. datasets-server feature type: `{"feature":{"dtype":"string","_type":"Value"},"_type":"List"}`.
3. Actual row (`default`, row 0): `"generations":["<think>\nOkay, so I need to find the speed of the ship ... </think>\n\nLet \\( v \\) be ..."]`
   → `generations[0]` is a **string** that starts with a literal `<think>` block.
4. HF discussion #2 error dump for the `extended` config lists
   `generations: list<element: string>`.

Source: https://datasets-server.huggingface.co/rows?dataset=open-r1%2FOpenR1-Math-220k&config=default&split=train&offset=0&length=2
Source: https://huggingface.co/api/datasets/open-r1/OpenR1-Math-220k/discussions/2

**There is no `reasoning_content` key anywhere in this dataset.** The belief that
`generations` is a list of dicts with a `reasoning_content` field is **not supported by the
current (main, last data commit 2025-02-18) revision**. The only list-of-dicts column is
`messages`, whose keys are exactly `role` and `content`. (`reasoning_content` is a
vLLM/SGLang API-response field and a chat-template variable — see §8 — not a column/key here.)
Status of any revision/derivative where `generations` is a list of dicts: **UNVERIFIED —
no evidence found.**

Practical: split each generation on `</think>` yourself.

### `correctness_math_verify` type

`List[bool]` (declared `Sequence(bool)`), one element per generation, e.g. `[true,false]`.
Observed list lengths 1–6 (corresponds to `correctness_count`).
- `correctness_llama` is also `List[bool]` but is **`null` in most rows** (the Llama judge
  covered 12% of samples per the card); row 0 has `"correctness_llama": null`.
- `finish_reasons` is `List[str]` but also **`null` in many rows** (row 0 `null`;
  row 100 = `["stop","stop"]`).
- Whether `correctness_math_verify` itself is ever `null`: **UNVERIFIED** (non-null in the
  rows I inspected). Defensive coding (`or []`) is advisable.

### Other verified content facts (useful for SFT)

- Card: "The traces were verified using Math Verify for most samples and
  Llama-3.3-70B-Instruct as a judge for 12% of the samples, and each problem contains at
  least one reasoning trace with a correct answer."
- Card: "`default` with 94k problems and that achieves the best performance after SFT."
  "`extended` with 131k samples where we add data sources like `cn_k12` … we found that the
  performance after SFT to be lower than the `default` subset".
- Card: generation instruction prepended to the user prompt was verbatim
  `"Please reason step by step, and put your final answer within \boxed{}."` with a
  **16k token limit per generation**.
- `problem_type` has 8 distinct values and `question_type` 3 (viewer column stats);
  `source` has 7 (e.g. `olympiads`, `aops_forum`, `cn_contest`, `cn_k12`).
- License: apache-2.0.

---

## 2. `bespokelabs/Bespoke-Stratos-17k`

- Exact repo id: **`bespokelabs/Bespoke-Stratos-17k`** (config `default`,
  `dataset_name: bespoke-stratos-17k`).
- Rows: **16,710** (split `train`; only split). `num_bytes` 267,122,885 (~267 MB),
  `download_size` 125,307,344 (~125 MB).
- Columns: **`system`** (`string`) and **`conversations`** (list).
- `conversations` **is a list of dicts with keys exactly `from` and `value`** (both strings):
  `"conversations":[{"from":"user","value":"..."},{"from":"assistant","value":"..."}]`
- `from` values: **`user`** and **`assistant`** (verified in row 0). The dataset-wide set of
  distinct `from` values was **not** exhaustively enumerated — **partially UNVERIFIED**
  (the `datasets-server` `/filter` endpoint rejects the nested `where` syntax needed to test
  this). The separate top-level `system` column holds the system prompt, so there is no
  reason to expect a `system` role inside `conversations`, but that is inference, not a quote.
- Assistant text uses custom delimiters, not `<think>`:
  `<|begin_of_thought|> … <|end_of_thought|>` then `<|begin_of_solution|> … <|end_of_solution|>`
  (verbatim from row 0).
- Composition per the card: "5k coding data from APPs and TACO, and 10k math data from AIME,
  MATH, and Olympiads subsets of the NuminaMATH dataset, and 1k science and puzzle data from
  STILL-2."
- Sources: https://datasets-server.huggingface.co/info?dataset=bespokelabs%2FBespoke-Stratos-17k ·
  https://datasets-server.huggingface.co/rows?dataset=bespokelabs%2FBespoke-Stratos-17k&config=default&split=train&offset=0&length=1 ·
  https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k/raw/main/README.md

---

## 3. `zwhe99/DeepMath-103K`

- Exact repo id: **`zwhe99/DeepMath-103K`** (config `default`, split `train`).
- Rows: **103,022**. `download_size` 2,136,106,260 (~2.14 GB);
  `dataset_size` 4,999,766,760 (~5.0 GB).
- Columns: **`question`**, **`final_answer`**, **`difficulty`** (`float64`), **`topic`**,
  **`r1_solution_1`**, **`r1_solution_2`**, **`r1_solution_3`**.
- Note: there is **no `answer` and no `solution` column** — rename/adapt.
- Source: https://datasets-server.huggingface.co/info?dataset=zwhe99%2FDeepMath-103K

---

## 4. `HuggingFaceH4/MATH-500`

- The 500 problems live in the split named **`test`** (the only split). Config name `default`.
- Rows: **500**. `dataset_size` 400,274 B; builder `json`.
- Columns: **`problem`** (string), **`solution`** (string), **`answer`** (string),
  **`subject`** (string), **`level`** (int64), **`unique_id`** (string) — all six confirmed.
- Source: https://datasets-server.huggingface.co/info?dataset=HuggingFaceH4%2FMATH-500

---

## 5. `openai/gsm8k`

- Config name **`main`** (a second config `socratic` also exists).
- Splits/rows: `main` → `train` **7,473**, `test` **1,319**. `socratic` → same counts.
- Columns (both configs): **`question`**, **`answer`** (strings).
- Card inconsistency to be aware of: the human-readable "Data Splits" table in the card labels
  the columns `train`/`validation`, but the config YAML and the actual split name are
  **`test`**.
- `answer` contains `<<calc>>` annotations and ends with `#### <final>`.
- Sources: https://datasets-server.huggingface.co/info?dataset=openai%2Fgsm8k ·
  https://huggingface.co/datasets/openai/gsm8k/raw/main/README.md

---

## 6. MinervaMath / AIME 2025 / AIME 2024

| repo | exists | split | rows | columns |
|---|---|---|---|---|
| `math-ai/minervamath` | yes | `test` (only) | **272** | `question`, `answer` |
| `math-ai/aime25` | yes | `test` (only) | **30** | `problem`, `answer`, `id` (all string) |
| `Maxwell-Jia/AIME_2024` | yes | `train` (only) | **30** | `ID`, `Problem`, `Solution`, `Answer` |

- `Maxwell-Jia/AIME_2024` gotchas: column names are **capitalised**, and `Answer` is
  **int64**, not string. Its card says "Format: JSONL … Size: 30 records" while the configured
  data file is `aime_2024_problems.parquet` (viewer builder: `parquet`).
- Sources: https://datasets-server.huggingface.co/info?dataset=math-ai%2Fminervamath ·
  https://datasets-server.huggingface.co/info?dataset=math-ai%2Faime25 ·
  https://datasets-server.huggingface.co/info?dataset=Maxwell-Jia%2FAIME_2024 ·
  https://huggingface.co/datasets/math-ai/aime25/raw/main/README.md ·
  https://huggingface.co/datasets/math-ai/minervamath/raw/main/README.md ·
  https://huggingface.co/datasets/Maxwell-Jia/AIME_2024/raw/main/README.md

---

## 7. `load_dataset` syntax, `HF_ENDPOINT`, and `hf` CLI

### `load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")` — **correct**

- Official signature (datasets 4.8.4 reference):
  `datasets.load_dataset(path: str, name: Optional[str] = None, data_dir=None, data_files=None, split=None, cache_dir=None, features=None, …)`
- Doc for the parameter, verbatim: "**name** (`str`, *optional*) — Defining the name of the
  dataset configuration."
- Official doc example, verbatim:
  ```
  # Load a subset or dataset configuration (here 'sst2')
  >>> from datasets import load_dataset
  >>> ds = load_dataset('nyu-mll/glue', 'sst2', split='train')
  ```
- The dataset card itself: `ds = load_dataset("open-r1/OpenR1-Math-220k", "default")`.
- HF team member (loubnabnl) in discussion #2: "No need to specify a config_name you can do
  `ds = load_dataset("open-r1/OpenR1-Math-220k", "default")` or
  `ds = load_dataset("open-r1/OpenR1-Math-220k", "extended")`". Same thread contains a user
  hitting `TypeError: … got multiple values for keyword argument 'config_name'` →
  **pass the config positionally or as `name=`, not as `config_name=`**.
- Omitting the config (`load_dataset("open-r1/OpenR1-Math-220k", split="train")`) returns/loads
  **all** configs (same thread shows a `DatasetDict{default: 93733, extended: 131396}`), i.e.
  it downloads both.
- Sources: https://huggingface.co/docs/datasets/en/package_reference/loading_methods ·
  https://huggingface.co/api/datasets/open-r1/OpenR1-Math-220k/discussions/2

### `HF_ENDPOINT`

- Implemented and honoured by the code:
  - `huggingface_hub/constants.py`: `ENDPOINT = os.getenv("HF_ENDPOINT", _HF_DEFAULT_ENDPOINT).rstrip("/")`
  - `datasets/config.py`: `HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")`
    followed by `HUB_DATASETS_URL = HF_ENDPOINT + "/datasets/{repo_id}/resolve/{revision}/{path}"`
- Usage (as documented by the mirror itself, https://hf-mirror.com/):
  - Linux/macOS: `export HF_ENDPOINT=https://hf-mirror.com`
  - Windows PowerShell: `$env:HF_ENDPOINT = "https://hf-mirror.com"`
  - Or per-process: `HF_ENDPOINT=https://hf-mirror.com python your_script.py`
- Must be set **before import**: "All environment variables are read at import time of
  `huggingface_hub`. Any modification made afterwards will not be taken into account."
- **Caveat (verified):** `HF_ENDPOINT` is **not documented on HF's official environment
  variables reference page** (that page documents `HF_INFERENCE_ENDPOINT`, `HF_HOME`,
  `HF_HUB_CACHE`, … but not `HF_ENDPOINT`). It works because it is read in code.
  `hf-mirror.com` is a **third-party community mirror**, not an official Hugging Face service,
  and its own instructions still show the legacy `huggingface-cli` command.
- Sources: https://raw.githubusercontent.com/huggingface/huggingface_hub/main/src/huggingface_hub/constants.py ·
  https://raw.githubusercontent.com/huggingface/datasets/main/src/datasets/config.py ·
  https://raw.githubusercontent.com/huggingface/huggingface_hub/main/docs/source/en/package_reference/environment_variables.md ·
  https://hf-mirror.com/

### `hf` CLI replaced `huggingface-cli` — **yes**

- Blog, published July 25, 2025, verbatim: "the Hugging Face CLI has been officially renamed
  from `huggingface-cli` to `hf`!" and "The legacy `huggingface-cli` remains active and
  fully-functional. We're keeping it around to ease the transition. If you use any command
  from the legacy CLI, you'll see a warning that points you to the new CLI equivalent".
- `hf download <repo> --local-dir <dir>` is the current command. Docs, verbatim: "### Download
  to a local folder … This is useful to get a workflow closer to what git commands offer. You
  can do that using the `--local-dir` option." Example from the docs:
  `hf download adept/fuyu-8b model-00001-of-00002.safetensors --local-dir fuyu`
- **For a dataset you must add `--repo-type dataset`**, e.g. the docs example
  `hf download HuggingFaceH4/ultrachat_200k --repo-type dataset`.
- `hf download` also accepts `hf://` URIs (e.g. `hf download hf://datasets/open-r1/OpenR1-Math-220k`)
  and `--include`/`--exclude`/`--revision`/`--dry-run`.
- Sources: https://huggingface.co/blog/hf-cli ·
  https://raw.githubusercontent.com/huggingface/huggingface_hub/main/docs/source/en/guides/cli.md

---

## 8. Qwen3.5: math prompt and thinking budget

### Qwen3.5 exists and the 2B/4B-Base checkpoints exist

- `Qwen/Qwen3.5-2B-Base`, `Qwen/Qwen3.5-4B-Base`, `Qwen/Qwen3.5-0.8B-Base`,
  `Qwen/Qwen3.5-9B-Base`, and post-trained `Qwen3.5-0.8B/2B/4B/9B/27B/35B-A3B/122B-A10B/397B-A17B`
  (created Feb 2026). Citation block on the cards:
  `title = {{Qwen3.5}: Towards Native Multimodal Agents}, month = {February}, year = {2026}, url = {https://qwen.ai/blog?id=qwen3.5}`.
- Source: https://huggingface.co/api/models?author=Qwen&search=Qwen3.5 · https://huggingface.co/Qwen/Qwen3.5-4B-Base

### Is `"Please reason step by step, and put your final answer within \boxed{}."` a documented Qwen recommendation? — **Yes, verbatim**

- It appears in **"Best Practices" → "3. Standardize Output Format"** of the Qwen3.5
  **post-trained** cards (identical wording on `Qwen3.5-4B` and `Qwen3.5-2B`), verbatim:
  > "**Math Problems**: Include "Please reason step by step, and put your final answer within
  > \boxed{}." in the prompt."
- The same sentence appears in the older **Qwen3** cards (e.g. `Qwen/Qwen3-4B`).
- **Important nuance:** the cards say include it "**in the prompt**" — they do **not** say
  "system prompt". Whether it should be a system-role message specifically is
  **UNVERIFIED** (an open community question exists at
  https://github.com/QwenLM/Qwen3/discussions/1395 with no Qwen answer in the thread).
- **The `-Base` cards contain no Best Practices / prompt section at all.** The
  `Qwen3.5-4B-Base` README ends at "Model Overview" + "Citation"; the boxed-prompt advice is
  on the post-trained models' cards. So for fine-tuning Qwen3.5-2B/4B-**Base** there is no
  official math system prompt in the model card — the boxed instruction is the *benchmark
  prompting* recommendation.
- Corroboration that this is the right instruction for this exact data: OpenR1-Math-220k was
  itself generated with that instruction prepended ("we … prepend the following instruction to
  the user prompt: `"Please reason step by step, and put your final answer within \boxed{}."`").
- Sources: https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/README.md ·
  https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/README.md ·
  https://huggingface.co/Qwen/Qwen3.5-4B-Base/raw/main/README.md ·
  https://huggingface.co/Qwen/Qwen3-4B/raw/main/README.md

### Recommended max output ("thinking") budget for math

- **Qwen3.5 (current), verbatim:** "**Adequate Output Length**: We recommend using an output
  length of 32,768 tokens for most queries. For benchmarking on highly complex problems, such
  as those found in math and programming competitions, we suggest setting the max output
  length to **81,920 tokens**."
- **Qwen3 (previous generation, e.g. `Qwen/Qwen3-4B`), verbatim:** same sentence but "…
  we suggest setting the max output length to **38,912 tokens**."
  → so 38,912 is the **Qwen3** number, not the Qwen3.5 number. Don't mix them.
- Code examples on both Qwen3.5-2B and Qwen3.5-4B cards use `max_tokens=81920` for
  thinking-mode calls.
- Sources: https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/README.md ·
  https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/README.md ·
  https://huggingface.co/Qwen/Qwen3-4B/raw/main/README.md

### Other Qwen3.5 facts that will bite the fine-tuning code

- Qwen3.5 is a **vision-language** architecture: `pipeline_tag: image-text-to-text`,
  transformers model type `qwen3_5` ("Gated DeltaNet → FFN … Gated Attention"; MoE variants use
  `qwen3_5_moe`). Context length **262,144** natively; vocab/embeddings **248,320**.
  The cards instruct installing transformers from main:
  `pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"`.
- `Qwen3.5-2B` **does not think by default**: "Qwen3.5-2B operates in non-thinking mode by
  default." And a warning: "In thinking mode, we have observed that when using the recommended
  sampling parameters, Qwen3.5-2B is more prone to entering thinking loops".
- Qwen3.5 dropped the Qwen3 style soft switch: "Qwen3.5 does not officially support the soft
  switch of Qwen3, i.e., `/think` and `/nothink`."
- The **`-Base` repos have no separate `chat_template.jinja`** (the post-trained `Qwen3.5-4B`
  does), **but `Qwen3.5-4B-Base/tokenizer_config.json` does embed a full `chat_template`**.
  That template:
  - reads `message.reasoning_content` when present, otherwise splits the assistant content on
    `</think>`/`<think>` — i.e. `reasoning_content` is a **chat-template** concept, not a
    dataset column;
  - with `enable_thinking=False` emits an empty `<think>\n\n</think>\n\n`;
  - has `<think>`/`</think>` tokens at ids 248068/248069 and special tokens
    `<|im_start|>` (248045) / `<|im_end|>` (248046);
  - `model_max_length: 262144`, `eos_token`/`pad_token` = `<|endoftext|>`, `bos_token: null`,
    `tokenizer_class: Qwen2Tokenizer`.
- The Base card notes: "the control tokens, e.g., `<|im_start|>` and `<|im_end|>` were trained
  to allow efficient LoRA-style PEFT with the official chat template".
- Sources: https://huggingface.co/Qwen/Qwen3.5-4B-Base/raw/main/README.md ·
  https://huggingface.co/Qwen/Qwen3.5-4B-Base/raw/main/tokenizer_config.json ·
  https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/README.md

---

## Anything still UNVERIFIED

1. Any HF revision/derivative of OpenR1-Math-220k where `generations` is a list of **dicts**
   with a `reasoning_content` key — no evidence found; contrary to the observed schema.
2. Whether `correctness_math_verify` is ever `null` (it is non-null in inspected rows;
   `correctness_llama` and `finish_reasons` demonstrably are `null` in some rows).
3. The complete distinct set of `from` values in Bespoke-Stratos-17k `conversations`
   (only `user` / `assistant` observed, in row 0).
4. Whether Qwen intends the `\boxed{}` instruction as a **system** message specifically; the
   cards only say "in the prompt".
5. Any official Qwen system-prompt template string for math beyond that one sentence.
