# ============================================================
#  DSAI4207 - build .venv  (PyTorch CUDA 12.8 + FLA + Unsloth)
#
#  Usage (works in Windows PowerShell 5.1):
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
#
#  NOTE: this file is intentionally ASCII-only. Windows PowerShell 5.1
#  decodes .ps1 files as ANSI unless they carry a BOM, which corrupts
#  non-ASCII literals. Keep it ASCII.
#
#  Output of each step goes to logs\setup\*.log
# ============================================================
$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'

# Project root: hard-coded for this project so we never install into the wrong venv
$Root = 'E:\DSAI\DSAI4207-new'
if (-not (Test-Path (Join-Path $Root 'project_plan.md'))) {
    $Root = (Get-Location).Path
}
$Root = (Resolve-Path $Root).Path
Write-Host "[root] $Root"

$Venv   = Join-Path $Root '.venv'
$Py     = Join-Path $Venv 'Scripts\python.exe'
$LogDir = Join-Path $Root 'logs\setup'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Keep pip temp inside the project (avoids system temp permission problems)
$env:TEMP = Join-Path $Root '.piptmp'
$env:TMP  = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'

$TORCH_INDEX = 'https://download.pytorch.org/whl/cu128'

function Say($msg) { Write-Host $msg -ForegroundColor Cyan }

# ---------- 0. venv ----------
if (-not (Test-Path $Py)) {
    Say '[0] creating .venv ...'
    python -m venv $Venv
}
if (-not (Test-Path $Py)) {
    Write-Host '[0] FAILED: .venv\Scripts\python.exe not found' -ForegroundColor Red
    exit 1
}
Write-Host "[0] venv python: $Venv"
& $Py --version
& $Py -m pip install --upgrade pip setuptools wheel 2>&1 | Out-Null

# ---------- 1. PyTorch CUDA 12.8 ----------
Say "`n=== [1/8] PyTorch cu128 ==="
& $Py -m pip install --timeout 120 --retries 5 --index-url $TORCH_INDEX `
    torch torchvision 2>&1 | Tee-Object -FilePath (Join-Path $LogDir '01_torch.log')
Write-Host "[1/8] exit=$LASTEXITCODE"

# ---------- 2. core deps ----------
Say "`n=== [2/8] core deps (transformers/peft/trl/datasets) ==="
& $Py -m pip install --timeout 120 --retries 5 `
    'transformers>=5.0.0' 'accelerate>=1.0.0' 'datasets>=3.0.0' `
    'peft>=0.14.0' 'trl>=0.13.0' 'huggingface_hub>=0.30.0' `
    safetensors sentencepiece protobuf einops ninja packaging pillow `
    2>&1 | Tee-Object -FilePath (Join-Path $LogDir '02_core.log')
Write-Host "[2/8] exit=$LASTEXITCODE"

# ---------- 3. FLA (flash-linear-attention) ----------
# transformers is_flash_linear_attention_available() needs:
#   (a) import fla works  (b) fla.__version__ >= 0.2.2  (c) torch.cuda available
# Windows has no official triton wheel -> install the triton-windows fork FIRST.
# Since fla 0.5, bare "flash-linear-attention" does not pull torch/triton,
# so the [cuda] extra is required for "import fla" to succeed.
Say "`n=== [3/8] FLA (triton-windows + flash-linear-attention[cuda]) ==="
& $Py -m pip install --timeout 120 --retries 5 triton-windows `
    2>&1 | Tee-Object -FilePath (Join-Path $LogDir '03a_triton.log')
Write-Host "[3a/8] triton-windows exit=$LASTEXITCODE"

& $Py -m pip install --timeout 180 --retries 5 --no-build-isolation 'flash-linear-attention[cuda]' `
    2>&1 | Tee-Object -FilePath (Join-Path $LogDir '03b_fla.log')
$flaRc = $LASTEXITCODE
Write-Host "[3b/8] fla[cuda] exit=$flaRc"
if ($flaRc -ne 0) {
    Write-Host '  [3b] retrying without the extra ...' -ForegroundColor Yellow
    & $Py -m pip install --timeout 180 --retries 5 --no-build-isolation `
        flash-linear-attention fla-core `
        2>&1 | Tee-Object -FilePath (Join-Path $LogDir '03c_fla_retry.log')
    Write-Host "[3c/8] fla plain exit=$LASTEXITCODE"
}

# ---------- 4. Unsloth ----------
Say "`n=== [4/8] Unsloth ==="
& $Py -m pip install --timeout 120 --retries 5 unsloth unsloth_zoo `
    2>&1 | Tee-Object -FilePath (Join-Path $LogDir '04_unsloth.log')
Write-Host "[4/8] exit=$LASTEXITCODE"

# ---------- 5. bitsandbytes ----------
Say "`n=== [5/8] bitsandbytes ==="
& $Py -m pip install --timeout 120 --retries 5 bitsandbytes `
    2>&1 | Tee-Object -FilePath (Join-Path $LogDir '05_bnb.log')
Write-Host "[5/8] exit=$LASTEXITCODE"

# ---------- 6. eval + plotting ----------
Say "`n=== [6/8] eval + plotting ==="
& $Py -m pip install --timeout 120 --retries 5 `
    'math-verify[antlr4_13_2]' lmstudio pandas matplotlib scipy openpyxl `
    2>&1 | Tee-Object -FilePath (Join-Path $LogDir '06_eval.log')
Write-Host "[6/8] exit=$LASTEXITCODE"

# ---------- 7. jupyter kernel ----------
Say "`n=== [7/8] ipykernel ==="
& $Py -m pip install --timeout 120 --retries 5 ipykernel nbformat nbclient jupyterlab `
    2>&1 | Tee-Object -FilePath (Join-Path $LogDir '07_kernel.log')
& $Py -m ipykernel install --user --name dsai4207 --display-name "Python (.venv DSAI4207)" `
    2>&1 | Tee-Object -FilePath (Join-Path $LogDir '07_kernel.log') -Append
Write-Host "[7/8] exit=$LASTEXITCODE"

# ---------- 8. verify ----------
Say "`n=== [8/8] verify ==="
& $Py -m pip freeze 2>&1 | Out-File -Encoding utf8 (Join-Path $Root 'requirements.txt')

$verify = @'
import importlib, sys
print("python :", sys.version.split()[0])
for n in ["torch","torchvision","transformers","accelerate","datasets","peft","trl",
          "triton","fla","unsloth","bitsandbytes","math_verify","lmstudio","scipy",
          "pandas","matplotlib"]:
    try:
        m = importlib.import_module(n)
        print("%-14s: %s" % (n, getattr(m, "__version__", "ok")))
    except Exception as e:
        print("%-14s: MISSING (%s: %s)" % (n, type(e).__name__, str(e)[:90]))
try:
    import torch
    print("torch.cuda ver :", torch.version.cuda)
    print("cuda available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("gpu            :", p.name, round(p.total_memory/1024**3,1), "GB")
        print("capability     :", torch.cuda.get_device_capability(0))
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        print("bf16 matmul ok :", (x @ x).dtype)
except Exception as e:
    print("torch check failed:", type(e).__name__, str(e)[:200])
try:
    from transformers.utils.import_utils import is_flash_linear_attention_available as f
    print("transformers FLA available:", f())
except Exception as e:
    print("FLA check failed:", type(e).__name__, str(e)[:200])
'@
$verify | Out-File -Encoding utf8 (Join-Path $LogDir 'verify.py')
& $Py (Join-Path $LogDir 'verify.py') 2>&1 | Tee-Object -FilePath (Join-Path $LogDir '08_verify.log')

Write-Host "`nDONE. Logs: $LogDir" -ForegroundColor Cyan
