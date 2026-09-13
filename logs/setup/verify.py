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
