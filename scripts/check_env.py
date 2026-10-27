import importlib
import os
import shutil
import subprocess
import sys


def section(title):
    print(f"\n=== {title} " + "=" * (60 - len(title)))


def ok(msg):
    print(f"  [OK]   {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")
    global failures
    failures += 1


failures = 0

section("Driver / GPU")
smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,compute_cap",
     "--format=csv,noheader"],
    capture_output=True, text=True,
)
print("  " + smi.stdout.strip())

section("PyTorch")
import torch  # noqa: E402

print(f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}")
if torch.cuda.is_available():
    ok(f"CUDA available: {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"  free VRAM: {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = a @ a
    torch.cuda.synchronize()
    ok(f"BF16 matmul ran, result norm {b.float().norm().item():.3e}")
else:
    fail("torch.cuda.is_available() is False -> driver / CUDA build mismatch")

section("Python packages")
for name in ["transformers", "accelerate", "datasets", "modelopt", "vllm", "lm_eval", "flashinfer"]:
    try:
        mod = importlib.import_module(name)
        ok(f"{name} {getattr(mod, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        fail(f"{name}: {type(e).__name__}: {e}")

section("ModelOpt: fake-quantize a tiny MLP")
try:
    import modelopt.torch.quantization as mtq

    model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 64)).cuda()
    x = torch.randn(32, 256, device="cuda")
    ref = model(x)

    def forward_loop(m):
        for _ in range(4):
            m(torch.randn(32, 256, device="cuda"))

    model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)
    err = (model(x) - ref).abs().mean() / ref.abs().mean()
    ok(f"INT8 fake-quant works, relative error {err.item():.4f} (expect ~1e-2)")
except Exception as e:  # noqa: BLE001
    fail(f"modelopt quantize: {type(e).__name__}: {e}")

section("ModelOpt CUDA extensions")
# On build failure ModelOpt only warns and falls back to slower kernels, so check explicitly
try:
    from modelopt.torch.quantization import extensions as ext

    for getter in ["get_cuda_ext", "get_cuda_ext_fp8", "get_cuda_ext_mx"]:
        try:
            (ok if getattr(ext, getter)() is not None else fail)(getter)
        except Exception as e:  # noqa: BLE001
            fail(f"{getter}: {type(e).__name__}: {e}")
except Exception as e:  # noqa: BLE001
    fail(f"import modelopt extensions: {e}")

section("Environment variables")
for var in ["LAB_DATA", "HF_HOME", "UV_CACHE_DIR", "VLLM_CACHE_ROOT", "CUDA_HOME"]:
    (ok if os.environ.get(var) else fail)(f"{var} set")
if os.environ.get("LAB_DATA"):
    print(f"  disk free on LAB_DATA: {shutil.disk_usage(os.environ['LAB_DATA']).free / 2**30:.0f} GiB")

section("Result")
print(f"  python {sys.version.split()[0]}  ->  {'ALL GOOD' if failures == 0 else f'{failures} FAILURE(S)'}")
sys.exit(1 if failures else 0)
