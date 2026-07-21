#!/usr/bin/env bash
# Run first on Shannon (`make verify-gpu`) to confirm the container can see
# the RTX 5090 before spending time on data download / training.
set -euo pipefail

echo "=== nvidia-smi (from inside container) ==="
nvidia-smi

echo
echo "=== torch CUDA check ==="
python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
    x = torch.randn(4096, 4096, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    print('matmul on GPU ok, sample value:', y[0, 0].item())
else:
    raise SystemExit('CUDA not available inside the container -- check --gpus all and the CUDA/driver version match in docker/Dockerfile')
"
