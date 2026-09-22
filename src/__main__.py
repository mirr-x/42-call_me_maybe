"""Compatibility entry point for the required ``python -m src`` command."""

import os

# The bundled SDK targets one GPU. Hide additional Kaggle GPUs before torch is
# imported so Accelerate cannot dispatch the model across incompatible devices.
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from call_me_maybe.__main__ import main


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time) / 60:.2f} minutes")
