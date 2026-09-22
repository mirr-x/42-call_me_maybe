"""Compatibility entry point for the required ``python -m src`` command."""

from call_me_maybe.__main__ import main


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time) / 60:.2f} minutes")

