"""Compatibility entry point for BPE text generation.

The maintained implementation lives in ``generate_bpe_fixed.py``. Keeping
this thin wrapper means both historical commands continue to work.
"""

from generate_bpe_fixed import main


if __name__ == "__main__":
    main()
