"""Evaluate checkpoints over multiple starts and explicit random seeds.

Install the package first: pip install -e .
See python scripts/eval_multiseed.py --help for the protocol options.
"""
from equitraj.multiseed import main

if __name__ == "__main__":
    main()
