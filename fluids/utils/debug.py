from __future__ import print_function
import sys


def fluids_print(s, **kwargs):
    print("[FLUIDS] " + str(s), **kwargs)


def fluids_assert(cond, em=None):
    if not cond:
        fluids_print("Error: " + em)
        exit(1)
