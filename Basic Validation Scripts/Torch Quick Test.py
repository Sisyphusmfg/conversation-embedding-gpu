# test_torch_import.py
import sys
print("Python path:", sys.executable)
print("Python paths:", sys.path)

try:
    import torch
    print("Torch imported successfully")
    print("Torch file location:", torch.__file__)
except ImportError as e:
    print("Import error:", e)