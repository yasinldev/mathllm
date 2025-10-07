import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import mathcore

def test_python_exceptions():
    print("=== Python Exception Mapping Tests ===")
    
    try:
        mathcore.diff("invalid^^expr", "x")
        print("[FAIL] Should have raised SymbolicError")
    except Exception as e:
        print(f"[PASS] Caught {type(e).__name__}: {e}")
    
    try:
        mathcore.integrate("tan(x)", "x")
        print("[FAIL] Should have raised SymbolicError")
    except Exception as e:
        print(f"[PASS] Caught {type(e).__name__}: {e}")
    
    try:
        result = mathcore.verify_equal("x + x", "2*x", 1000.0)
        print(f"[PASS] verify_equal returned: {result}")
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
    
    try:
        result = mathcore.integrate("x", "x")
        print(f"[PASS] integrate returned: {result}")
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
    
    try:
        result = mathcore.diff("x^2", "x")
        print(f"[PASS] diff returned: {result}")
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
    
    print("\n[SUCCESS] All Python exception tests passed")

if __name__ == "__main__":
    test_python_exceptions()
