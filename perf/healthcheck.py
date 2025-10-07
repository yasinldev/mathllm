#!/usr/bin/env python3
import requests
import time
import sys
from typing import Optional

def check_health(api_base: str, timeout: int = 120, interval: int = 2) -> bool:
    url = f"{api_base}/models"
    start_time = time.time()
    
    print(f"Healthcheck: Polling {url}...")
    print(f"Timeout: {timeout}s, Interval: {interval}s\n")
    
    attempt = 0
    while time.time() - start_time < timeout:
        attempt += 1
        elapsed = time.time() - start_time
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('data', [])
                model_ids = [m.get('id', 'unknown') for m in models]
                
                print(f"✓ Server is ready! (attempt {attempt}, elapsed {elapsed:.1f}s)")
                print(f"  Available models: {model_ids}")
                return True
            else:
                print(f"  Attempt {attempt}: HTTP {response.status_code} (elapsed {elapsed:.1f}s)")
        except requests.exceptions.ConnectionError:
            print(f"  Attempt {attempt}: Connection refused (elapsed {elapsed:.1f}s)")
        except requests.exceptions.Timeout:
            print(f"  Attempt {attempt}: Timeout (elapsed {elapsed:.1f}s)")
        except Exception as e:
            print(f"  Attempt {attempt}: {type(e).__name__}: {e} (elapsed {elapsed:.1f}s)")
        
        time.sleep(interval)
    
    print(f"\n✗ Healthcheck failed after {timeout}s")
    return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python healthcheck.py <api_base> [timeout]")
        print("Example: python healthcheck.py http://localhost:8009/v1 120")
        sys.exit(1)
    
    api_base = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    
    success = check_health(api_base, timeout)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
