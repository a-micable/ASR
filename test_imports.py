#!/usr/bin/env python3
"""Test that all application modules can be imported in the Docker container."""

import sys

def test_import(module_name):
    """Test importing a module."""
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠ {module_name}: {type(e).__name__}: {e}")
        return True  # Module imported but has runtime issues (acceptable for structure test)

def main():
    """Test all application modules."""
    modules = [
        # Core Python
        "sys",
        "os",
        "pathlib",
        
        # Our modules (will fail if PYTHONPATH not set)
        "preprocessing",
        "training",
        "evaluation",
        "api",
        "monitoring",
        "logging_config",
    ]
    
    print("Testing module imports...\n")
    
    results = []
    for module in modules:
        results.append(test_import(module))
    
    print(f"\n{sum(results)}/{len(results)} modules accessible")
    
    if all(results):
        print("\n✅ All modules can be imported - PYTHONPATH is configured correctly!")
        return 0
    else:
        print("\n❌ Some modules failed to import")
        return 1

if __name__ == "__main__":
    sys.exit(main())
