#!/usr/bin/env python3
"""
Quick verification script for MultiMin package installation
"""

import sys


def main():
    print("=" * 60)
    print("MultiMin Package Verification")
    print("=" * 60)

    try:
        import multimin as mn

        print("✓ Package imported successfully")

        # Check version
        print(f"✓ Version: {mn.__version__}")

        # Check authors
        print(f"✓ Authors: {mn.__author__}")

        # Check email
        print(f"✓ Contact: {mn.__email__}")

        # Check base class
        # Assuming MultiMinBase corresponds to the deprecated MultiNEAsBase or similar
        # If MultiNEAsBase was renamed to MultiMinBase, use that.
        # Based on previous edits, I know MultiNEAsBase was renamed to MultiMinBase in __init__.py
        if hasattr(mn, "MultiMinBase"):
            obj = mn.MultiMinBase()
            print(f"✓ Base class instantiated: {type(obj).__name__}")
        else:
            # Fallback if MultiMinBase is not exposed or named differently
            # Checking for ComposedMultiVariateNormal as a core class
            obj = mn.ComposedMultiVariateNormal()
            print(f"✓ Core class instantiated: {type(obj).__name__}")

        # Check string representation
        str_repr = str(obj)
        print(f"✓ String representation works")

        print("\n" + "=" * 60)
        print("All checks passed! MultiMin is ready to use.")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
