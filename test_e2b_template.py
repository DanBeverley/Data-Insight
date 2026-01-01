"""
Test script to verify E2B template has pre-installed packages.
Run this independently to check if the template is correctly built.
"""

from dotenv import load_dotenv

load_dotenv()

from e2b_code_interpreter import Sandbox

TEMPLATE_NAME = "data-insight-sandbox"


def test_template():
    print(f"Creating sandbox with template: {TEMPLATE_NAME}")

    try:
        sandbox = Sandbox.create(template=TEMPLATE_NAME, timeout=300)
        print("Sandbox created successfully!")

        # Test statsmodels
        print("\n--- Testing statsmodels ---")
        result = sandbox.run_code("import statsmodels; print(f'statsmodels version: {statsmodels.__version__}')")
        print(f"Result: {result}")
        if result:
            print(f"  text: {getattr(result, 'text', 'N/A')}")
            print(f"  logs: {getattr(result, 'logs', 'N/A')}")
            print(f"  error: {getattr(result, 'error', 'N/A')}")

        # Test flaml
        print("\n--- Testing flaml ---")
        result = sandbox.run_code("import flaml; print(f'flaml version: {flaml.__version__}')")
        print(f"Result: {result}")
        if result:
            print(f"  text: {getattr(result, 'text', 'N/A')}")
            print(f"  logs: {getattr(result, 'logs', 'N/A')}")
            print(f"  error: {getattr(result, 'error', 'N/A')}")

        # Test which Python is being used
        print("\n--- Python path ---")
        result = sandbox.run_code("import sys; print(sys.executable)")
        print(f"Result: {result}")
        if result:
            print(f"  text: {getattr(result, 'text', 'N/A')}")
            print(f"  logs: {getattr(result, 'logs', 'N/A')}")

        sandbox.close()
        print("\nSandbox closed.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_template()
