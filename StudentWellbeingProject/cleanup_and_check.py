import os
import sys
import re
import shutil

# 1. Clean up all __pycache__ folders and .pyc files
print("Cleaning up __pycache__ folders and .pyc files...")
for root, dirs, files in os.walk('.', topdown=False):
    for name in dirs:
        if name == '__pycache__':
            pycache_path = os.path.join(root, name)
            print(f"Deleting {pycache_path}")
            try:
                shutil.rmtree(pycache_path)
            except Exception as e:
                print(f"Error deleting {pycache_path}: {e}")
    for name in files:
        if name.endswith('.pyc'):
            pyc_path = os.path.join(root, name)
            print(f"Deleting {pyc_path}")
            try:
                os.remove(pyc_path)
            except Exception as e:
                print(f"Error deleting {pyc_path}: {e}")

# 2. Check for file name conflicts with standard library modules
conflicts = ['multiprocessing.py', 'flask.py', 'subprocess.py', 'threading.py', 'os.py', 'json.py']
found_conflicts = []
for fname in os.listdir('.'):
    if fname in conflicts:
        found_conflicts.append(fname)
if found_conflicts:
    print("\nWARNING: The following files may conflict with standard library modules:")
    for fname in found_conflicts:
        print(f" - {fname}")
    print("Consider renaming these files.")
else:
    print("\nNo standard library filename conflicts detected in the current directory.")

# 3. Print a reminder to run from terminal
print("\nTo run your app, use the terminal and execute:")
print("python recommendation_app.py --with-fake-scraper")

# 4. Check for hidden imports of recommendation_app.py
print("\nChecking for hidden imports of recommendation_app.py in other files...")
for root, dirs, files in os.walk('.'):
    for fname in files:
        if fname.endswith('.py') and fname != 'recommendation_app.py':
            try:
                with open(os.path.join(root, fname), encoding='utf-8') as f:
                    content = f.read()
                    if re.search(r'(import|from)\s+recommendation_app', content):
                        print(f"WARNING: {fname} imports recommendation_app.py. This may cause issues with multiprocessing.")
            except Exception as e:
                print(f"Could not read {fname}: {e}")
print("\nCleanup and checks complete.")
