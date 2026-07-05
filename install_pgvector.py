"""Install pgvector files into PostgreSQL 16 directory (requires admin)."""
import shutil
import os
import zipfile
import tempfile

PG_DIR = r"C:\Program Files\PostgreSQL\16"
ZIP_PATH = os.path.join(tempfile.gettempdir(), "pgvector_pg16.zip")
EXTRACT_DIR = os.path.join(tempfile.gettempdir(), "pgvector_extract")

def main():
    # Re-download if needed
    if not os.path.exists(ZIP_PATH):
        import urllib.request
        url = "https://github.com/andreiramani/pgvector_pgsql_windows/releases/download/0.8.3_16.14/vector.v0.8.3-pg16.zip"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, ZIP_PATH)

    # Extract
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_DIR)
    
    # Copy files
    copied = 0
    for root, dirs, files in os.walk(EXTRACT_DIR):
        for f in files:
            src = os.path.join(root, f)
            if f == "vector.control" or (f.endswith(".sql") and "vector" in f.lower()):
                dst = os.path.join(PG_DIR, "share", "extension", f)
                shutil.copy2(src, dst)
                print(f"Copied {f} -> share/extension/")
                copied += 1
            elif f.endswith(".dll"):
                dst = os.path.join(PG_DIR, "lib", f)
                shutil.copy2(src, dst)
                print(f"Copied {f} -> lib/")
                copied += 1

    # Cleanup
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    shutil.rmtree(EXTRACT_DIR)
    
    print(f"\nDone! {copied} files installed.")

if __name__ == "__main__":
    main()
