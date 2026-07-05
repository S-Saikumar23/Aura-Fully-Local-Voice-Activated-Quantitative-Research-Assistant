"""Setup script to create the aura_finance database and enable pgvector."""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

DB_PASSWORD = "sai123"

def main():
    # Step 1: Connect to default 'postgres' database to create aura_finance
    print("[1/3] Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        host="localhost", port=5432,
        user="postgres", password=DB_PASSWORD,
        dbname="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Check if database exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'aura_finance'")
    exists = cur.fetchone()

    if exists:
        print("   Database 'aura_finance' already exists!")
    else:
        cur.execute("CREATE DATABASE aura_finance")
        print("   Database 'aura_finance' created successfully!")

    cur.close()
    conn.close()

    # Step 2: Connect to aura_finance and enable pgvector
    print("[2/3] Enabling pgvector extension...")
    conn2 = psycopg2.connect(
        host="localhost", port=5432,
        user="postgres", password=DB_PASSWORD,
        dbname="aura_finance"
    )
    conn2.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur2 = conn2.cursor()
    cur2.execute("CREATE EXTENSION IF NOT EXISTS vector")
    print("   pgvector extension enabled!")
    cur2.close()
    conn2.close()

    # Step 3: Update .env with correct password
    print("[3/3] Updating .env with correct DATABASE_URL...")
    env_path = r"c:\Users\Lenovo\OneDrive\Documents\Projects\Aura\Aura-main\.env"
    with open(env_path, "r") as f:
        content = f.read()

    content = content.replace(
        "DATABASE_URL=postgresql://postgres:postgres@localhost:5432/aura_finance",
        "DATABASE_URL=postgresql://postgres:sai123@localhost:5432/aura_finance"
    )

    with open(env_path, "w") as f:
        f.write(content)
    print("   .env updated with correct password!")

    print("\nDatabase setup complete!")


if __name__ == "__main__":
    main()
