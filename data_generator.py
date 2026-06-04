"""
data_generator.py
─────────────────────────────────────────────────────────────────────────────
Simulates live business transactions and streams them into SQLite at a rate
of 1-2 records per second.

Run:  python data_generator.py
Stop: Ctrl-C
"""

import sqlite3
import random
import time
import uuid
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────

DB_PATH = "business_data.db"

REGIONS = ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East & Africa"]

CATEGORIES = {
    "Electronics":   (120, 1_800),
    "Clothing":      (25,  350),
    "Home & Garden": (40,  900),
    "Sports":        (30,  600),
    "Books":         (10,  120),
    "Food & Bev":    (15,  200),
    "Toys":          (20,  280),
    "Automotive":    (80,  2_500),
}

# ── Database helpers ────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    """Create the transactions table if it does not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            order_id         TEXT    NOT NULL UNIQUE,
            customer_region  TEXT    NOT NULL,
            product_category TEXT    NOT NULL,
            sale_amount      REAL    NOT NULL,
            quantity         INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ts ON transactions (timestamp)"
    )
    conn.commit()


def insert_transaction(conn: sqlite3.Connection, record: dict) -> None:
    conn.execute(
        """
        INSERT INTO transactions
            (timestamp, order_id, customer_region, product_category, sale_amount, quantity)
        VALUES
            (:timestamp, :order_id, :customer_region, :product_category, :sale_amount, :quantity)
        """,
        record,
    )
    conn.commit()


# ── Transaction factory ─────────────────────────────────────────────────────

def generate_transaction() -> dict:
    category = random.choice(list(CATEGORIES.keys()))
    low, high = CATEGORIES[category]
    quantity = random.randint(1, 10)
    unit_price = round(random.uniform(low, high), 2)

    # Introduce realistic spikes: 5 % of orders are bulk / premium
    if random.random() < 0.05:
        quantity = random.randint(11, 50)
        unit_price = round(unit_price * random.uniform(0.7, 0.9), 2)  # bulk discount

    return {
        "timestamp":        datetime.utcnow().isoformat(sep=" ", timespec="seconds"),
        "order_id":         str(uuid.uuid4()),
        "customer_region":  random.choice(REGIONS),
        "product_category": category,
        "sale_amount":      round(unit_price * quantity, 2),
        "quantity":         quantity,
    }


# ── Main loop ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[data_generator] Connecting to '{DB_PATH}' …")
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    init_db(conn)

    total = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    print(f"[data_generator] Existing rows: {total:,}")
    print("[data_generator] Streaming transactions — press Ctrl-C to stop.\n")

    try:
        while True:
            # Generate 1 or 2 transactions per tick
            batch_size = random.randint(1, 2)
            for _ in range(batch_size):
                tx = generate_transaction()
                insert_transaction(conn, tx)
                print(
                    f"  ✚  {tx['timestamp']}  |  {tx['customer_region']:<25}"
                    f"|  {tx['product_category']:<14}  |  ${tx['sale_amount']:>10,.2f}"
                    f"  qty={tx['quantity']}"
                )

            # Sleep 0.5–1.0 s  →  roughly 1-2 tx/s on average
            time.sleep(random.uniform(0.5, 1.0))

    except KeyboardInterrupt:
        print("\n[data_generator] Stopped by user.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
