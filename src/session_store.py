"""SQLite persistence for KAYAN customer sessions and truthful end-of-day snapshots."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4
from zoneinfo import ZoneInfo

from .communication_contract import SCHEMA_VERSION, validate_event


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CustomerSessionStore:
    """Keep customer state isolated without treating browsing as a sale."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.initialize_business_day(datetime.now(ZoneInfo("Africa/Cairo")).date().isoformat())

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                PRAGMA journal_mode=DELETE;
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    conversation_json TEXT NOT NULL DEFAULT '[]',
                    agent_state_json TEXT NOT NULL DEFAULT '{}',
                    removed INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS customer_activity (
                    activity_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    business_day TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
                );
                CREATE INDEX IF NOT EXISTS customer_activity_day_idx
                    ON customer_activity(business_day, event_type);
                CREATE TABLE IF NOT EXISTS daily_snapshots (
                    business_day TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS business_days (
                    day_id TEXT PRIMARY KEY,
                    day_number INTEGER NOT NULL UNIQUE,
                    business_date TEXT NOT NULL,
                    timezone TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL CHECK(status IN ('active','closed')),
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    reports_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_business_day_idx
                    ON business_days(status) WHERE status = 'active';
                CREATE TABLE IF NOT EXISTS business_day_customers (
                    day_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    linked_at TEXT NOT NULL,
                    PRIMARY KEY(day_id, customer_id),
                    FOREIGN KEY(day_id) REFERENCES business_days(day_id),
                    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
                );
                CREATE INDEX IF NOT EXISTS business_day_customers_customer_idx
                    ON business_day_customers(customer_id, day_id);
                CREATE TABLE IF NOT EXISTS conversation_archives (
                    archive_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    message_count INTEGER NOT NULL,
                    conversation_json TEXT NOT NULL,
                    agent_state_json TEXT NOT NULL,
                    archived_at TEXT NOT NULL,
                    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
                );
                CREATE INDEX IF NOT EXISTS conversation_archives_customer_idx
                    ON conversation_archives(customer_id, archived_at);
                CREATE TABLE IF NOT EXISTS app_settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS demo_outfits (
                    outfit_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS demo_outfit_items (
                    outfit_id TEXT NOT NULL,
                    slot_key TEXT NOT NULL,
                    variant_id TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY(outfit_id, slot_key)
                );
                CREATE TABLE IF NOT EXISTS demo_mock_purchase_requests (
                    request_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    outfit_id TEXT NOT NULL,
                    total_piasters INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    items_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    confirmation_key TEXT,
                    business_day TEXT
                );
                CREATE TABLE IF NOT EXISTS customer_creation_requests (
                    request_key TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS communication_events (
                    event_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    source_agent TEXT NOT NULL,
                    target_agent TEXT,
                    timestamp TEXT NOT NULL,
                    business_day_id TEXT NOT NULL,
                    customer_id TEXT,
                    interaction_key TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS communication_event_dedupe_idx
                    ON communication_events(event_type,business_day_id,customer_id,interaction_key)
                    WHERE interaction_key IS NOT NULL;
                CREATE INDEX IF NOT EXISTS communication_event_day_type_idx
                    ON communication_events(business_day_id,event_type,timestamp);
                CREATE TABLE IF NOT EXISTS communication_deliveries (
                    event_id TEXT NOT NULL,
                    consumer_agent TEXT NOT NULL,
                    delivered_at TEXT NOT NULL,
                    PRIMARY KEY(event_id,consumer_agent),
                    FOREIGN KEY(event_id) REFERENCES communication_events(event_id)
                );
                CREATE TABLE IF NOT EXISTS ceo_tasks (
                    task_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    assigned_date TEXT NOT NULL,
                    assigned_by TEXT NOT NULL CHECK(assigned_by='CEO'),
                    assigned_to TEXT NOT NULL CHECK(assigned_to='Sales'),
                    is_done INTEGER NOT NULL DEFAULT 0 CHECK(is_done IN (0,1)),
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ceo_tasks_date_idx ON ceo_tasks(assigned_date,created_at);
                CREATE TABLE IF NOT EXISTS communication_reports (
                    business_day_id TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    report_json TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    PRIMARY KEY(business_day_id,agent,schema_version)
                );
            """)
            # Additive migrations preserve existing customer and purchase history.
            customer_columns = {row[1] for row in conn.execute("PRAGMA table_info(customers)")}
            if "removed" not in customer_columns:
                conn.execute("ALTER TABLE customers ADD COLUMN removed INTEGER NOT NULL DEFAULT 0")
            item_columns = {row[1] for row in conn.execute("PRAGMA table_info(demo_outfit_items)")}
            if "quantity" not in item_columns:
                conn.execute("ALTER TABLE demo_outfit_items ADD COLUMN quantity INTEGER NOT NULL DEFAULT 1")
            purchase_columns = {row[1] for row in conn.execute("PRAGMA table_info(demo_mock_purchase_requests)")}
            if "confirmation_key" not in purchase_columns:
                conn.execute("ALTER TABLE demo_mock_purchase_requests ADD COLUMN confirmation_key TEXT")
            if "business_day" not in purchase_columns:
                conn.execute("ALTER TABLE demo_mock_purchase_requests ADD COLUMN business_day TEXT")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS mock_purchase_confirmation_idx ON demo_mock_purchase_requests(confirmation_key)")

    def initialize_business_day(self, business_date: str, timezone_name: str = "Africa/Cairo") -> dict:
        """Add Day 1 on a fresh database and import old date-keyed snapshots read-only."""
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute("SELECT business_day, snapshot_json, created_at FROM daily_snapshots ORDER BY business_day").fetchall()
            count = conn.execute("SELECT COUNT(*) FROM business_days").fetchone()[0]
            if count == 0 and rows:
                for index, row in enumerate(rows, start=1):
                    day_id = f"legacy-day-{row['business_day']}"
                    try:
                        summary = json.loads(row["snapshot_json"] or "{}")
                    except json.JSONDecodeError:
                        summary = {"business_day": row["business_day"], "legacy_snapshot_unparsed": True}
                    summary.setdefault("business_day_id", day_id)
                    summary.setdefault("business_date", row["business_day"])
                    summary.setdefault("day_number", index)
                    reports = {"ceo": summary.get("ceo", {}), "marketing": summary.get("marketing", {}), "inventory": summary.get("inventory", {})}
                    conn.execute("INSERT INTO business_days(day_id,day_number,business_date,timezone,started_at,ended_at,status,summary_json,reports_json,created_at) VALUES(?,?,?,?,?,?,'closed',?,?,?)",
                                 (day_id,index,row["business_day"],timezone_name,row["created_at"],row["created_at"],json.dumps(summary,ensure_ascii=False),json.dumps(reports,ensure_ascii=False),row["created_at"]))
                    members = conn.execute("SELECT DISTINCT customer_id FROM customer_activity WHERE business_day = ?", (row["business_day"],)).fetchall()
                    for member in members:
                        conn.execute("INSERT OR IGNORE INTO business_day_customers(day_id,customer_id,linked_at) VALUES(?,?,?)", (day_id,member["customer_id"],row["created_at"]))
            count = conn.execute("SELECT COUNT(*) FROM business_days").fetchone()[0]
            active = conn.execute("SELECT * FROM business_days WHERE status='active'").fetchone()
            if not active and count == 0:
                day_id = f"day-{uuid4().hex}"
                conn.execute("INSERT INTO business_days(day_id,day_number,business_date,timezone,started_at,status,created_at) VALUES(?,1,?,?,?,'active',?)",
                             (day_id,business_date,timezone_name,now,now))
                # Existing unversioned customer sessions belong to the first
                # business day when enabling day tracking on an older database.
                legacy_customers = conn.execute("SELECT customer_id FROM customers").fetchall()
                for customer in legacy_customers:
                    conn.execute("INSERT OR IGNORE INTO business_day_customers(day_id,customer_id,linked_at) VALUES(?,?,?)", (day_id,customer["customer_id"],now))
                conn.execute("INSERT INTO app_settings(setting_key,setting_value,updated_at) VALUES('active_business_day_id',?,?) ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value,updated_at=excluded.updated_at", (day_id,now))
                active = conn.execute("SELECT * FROM business_days WHERE day_id=?", (day_id,)).fetchone()
            elif active:
                conn.execute("INSERT INTO app_settings(setting_key,setting_value,updated_at) VALUES('active_business_day_id',?,?) ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value,updated_at=excluded.updated_at", (active["day_id"],now))
            else:
                conn.execute("DELETE FROM app_settings WHERE setting_key='active_customer_id'")
                conn.execute("DELETE FROM app_settings WHERE setting_key='active_business_day_id'")
            latest = conn.execute("SELECT * FROM business_days ORDER BY day_number DESC LIMIT 1").fetchone()
            return dict(active or latest) if (active or latest) else {}

    def active_business_day(self) -> dict | None:
        with self._connect() as conn:
            row=conn.execute("SELECT * FROM business_days WHERE status='active' ORDER BY day_number DESC LIMIT 1").fetchone()
            return dict(row) if row else None

    def current_business_day(self) -> dict | None:
        with self._connect() as conn:
            row=conn.execute("SELECT * FROM business_days ORDER BY day_number DESC LIMIT 1").fetchone()
            return dict(row) if row else None

    def business_day_label(self) -> str:
        record=self.current_business_day()
        if not record: return "No business day"
        status="Active" if record["status"]=="active" else "Closed"
        return f"Day {record['day_number']} — {status} ({record['business_date']})"

    def start_new_day(self, business_date: str, timezone_name: str = "Africa/Cairo") -> tuple[dict, bool]:
        """Create the next day once; repeated clicks return the active day unchanged."""
        now=_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active=conn.execute("SELECT * FROM business_days WHERE status='active' ORDER BY day_number DESC LIMIT 1").fetchone()
            if active:
                return dict(active),False
            latest=conn.execute("SELECT MAX(day_number) FROM business_days").fetchone()[0] or 0
            day_id=f"day-{uuid4().hex}"
            conn.execute("INSERT INTO business_days(day_id,day_number,business_date,timezone,started_at,status,created_at) VALUES(?,?,?,?,?,'active',?)",
                         (day_id,int(latest)+1,business_date,timezone_name,now,now))
            conn.execute("DELETE FROM app_settings WHERE setting_key='active_customer_id'")
            conn.execute("INSERT INTO app_settings(setting_key,setting_value,updated_at) VALUES('active_business_day_id',?,?) ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value,updated_at=excluded.updated_at", (day_id,now))
            return dict(conn.execute("SELECT * FROM business_days WHERE day_id=?",(day_id,)).fetchone()),True

    def list_business_days(self) -> list[dict]:
        with self._connect() as conn:
            rows=conn.execute("SELECT day_id,day_number,business_date,timezone,started_at,ended_at,status,summary_json,reports_json FROM business_days ORDER BY day_number DESC").fetchall()
        result=[]
        for row in rows:
            item=dict(row)
            item["summary"]=json.loads(item.pop("summary_json") or "{}")
            item["reports"]=json.loads(item.pop("reports_json") or "{}")
            result.append(item)
        return result

    def business_day_snapshot(self, day_id: str) -> dict | None:
        with self._connect() as conn:
            row=conn.execute("SELECT summary_json,reports_json FROM business_days WHERE day_id=?",(day_id,)).fetchone()
        if not row: return None
        summary=json.loads(row["summary_json"] or "{}")
        summary["reports"]=json.loads(row["reports_json"] or "{}")
        return summary

    def _link_customer_to_active_day(self, conn: sqlite3.Connection, customer_id: str, now: str) -> str | None:
        active=conn.execute("SELECT day_id FROM business_days WHERE status='active' ORDER BY day_number DESC LIMIT 1").fetchone()
        if not active: return None
        conn.execute("INSERT OR IGNORE INTO business_day_customers(day_id,customer_id,linked_at) VALUES(?,?,?)", (active["day_id"],customer_id,now))
        return active["day_id"]

    def _next_customer_label(self, conn: sqlite3.Connection) -> str:
        count = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        return f"Customer {count + 1}"

    def create_customer(self, display_name: str = "") -> str:
        customer_id = f"cust-{uuid4().hex[:12]}"
        now = _now()
        with self._connect() as conn:
            if not conn.execute("SELECT 1 FROM business_days WHERE status='active'").fetchone():
                raise RuntimeError("Start a business day before creating a customer")
            name = (display_name or "").strip()[:80] or self._next_customer_label(conn)
            conn.execute(
                "INSERT INTO customers(customer_id, display_name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (customer_id, name, now, now),
            )
            self._link_customer_to_active_day(conn, customer_id, now)
        return customer_id

    def create_customer_once(self, request_key: str, display_name: str = "") -> str:
        """Idempotent customer creation for retried UI events."""
        key = (request_key or "").strip()
        if not key:
            return self.create_customer(display_name)
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = conn.execute("SELECT day_id FROM business_days WHERE status='active'").fetchone()
            if not active:
                raise RuntimeError("Start a business day before creating a customer")
            scoped_key = f"{active['day_id']}:{key}"
            existing = conn.execute("SELECT customer_id FROM customer_creation_requests WHERE request_key = ?", (scoped_key,)).fetchone()
            if existing:
                return existing["customer_id"]
            customer_id = f"cust-{uuid4().hex[:12]}"
            name = (display_name or "").strip()[:80] or self._next_customer_label(conn)
            conn.execute("INSERT INTO customers(customer_id, display_name, created_at, updated_at) VALUES (?, ?, ?, ?)", (customer_id, name, now, now))
            self._link_customer_to_active_day(conn, customer_id, now)
            conn.execute("INSERT INTO customer_creation_requests VALUES (?, ?, ?)", (scoped_key, customer_id, now))
        return customer_id

    def list_customers(self, include_ended: bool = False, include_removed: bool = False) -> list[dict]:
        query = ("SELECT c.customer_id, c.display_name, c.active, c.removed, c.created_at, c.updated_at "
                 "FROM customers c JOIN business_day_customers m ON m.customer_id=c.customer_id "
                 "JOIN business_days d ON d.day_id=m.day_id")
        filters = ["d.status='active'"]
        if not include_ended:
            filters.append("c.active = 1")
        if not include_removed:
            filters.append("c.removed = 0")
        query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY c.updated_at DESC"
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query)]

    def removed_customers(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT c.customer_id, c.display_name, c.active, c.removed, c.created_at, c.updated_at FROM customers c "
                "JOIN business_day_customers m ON m.customer_id=c.customer_id JOIN business_days d ON d.day_id=m.day_id "
                "WHERE d.status='active' AND c.removed = 1 ORDER BY c.updated_at DESC"
            )]

    def is_current_day_customer(self, customer_id: str | None) -> bool:
        if not customer_id:
            return False
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM business_day_customers m JOIN business_days d ON d.day_id=m.day_id "
                                "JOIN customers c ON c.customer_id=m.customer_id WHERE m.customer_id=? AND d.status='active' "
                                "AND c.active=1 AND c.removed=0 LIMIT 1", (customer_id,)).fetchone() is not None

    def remove_customer(self, customer_id: str, business_day: str) -> bool:
        """Soft-remove one customer from active lists while preserving records."""
        with self._connect() as conn:
            # Serialize concurrent clicks and require the exact customer to be
            # active in the requested open business day.  A stale browser state
            # must never archive a different or already-completed customer.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT c.removed, c.active FROM customers c "
                "JOIN business_day_customers m ON m.customer_id=c.customer_id "
                "JOIN business_days d ON d.day_id=m.day_id "
                "WHERE c.customer_id=? AND (m.day_id=? OR d.business_date=?) AND d.status='active' LIMIT 1",
                (customer_id, business_day, business_day),
            ).fetchone()
            if not row or row["removed"] or not row["active"]:
                return False
            now = _now()
            changed = conn.execute(
                "UPDATE customers SET active=0, removed=1, updated_at=? "
                "WHERE customer_id=? AND active=1 AND removed=0",
                (now, customer_id),
            )
            if changed.rowcount != 1:
                return False
            conn.execute("DELETE FROM app_settings WHERE setting_key = 'active_customer_id' AND setting_value = ?", (customer_id,))
            conn.execute("INSERT INTO customer_activity VALUES (?, ?, ?, ?, ?, ?)",
                         (f"activity-{uuid4().hex[:12]}", customer_id, business_day, "customer_removed", "{}", now))
        return True

    def restore_customer(self, customer_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("UPDATE customers SET active = 1, removed = 0, updated_at = ? WHERE customer_id = ? AND removed = 1", (_now(), customer_id))
            return cur.rowcount == 1

    def customer(self, customer_id: str | None) -> dict | None:
        if not customer_id:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
        return dict(row) if row else None

    def is_active(self, customer_id: str | None) -> bool:
        customer = self.customer(customer_id)
        return bool(customer and customer["active"])

    def set_active_customer(self, customer_id: str) -> bool:
        if not self.is_current_day_customer(customer_id):
            return False
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO app_settings(setting_key, setting_value, updated_at) VALUES ('active_customer_id', ?, ?) "
                "ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value, updated_at=excluded.updated_at",
                (customer_id, _now()),
            )
        return True

    def active_customer_id(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT setting_value FROM app_settings WHERE setting_key = 'active_customer_id'").fetchone()
        customer_id = row["setting_value"] if row else None
        return customer_id if self.is_current_day_customer(customer_id) else None

    def save_conversation(self, customer_id: str, history: list[dict], agent_state: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE customers SET conversation_json = ?, agent_state_json = ?, updated_at = ? WHERE customer_id = ?",
                (json.dumps(history, ensure_ascii=False), json.dumps(agent_state, ensure_ascii=False), _now(), customer_id),
            )

    def conversation(self, customer_id: str) -> tuple[list[dict], dict]:
        customer = self.customer(customer_id)
        if not customer:
            return [], {}
        return json.loads(customer["conversation_json"]), json.loads(customer["agent_state_json"])

    def clear_conversation(self, customer_id: str) -> tuple[bool, int]:
        """Archive the current chat, clear its active context, and retain preferences.

        The transaction makes the archive and active reset atomic. Customer
        activity, selected products, mock purchase requests and daily snapshots
        are stored separately and are deliberately untouched.
        """
        with self._connect() as conn:
            customer = conn.execute(
                "SELECT active, conversation_json, agent_state_json FROM customers WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()
            if not customer or not customer["active"]:
                return False, 0
            history = json.loads(customer["conversation_json"] or "[]")
            stored_state = json.loads(customer["agent_state_json"] or "{}")
            # Preserve explicit long-term customer preferences while clearing
            # search filters, pending turns, displayed products and references.
            demo_state = stored_state.get("demo_conversation", stored_state)
            favorite_colors = demo_state.get("favorite_colors", []) if isinstance(demo_state, dict) else []
            style_preferences = demo_state.get("style_preferences", []) if isinstance(demo_state, dict) else []
            next_state = {
                "demo_conversation": {
                    "favorite_colors": favorite_colors if isinstance(favorite_colors, list) else [],
                    "style_preferences": style_preferences if isinstance(style_preferences, list) else [],
                    "active": {},
                    "displayed_product_ids": [],
                    "displayed_variant_ids": [],
                    "focused_product_id": None,
                    "focused_variant_id": None,
                    "selected_color": None,
                    "selected_size": None,
                    "pending_action": {},
                }
            }
            message_count = len(history)
            if message_count:
                conn.execute(
                    "INSERT INTO conversation_archives VALUES (?, ?, ?, ?, ?, ?)",
                    (f"archive-{uuid4().hex[:12]}", customer_id, message_count,
                     json.dumps(history, ensure_ascii=False),
                     json.dumps(stored_state, ensure_ascii=False), _now()),
                )
            conn.execute(
                "UPDATE customers SET conversation_json = '[]', agent_state_json = ?, updated_at = ? WHERE customer_id = ?",
                (json.dumps(next_state, ensure_ascii=False), _now(), customer_id),
            )
        return bool(message_count), message_count

    def archived_conversations(self, customer_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT archive_id, customer_id, message_count, conversation_json, agent_state_json, archived_at "
                "FROM conversation_archives WHERE customer_id = ? ORDER BY archived_at",
                (customer_id,),
            ).fetchall()
        return [
            {**dict(row), "conversation": json.loads(row["conversation_json"]),
             "agent_state": json.loads(row["agent_state_json"])}
            for row in rows
        ]

    def record_activity(self, customer_id: str, business_day: str, event_type: str, payload: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO customer_activity VALUES (?, ?, ?, ?, ?, ?)",
                (f"activity-{uuid4().hex[:12]}", customer_id, business_day, event_type,
                 json.dumps(payload, ensure_ascii=False), _now()),
            )

    @staticmethod
    def _insert_communication_event(conn: sqlite3.Connection, event_type: str, source_agent: str,
                                    target_agent: str | None, business_day_id: str,
                                    payload: dict, customer_id: str | None = None,
                                    interaction_key: str | None = None,
                                    event_id_override: str | None = None) -> str:
        validate_event(event_type, source_agent, target_agent, business_day_id, payload)
        now = _now()
        event_id = event_id_override or f"event-{uuid4().hex}"
        conn.execute(
            "INSERT OR IGNORE INTO communication_events(event_id,schema_version,event_type,source_agent,target_agent,"
            "timestamp,business_day_id,customer_id,interaction_key,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (event_id, SCHEMA_VERSION, event_type, source_agent, target_agent, now, business_day_id,
             customer_id, interaction_key, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        )
        if interaction_key:
            row = conn.execute("SELECT event_id FROM communication_events WHERE event_type=? AND business_day_id=? "
                               "AND customer_id IS ? AND interaction_key=?",
                               (event_type,business_day_id,customer_id,interaction_key)).fetchone()
            if row:
                return row["event_id"]
        return event_id

    def publish_communication_event(self, event_type: str, source_agent: str, target_agent: str | None,
                                    business_day_id: str, payload: dict, customer_id: str | None = None,
                                    interaction_key: str | None = None) -> str:
        """Validate and persist an event once; reading/consuming is a separate operation."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._insert_communication_event(conn,event_type,source_agent,target_agent,
                                                     business_day_id,payload,customer_id,interaction_key)

    def consume_communication_events(self, consumer_agent: str, business_day_id: str | None = None) -> list[dict]:
        """Return newly delivered local events and record one idempotent delivery per receiver."""
        if consumer_agent not in {"ceo","sales","marketing","inventory"}:
            raise ValueError("Unknown communication-contract consumer")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            query = ("SELECT e.* FROM communication_events e LEFT JOIN communication_deliveries d "
                     "ON d.event_id=e.event_id AND d.consumer_agent=? "
                     "WHERE d.event_id IS NULL AND (e.target_agent IS NULL OR e.target_agent=?)")
            params: list = [consumer_agent,consumer_agent]
            if business_day_id:
                query += " AND e.business_day_id=?"; params.append(business_day_id)
            query += " ORDER BY e.timestamp,e.event_id"
            rows = conn.execute(query,params).fetchall()
            now = _now()
            for row in rows:
                conn.execute("INSERT OR IGNORE INTO communication_deliveries VALUES(?,?,?)",
                             (row["event_id"],consumer_agent,now))
            return [{**dict(row),"payload":json.loads(row["payload_json"] or "{}")} for row in rows]

    def communication_event_count(self, business_day_id: str | None = None,
                                  event_types: list[str] | None = None) -> int:
        query="SELECT COUNT(*) FROM communication_events WHERE 1=1"; params=[]
        if business_day_id:
            query+=" AND business_day_id=?"; params.append(business_day_id)
        if event_types:
            query+=" AND event_type IN ("+",".join("?" for _ in event_types)+")"; params.extend(event_types)
        with self._connect() as conn:
            return int(conn.execute(query,params).fetchone()[0])

    def create_ceo_task(self, title: str, assigned_date: str) -> str:
        value=(title or "").strip()[:240]
        if not value: raise ValueError("Task description is required")
        from datetime import date
        assigned_date=date.fromisoformat(assigned_date).isoformat()
        day=self.active_business_day() or self.current_business_day()
        if not day: raise RuntimeError("Create a business day before assigning tasks")
        task_id=f"task-{uuid4().hex}"
        now=_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO ceo_tasks(task_id,title,assigned_date,assigned_by,assigned_to,is_done,created_at,updated_at) VALUES(?,?,?,'CEO','Sales',0,?,?)",
                         (task_id,value,assigned_date,now,now))
            self._insert_communication_event(conn,"ceo.task_assigned","ceo","sales",day["day_id"],
                {"task_id":task_id,"title":value,"assigned_date":assigned_date,"assigned_to":"Sales"},
                interaction_key=f"task-assigned:{task_id}")
        return task_id

    def update_ceo_task(self, task_id: str, title: str, assigned_date: str) -> bool:
        value=(title or "").strip()[:240]
        if not value: raise ValueError("Task description is required")
        from datetime import date
        assigned_date=date.fromisoformat(assigned_date).isoformat()
        day=self.active_business_day() or self.current_business_day()
        if not day: raise RuntimeError("No business day exists")
        now=_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old=conn.execute("SELECT title,assigned_date FROM ceo_tasks WHERE task_id=?",(task_id,)).fetchone()
            if not old:return False
            changes={"title":{"from":old["title"],"to":value},"assigned_date":{"from":old["assigned_date"],"to":assigned_date}}
            conn.execute("UPDATE ceo_tasks SET title=?,assigned_date=?,updated_at=? WHERE task_id=?",(value,assigned_date,now,task_id))
            self._insert_communication_event(conn,"ceo.task_updated","ceo","sales",day["day_id"],
                {"task_id":task_id,"changes":changes},interaction_key=f"task-updated:{task_id}:{now}")
        return True

    def list_ceo_tasks(self, assigned_date: str | None = None) -> list[dict]:
        query="SELECT * FROM ceo_tasks"; params=[]
        if assigned_date:
            query+=" WHERE assigned_date=?"; params.append(assigned_date)
        query+=" ORDER BY assigned_date,created_at,task_id"
        with self._connect() as conn:return [dict(row) for row in conn.execute(query,params)]

    def set_task_done_by_sales(self, task_id: str, is_done: bool, source_agent: str = "sales") -> bool:
        if source_agent!="sales": raise PermissionError("Only the Sales execution agent may update task completion")
        day=self.active_business_day() or self.current_business_day()
        if not day: raise RuntimeError("No business day exists")
        now=_now(); value=int(bool(is_done))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            task=conn.execute("SELECT title,is_done FROM ceo_tasks WHERE task_id=? AND assigned_to='Sales'",(task_id,)).fetchone()
            if not task:return False
            if int(task["is_done"])==value:return True
            conn.execute("UPDATE ceo_tasks SET is_done=?,completed_at=?,updated_at=? WHERE task_id=?",
                         (value,now if value else None,now,task_id))
            event_type="sales.task_completed" if value else "sales.task_status_updated"
            payload={"task_id":task_id,"title":task["title"],"is_done":bool(value),"completed_at":now if value else None}
            self._insert_communication_event(conn,event_type,"sales","ceo",day["day_id"],payload,
                                             interaction_key=f"task-status:{task_id}:{value}:{now}")
        return True

    def list_communication_reports(self, agent: str | None = None) -> list[dict]:
        query="SELECT business_day_id,agent,schema_version,report_json,generated_at FROM communication_reports";params=[]
        if agent:query+=" WHERE agent=?";params.append(agent)
        query+=" ORDER BY generated_at DESC,business_day_id DESC"
        with self._connect() as conn:
            return [{**dict(row),"report":json.loads(row["report_json"] or "{}")} for row in conn.execute(query,params)]

    def communication_report(self, business_day_id: str, agent: str) -> dict | None:
        if agent not in {"sales","ceo","marketing","inventory"}:return None
        with self._connect() as conn:
            row=conn.execute("SELECT business_day_id,agent,schema_version,report_json,generated_at FROM communication_reports WHERE business_day_id=? AND agent=? ORDER BY schema_version DESC LIMIT 1",(business_day_id,agent)).fetchone()
        return {**dict(row),"report":json.loads(row["report_json"] or "{}")} if row else None

    def marketing_contract_data(self, from_date: str | None = None, to_date: str | None = None) -> dict:
        with self._connect() as conn:
            query=("SELECT e.* FROM communication_events e JOIN business_days d ON d.day_id=e.business_day_id "
                   "WHERE e.event_type IN ('sales.customer_interest_recorded','sales.customer_feedback_recorded',"
                   "'sales.unmet_demand_recorded','sales.purchase_confirmed')")
            params=[]
            if from_date:query+=" AND d.business_date>=?";params.append(from_date)
            if to_date:query+=" AND d.business_date<=?";params.append(to_date)
            query+=" ORDER BY e.timestamp,e.event_id"
            events=[{**dict(row),"payload":json.loads(row["payload_json"] or "{}")} for row in conn.execute(query,params)]
            days_query="SELECT COUNT(DISTINCT a.customer_id) FROM customer_activity a JOIN business_days d ON d.day_id=a.business_day "
            days_query+="WHERE a.event_type='customer_message'"
            day_params=[]
            if from_date:days_query+=" AND d.business_date>=?";day_params.append(from_date)
            if to_date:days_query+=" AND d.business_date<=?";day_params.append(to_date)
            conversations=int(conn.execute(days_query,day_params).fetchone()[0])
        interests=[e for e in events if e["event_type"]=="sales.customer_interest_recorded"]
        feedback=[e for e in events if e["event_type"]=="sales.customer_feedback_recorded"]
        unmet=[e for e in events if e["event_type"]=="sales.unmet_demand_recorded"]
        purchases=[e for e in events if e["event_type"]=="sales.purchase_confirmed"]
        products={}
        for event in interests:
            product=event["payload"].get("product_id","unknown");products[product]=products.get(product,0)+1
        return {"conversation_customer_count":conversations,"interest_count":len(interests),"product_interest_counts":products,
                "explicit_positive_count":sum(e["payload"].get("feedback_type")=="positive" for e in feedback),
                "accepted_recommendation_count":sum(e["payload"].get("feedback_type")=="accepted" for e in feedback),
                "rejection_count":sum(e["payload"].get("feedback_type")=="rejected" for e in feedback),
                "unmet_demand_count":len(unmet),"confirmed_mock_purchase_count":len(purchases),
                "customer_details":[{"event_type":e["event_type"],"business_day_id":e["business_day_id"],**e["payload"]}
                                    for e in events],"events":events}

    def inventory_contract_data(self, from_date: str | None = None, to_date: str | None = None) -> dict:
        with self._connect() as conn:
            query=("SELECT p.*,c.display_name,d.business_date FROM demo_mock_purchase_requests p "
                   "LEFT JOIN customers c ON c.customer_id=p.customer_id LEFT JOIN business_days d ON d.day_id=p.business_day "
                   "WHERE p.status='confirmed_mock_purchase'")
            params=[]
            if from_date:query+=" AND d.business_date>=?";params.append(from_date)
            if to_date:query+=" AND d.business_date<=?";params.append(to_date)
            query+=" ORDER BY p.created_at,p.request_id"
            orders=[]; by_variant={}; total=0
            for row in conn.execute(query,params):
                order=dict(row); order["items"]=json.loads(order.pop("items_json") or "[]"); order["total_piasters"]=int(order["total_piasters"]); total+=order["total_piasters"]
                for item in order["items"]:
                    vid=str(item.get("variant_id", "")); qty=int(item.get("quantity",1)); unit=int(item.get("unit_price_piasters",0))
                    rec=by_variant.setdefault(vid,{"variant_id":vid,"product_id":item.get("product_id"),"quantity_sold":0,"unit_price_piasters":unit,"revenue_piasters":0})
                    rec["quantity_sold"]+=qty; rec["revenue_piasters"]+=qty*unit
                orders.append(order)
        return {"orders":orders,"variant_sales":list(by_variant.values()),"confirmed_mock_orders":len(orders),
                "mock_revenue_piasters":total,"cost_of_goods_piasters":None,"gross_profit_piasters":None,
                "net_profit_piasters":None,"stock_deducted":False,
                "stock_note":"Workbook stock is simulated DEMO data; no authoritative inventory feed or stock deduction is connected."}

    def end_customer(self, customer_id: str, business_day: str) -> None:
        if not self.is_active(customer_id):
            return
        with self._connect() as conn:
            conn.execute("UPDATE customers SET active = 0, updated_at = ? WHERE customer_id = ?", (_now(), customer_id))
        self.record_activity(customer_id, business_day, "customer_completed", {})

    def day_is_closed(self, business_day: str) -> bool:
        with self._connect() as conn:
            return (conn.execute("SELECT 1 FROM business_days WHERE day_id=? AND status='closed'", (business_day,)).fetchone() is not None
                    or conn.execute("SELECT 1 FROM daily_snapshots WHERE business_day = ?", (business_day,)).fetchone() is not None)

    def _contract_reports_for_day(self, conn: sqlite3.Connection, day: sqlite3.Row,
                                  members: list[str], events: list[dict], purchases: list[dict]) -> dict:
        generated_at=_now();day_id=day["day_id"]
        tasks=[dict(row) for row in conn.execute("SELECT task_id,title,assigned_date,assigned_by,assigned_to,is_done,completed_at,created_at,updated_at FROM ceo_tasks WHERE assigned_date=? ORDER BY created_at,task_id",(day["business_date"],))]
        task_rows=[{**task,"is_done":bool(task["is_done"]),"completion_state":"Done" if task["is_done"] else "Not Done"} for task in tasks]
        completed=sum(bool(task["is_done"]) for task in tasks)
        typed_events=[]
        for event in events:
            item=dict(event);item["payload"]=json.loads(item.pop("payload_json") or "{}")
            typed_events.append(item)
        interest=[e for e in typed_events if e["event_type"]=="sales.customer_interest_recorded"]
        feedback=[e for e in typed_events if e["event_type"]=="sales.customer_feedback_recorded"]
        unmet=[e for e in typed_events if e["event_type"]=="sales.unmet_demand_recorded"]
        product_interest={};positive={};rejections={};rejection_reasons={}
        for event in interest:
            pid=str(event["payload"].get("product_id","unknown"));product_interest[pid]=product_interest.get(pid,0)+1
        for event in feedback:
            p=event["payload"];pid=str(p.get("product_id","unknown"));kind=p.get("feedback_type")
            if kind=="positive":positive[pid]=positive.get(pid,0)+1
            elif kind=="rejected":
                rejections[pid]=rejections.get(pid,0)+1
                reason=str(p.get("reason") or "not specified")
                bucket=rejection_reasons.setdefault(pid,{})
                bucket[reason]=bucket.get(reason,0)+1
        legacy_categories={};legacy_colors={};legacy_products={}
        for event in conn.execute("SELECT payload_json FROM customer_activity WHERE business_day=? AND event_type='ranked_catalog_inquiry'",(day_id,)):
            payload=json.loads(event["payload_json"] or "{}")
            for key,target in (("categories",legacy_categories),("colors",legacy_colors),("product_ids",legacy_products)):
                for value in payload.get(key,[]):target[str(value)]=target.get(str(value),0)+1
        customer_details=[{"event_id":e["event_id"],"event_type":e["event_type"],"timestamp":e["timestamp"],"customer_id":e.get("customer_id"),**e["payload"]}
                          for e in typed_events if e["event_type"].startswith("sales.customer_") or e["event_type"]=="sales.unmet_demand_recorded"]
        inventory_lines={}; purchase_details=[]; mock_value=0
        for purchase in purchases:
            items=json.loads(purchase["items_json"] or "[]");total=int(purchase["total_piasters"]);mock_value+=total
            customer=conn.execute("SELECT display_name FROM customers WHERE customer_id=?",(purchase["customer_id"],)).fetchone()
            purchase_details.append({"order_id":purchase["request_id"],"event_id":f"purchase:{purchase['request_id']}",
                "customer_id":purchase["customer_id"],"customer_label":customer["display_name"] if customer else "Customer (archived)",
                "business_day_id":day_id,"timestamp":purchase["created_at"],"items":items,
                "total_piasters":total,"status":"confirmed_mock_purchase","transaction_type":"MOCK"})
            for item in items:
                if isinstance(item,str):item={"variant_id":item,"quantity":1}
                variant_id=str(item.get("variant_id",""));quantity=int(item.get("quantity",1));unit=int(item.get("unit_price_piasters",0))
                rec=inventory_lines.setdefault(variant_id,{"product_id":item.get("product_id"),"variant_id":variant_id,
                    "color":item.get("color"),"size":item.get("size"),"quantity_sold":0,
                    "unit_price_piasters":unit,"revenue_piasters":0})
                rec["quantity_sold"]+=quantity;rec["revenue_piasters"]+=quantity*unit
        report_base={"schema_version":SCHEMA_VERSION,"business_day_id":day_id,"business_date":day["business_date"],"generated_at":generated_at}
        sales={**report_base,"customers_served":len(members),"messages":sum(e["event_type"]=="customer_message" for e in conn.execute("SELECT event_type FROM customer_activity WHERE business_day=?",(day_id,))),
            "confirmed_mock_orders":len(purchase_details),"mock_revenue_piasters":mock_value,
            "real_sales_revenue_piasters":0,"profit_loss_piasters":None,
            "note":"MOCK transaction value only; real sales, costs and profit/loss are unavailable."}
        ceo={**report_base,"total_tasks":len(tasks),"completed_tasks":completed,"not_done_tasks":len(tasks)-completed,
            "completion_text":f"{completed} / {len(tasks)}","tasks":task_rows,
            "real_sales_piasters":0,"pending_tasks":len(tasks)-completed,
            "mock_purchase_requests":len(purchase_details),"mock_value_piasters":mock_value,
            "note":"Tasks are counted only for this assigned date. End Day does not complete tasks automatically."}
        marketing={**report_base,"conversation_customer_count":len(members),"product_interest_counts":product_interest,
            "explicit_positive_feedback_counts":positive,"explicit_rejection_counts":rejections,
            "rejection_reason_counts":rejection_reasons,"unmet_demand_count":len(unmet),
            "confirmed_mock_purchases":purchase_details,"customer_specific_feedback":customer_details,
            "category_interest":legacy_categories,"color_interest":legacy_colors,"ranked_product_interest":legacy_products,
            "product_views":sum(legacy_products.values()),"mock_purchased_variant_quantities":{
                vid:row["quantity_sold"] for vid,row in inventory_lines.items()},
            "note":"Interest, positive feedback, acceptance, rejection, unmet demand and MOCK purchases are separate event types."}
        inventory={**report_base,"confirmed_mock_orders":purchase_details,"variant_sales":list(inventory_lines.values()),
            "mock_revenue_piasters":mock_value,"cost_of_goods_piasters":None,"gross_profit_piasters":None,"other_expenses_piasters":None,"net_profit_loss_piasters":None,
            "current_stock_authoritative":False,"stock_deducted":False,
            "stock_note":"Workbook stock is simulated DEMO data; no authoritative inventory feed or stock deduction is connected."}
        return {"schema_version":SCHEMA_VERSION,"sales":sales,"ceo":ceo,"marketing":marketing,"inventory":inventory}

    def end_active_day(self) -> tuple[dict, bool]:
        """Close the active business day and freeze its factual report payloads."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            day = conn.execute("SELECT * FROM business_days WHERE status='active'").fetchone()
            if not day:
                latest = conn.execute("SELECT summary_json FROM business_days ORDER BY day_number DESC LIMIT 1").fetchone()
                if latest:
                    return json.loads(latest["summary_json"] or "{}"), False
                return {}, False
            day_id = day["day_id"]
            members = [row["customer_id"] for row in conn.execute(
                "SELECT customer_id FROM business_day_customers WHERE day_id=?", (day_id,))]
            events = [dict(row) for row in conn.execute(
                "SELECT activity_id,customer_id,event_type,payload_json,created_at FROM customer_activity WHERE business_day=? ORDER BY created_at", (day_id,))]
            contract_events = [dict(row) for row in conn.execute(
                "SELECT event_id,event_type,timestamp,business_day_id,customer_id,payload_json FROM communication_events WHERE business_day_id=? ORDER BY timestamp,event_id",(day_id,))]
            purchases = [dict(row) for row in conn.execute(
                "SELECT request_id,customer_id,total_piasters,items_json,created_at FROM demo_mock_purchase_requests "
                "WHERE status='confirmed_mock_purchase' AND business_day=? ORDER BY created_at", (day_id,))]
            conversations = {cid: json.loads(row["conversation_json"] or "[]") for cid, row in (
                (cid, conn.execute("SELECT conversation_json FROM customers WHERE customer_id=?", (cid,)).fetchone()) for cid in members
            ) if row}
            customer_rows = [dict(row) for row in conn.execute(
                "SELECT customer_id,display_name,active,removed,agent_state_json FROM customers WHERE customer_id IN "
                "(SELECT customer_id FROM business_day_customers WHERE day_id=?)", (day_id,))]
            messages = [e for e in events if e["event_type"] == "customer_message"]
            inquiries = [e for e in events if e["event_type"] == "ranked_catalog_inquiry"]
            completed = sum(e["event_type"] == "customer_completed" for e in events)
            mock_value = sum(int(p["total_piasters"]) for p in purchases)
            variants: dict[str, int] = {}
            product_interest: dict[str, int] = {}
            category_interest: dict[str, int] = {}
            color_interest: dict[str, int] = {}
            for purchase in purchases:
                for item in json.loads(purchase["items_json"] or "[]"):
                    if isinstance(item, str):
                        variant_id, quantity = item, 1
                    else:
                        variant_id, quantity = item.get("variant_id", ""), int(item.get("quantity", 1))
                    if variant_id:
                        variants[variant_id] = variants.get(variant_id, 0) + quantity
            for event in inquiries:
                payload = json.loads(event["payload_json"] or "{}")
                for key, target in (("product_ids", product_interest), ("categories", category_interest), ("colors", color_interest)):
                    for value in payload.get(key, []):
                        target[str(value)] = target.get(str(value), 0) + 1
            preferences = {}
            for customer in customer_rows:
                state = json.loads(customer["agent_state_json"] or "{}")
                demo = state.get("demo_conversation", state)
                colors = demo.get("favorite_colors", []) if isinstance(demo, dict) else []
                if colors:
                    preferences[customer["customer_id"]] = colors
            ended_at = _now()
            summary = {
                "schema_version": 3, "business_day_id": day_id, "day_number": day["day_number"],
                "business_date": day["business_date"], "timezone": day["timezone"],
                "started_at": day["started_at"], "ended_at": ended_at, "status": "closed",
                "customer_session_ids": members, "conversation_message_counts": {cid: len(hist) for cid, hist in conversations.items()},
                "conversation_records": [{"customer_id": cid, "message_count": len(conversations.get(cid, []))} for cid in members],
                "events": {"total": len(events), "customer_messages": len(messages), "product_inquiries": len(inquiries),
                           "completed_customers": completed, "activity_ids": [e["activity_id"] for e in events]},
                "mock_purchase_records": [{"request_id": p["request_id"], "customer_id": p["customer_id"],
                    "created_at": p["created_at"], "total_piasters": int(p["total_piasters"]),
                    "items": json.loads(p["items_json"] or "[]")} for p in purchases],
                "sales": {"confirmed_orders": 0, "revenue_piasters": 0, "note": "No real payment/order ledger is connected."},
                "demo_mock_purchases": {"confirmed_requests": len(purchases), "simulated_value_piasters": mock_value,
                    "verified_variant_quantities": variants, "note": "Demo mock purchase requests only; not sales or real revenue."},
                "customer_sessions": {"customers_served": len(members), "completed": completed,
                    "active_at_close": sum(bool(c["active"]) and not bool(c["removed"]) for c in customer_rows),
                    "removed_from_active_list": sum(bool(c["removed"]) for c in customer_rows),
                    "pending_tasks": None, "task_metrics_note": "No task ledger is implemented; task counts are unavailable."},
            }
            reports=self._contract_reports_for_day(conn,day,members,contract_events,purchases)
            summary["reports"] = reports
            for agent in ("ceo","marketing","inventory","sales"):
                report=reports[agent]
                conn.execute("INSERT OR IGNORE INTO communication_reports VALUES(?,?,?,?,?)",
                             (day_id,agent,SCHEMA_VERSION,json.dumps(report,ensure_ascii=False,sort_keys=True),ended_at))
            self._insert_communication_event(conn,"sales.business_day_closed","sales",None,day_id,
                {"business_day_id":day_id,"report_versions":{agent:SCHEMA_VERSION for agent in ("ceo","marketing","inventory","sales")}},
                interaction_key=f"business-day-closed:{day_id}")
            conn.execute("UPDATE business_days SET status='closed',ended_at=?,summary_json=?,reports_json=? WHERE day_id=? AND status='active'",
                         (ended_at,json.dumps(summary,ensure_ascii=False),json.dumps(reports,ensure_ascii=False),day_id))
            # day_id keys allow multiple business days on the same calendar date.
            conn.execute("INSERT OR IGNORE INTO daily_snapshots VALUES (?,3,?,?)",
                         (day_id,json.dumps(summary,ensure_ascii=False),ended_at))
            conn.execute("DELETE FROM app_settings WHERE setting_key IN ('active_customer_id','active_business_day_id')")
            return summary, True

    def end_day(self, business_day: str) -> tuple[dict, bool]:
        """Create one immutable report per day; return the prior report on repeat."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT snapshot_json FROM daily_snapshots WHERE business_day = ?", (business_day,)).fetchone()
            if existing:
                return json.loads(existing["snapshot_json"]), False

            events = [dict(row) for row in conn.execute(
                "SELECT event_type, payload_json FROM customer_activity WHERE business_day = ?", (business_day,)
            )]
            customers = conn.execute("SELECT customer_id, active, removed, agent_state_json FROM customers").fetchall()
            messages = [event for event in events if event["event_type"] == "customer_message"]
            inquiries = [event for event in events if event["event_type"] == "ranked_catalog_inquiry"]
            completed = sum(1 for event in events if event["event_type"] == "customer_completed")
            purchases = [dict(row) for row in conn.execute(
                "SELECT request_id, customer_id, total_piasters, items_json, created_at FROM demo_mock_purchase_requests WHERE status = 'confirmed_mock_purchase' AND business_day = ?", (business_day,)
            )]
            demo_value_piasters = sum(int(row["total_piasters"]) for row in purchases)
            purchased_variants: dict[str, int] = {}
            for purchase in purchases:
                for item in json.loads(purchase["items_json"] or "[]"):
                    if isinstance(item, str):
                        variant_id, quantity = item, 1
                    else:
                        variant_id, quantity = item.get("variant_id", ""), int(item.get("quantity", 1))
                    if variant_id:
                        purchased_variants[variant_id] = purchased_variants.get(variant_id, 0) + quantity
            ranked_product_ids: dict[str, int] = {}
            for event in inquiries:
                for sku in json.loads(event["payload_json"]).get("product_ids", []):
                    ranked_product_ids[sku] = ranked_product_ids.get(sku, 0) + 1
            preferences: dict[str, list[str]] = {}
            for customer in customers:
                state = json.loads(customer["agent_state_json"] or "{}")
                colors = state.get("favorite_colors", [])
                if colors:
                    preferences[customer["customer_id"]] = colors
            snapshot = {
                "schema_version": 2,
                "business_day": business_day,
                "generated_at": _now(),
                "customer_sessions": {
                    "total_known": len(customers),
                    "active_at_close": sum(1 for customer in customers if customer["active"]),
                    "removed_from_active_list": sum(1 for customer in customers if customer["removed"]),
                    "completed_today": completed,
                    "customer_messages_today": len(messages),
                },
                "sales": {"confirmed_orders": 0, "revenue_egp": 0, "note": "No real order or payment workflow is implemented."},
                "demo_mock_purchases": {
                    "confirmed_requests": len(purchases),
                    "simulated_value_egp": demo_value_piasters / 100,
                    "verified_variant_quantities": purchased_variants,
                    "note": "Mock purchases only; these are not real sales, revenue, or stock deductions.",
                },
                "marketing": {
                    "ranked_catalog_inquiries": len(inquiries),
                    "ranked_product_interest": ranked_product_ids,
                    "stored_favorite_colors_by_customer": preferences,
                },
                "inventory": {"mock_requested_variant_quantities": purchased_variants, "stock_deducted": False,
                              "verified_purchased_variants": [],
                              "note": "Demo mock purchase requests are not verified real orders; no stock was deducted."},
                "pending_tasks": [],
            }
            conn.execute(
                "INSERT INTO daily_snapshots VALUES (?, ?, ?, ?)",
                (business_day, 2, json.dumps(snapshot, ensure_ascii=False), _now()),
            )
        return snapshot, True

    def create_outfit(self, customer_id: str, display_name: str = "") -> str:
        outfit_id = f"outfit-{uuid4().hex[:12]}"
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM demo_outfits WHERE customer_id = ?", (customer_id,)).fetchone()[0]
            name = (display_name or "").strip()[:80] or f"Mannequin {count + 1}"
            now = _now()
            conn.execute("INSERT INTO demo_outfits VALUES (?, ?, ?, ?, ?)", (outfit_id, customer_id, name, now, now))
        return outfit_id

    def list_outfits(self, customer_id: str) -> list[dict]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM demo_outfits WHERE customer_id = ? ORDER BY created_at", (customer_id,))]

    def add_or_replace_outfit_item(self, outfit_id: str, variant_id: str, category: str, product_id: str, quantity: int = 1) -> None:
        # The Sales Conversation right panel is a cart, not an outfit slot:
        # keep distinct products in the same category as separate line items.
        slot_key = f"{category}:{product_id}"
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO demo_outfit_items(outfit_id, slot_key, variant_id, added_at, quantity) VALUES (?, ?, ?, ?, ?) ON CONFLICT(outfit_id, slot_key) DO UPDATE SET variant_id = excluded.variant_id, quantity = excluded.quantity, added_at = excluded.added_at",
                (outfit_id, slot_key, variant_id, _now(), max(1, int(quantity))),
            )
            conn.execute("UPDATE demo_outfits SET updated_at = ? WHERE outfit_id = ?", (_now(), outfit_id))

    def remove_outfit_item(self, outfit_id: str, slot_key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM demo_outfit_items WHERE outfit_id = ? AND slot_key = ?", (outfit_id, slot_key))

    def outfit_record(self, outfit_id: str | None) -> dict:
        if not outfit_id:
            return {"outfit_id": None, "display_name": "No mannequin", "items": []}
        with self._connect() as conn:
            outfit = conn.execute("SELECT * FROM demo_outfits WHERE outfit_id = ?", (outfit_id,)).fetchone()
            items = conn.execute("SELECT slot_key, variant_id, quantity FROM demo_outfit_items WHERE outfit_id = ? ORDER BY added_at", (outfit_id,)).fetchall()
        return {"outfit_id": outfit_id,
                "customer_id": outfit["customer_id"] if outfit else None,
                "display_name": outfit["display_name"] if outfit else "No mannequin",
                "items": [dict(item) for item in items]}

    def set_item_quantity(self, outfit_id: str, slot_key: str, quantity: int) -> bool:
        value = int(quantity)
        if value < 1 or value > 99:
            return False
        with self._connect() as conn:
            cur = conn.execute("UPDATE demo_outfit_items SET quantity = ?, added_at = ? WHERE outfit_id = ? AND slot_key = ?", (value, _now(), outfit_id, slot_key))
            conn.execute("UPDATE demo_outfits SET updated_at = ? WHERE outfit_id = ?", (_now(), outfit_id))
            return cur.rowcount == 1

    def record_mock_purchase(self, customer_id: str, outfit_id: str, total_piasters: int, variant_ids: list, confirmation_key: str | None = None, business_day: str | None = None) -> str:
        """Persist a confirmed mock request exactly once per UI confirmation token."""
        request_id = f"mock-{uuid4().hex[:12]}"
        items = variant_ids if variant_ids and isinstance(variant_ids[0], dict) else [{"variant_id": value, "quantity": 1} for value in variant_ids]
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if confirmation_key:
                existing = conn.execute("SELECT request_id FROM demo_mock_purchase_requests WHERE confirmation_key = ?", (confirmation_key,)).fetchone()
                if existing:
                    return existing["request_id"]
            conn.execute("INSERT INTO demo_mock_purchase_requests(request_id, customer_id, outfit_id, total_piasters, status, items_json, created_at, confirmation_key, business_day) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (request_id, customer_id, outfit_id, total_piasters, "confirmed_mock_purchase", json.dumps(items, ensure_ascii=False), now, confirmation_key, business_day))
            if business_day:
                conn.execute("INSERT INTO customer_activity VALUES (?, ?, ?, ?, ?, ?)",
                             (f"activity-{uuid4().hex[:12]}", customer_id, business_day, "mock_purchase_confirmed",
                              json.dumps({"request_id": request_id, "items": items, "total_piasters": total_piasters}, ensure_ascii=False), now))
                customer=conn.execute("SELECT display_name FROM customers WHERE customer_id=?",(customer_id,)).fetchone()
                event_id=f"event-purchase-{request_id}"
                self._insert_communication_event(conn,"sales.purchase_confirmed","sales",None,business_day,
                    {"event_id":event_id,"order_id":request_id,"customer_id":customer_id,
                     "customer_label":customer["display_name"] if customer else "Customer (archived)",
                     "business_day_id":business_day,"timestamp":now,"items":items,
                     "total_piasters":int(total_piasters),"status":"confirmed_mock_purchase",
                     "transaction_type":"MOCK","stock_deducted":False},
                    customer_id=customer_id,interaction_key=f"purchase:{request_id}",event_id_override=event_id)
        return request_id

    def mock_purchase(self, request_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM demo_mock_purchase_requests WHERE request_id = ?", (request_id,)).fetchone()
        if not row:
            return None
        record = dict(row)
        record["items"] = json.loads(record["items_json"] or "[]")
        return record

    def mock_purchase_for_confirmation(self, confirmation_key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT request_id FROM demo_mock_purchase_requests WHERE confirmation_key = ?", (confirmation_key,)).fetchone()
        return self.mock_purchase(row["request_id"]) if row else None
