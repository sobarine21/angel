# whatsapp_supabase_webhook.py
import os
import logging
import requests
from flask import Flask, request, jsonify
from urllib.parse import quote_plus
from datetime import datetime

# --- CONFIG (expected to be set in env) ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

META_TOKEN = os.environ.get("META_WHATSAPP_TOKEN", "")
VERIFY_TOKEN = os.environ.get("META_VERIFY_TOKEN", "verify123")
PHONE_NUMBER_ID = os.environ.get("META_PHONE_NUMBER_ID", "")

# Basic checks
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Supabase URL and SERVICE_ROLE_KEY must be set in environment variables!")

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whatsapp_webhook")

# --- HTTP defaults ---
REQUEST_TIMEOUT = 12  # seconds
HEADERS_SERVICE = {
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Content-Type": "application/json",
}
HEADERS_ANON = {
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}" if SUPABASE_ANON_KEY else "",
    "apikey": SUPABASE_ANON_KEY if SUPABASE_ANON_KEY else "",
    "Content-Type": "application/json",
}

# --- FLASK APP ---
app = Flask(__name__)

# --- HELPERS ---


def check_user(token: str):
    if not token:
        return None, "Missing Authorization header"

    if not token.lower().startswith("bearer "):
        token = f"Bearer {token}"

    try:
        headers = {"Authorization": token}
        resp = requests.get(f"{SUPABASE_URL}/auth/v1/user", headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.info(f"Supabase auth check failed: {resp.status_code} {resp.text}")
            return None, "Unauthorized"
        return resp.json(), None
    except Exception as e:
        logger.exception("User auth failed")
        return None, str(e)


def increment_query_count(user_id: str):
    try:
        url = f"{SUPABASE_URL}/rpc/check_and_increment_query_count"
        payload = {"p_user_id": user_id}
        resp = requests.post(url, json=payload, headers=HEADERS_SERVICE, timeout=REQUEST_TIMEOUT)
        if resp.status_code not in (200, 201, 202):
            logger.error("RPC increment_query_count failed: %s %s", resp.status_code, resp.text)
            return None, f"RPC error: {resp.status_code}"
        return resp.json(), None
    except Exception as e:
        logger.exception("Error calling RPC")
        return None, str(e)


def _build_or_clause(cols, value, exact: bool):
    parts = []
    if exact:
        for c in cols:
            parts.append(f"{c}.eq.{value}")
    else:
        for c in cols:
            parts.append(f"{c}.ilike.%25{value}%25")
    joined = ",".join(parts)
    return f"({joined})"


def search_table_via_functions(tablename: str, cols: list, search_term: str, exact: bool):
    try:
        base = f"{SUPABASE_URL}/rest/v1"
        safe_table = quote_plus(tablename)
        or_clause = _build_or_clause(cols, search_term, exact)
        qs = f"select=*&or={quote_plus(or_clause, safe='')}"
        url = f"{base}/{safe_table}?{qs}"
        logger.debug("Searching table URL: %s", url)

        resp = requests.get(url, headers=HEADERS_SERVICE, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 401:
            return {"table": tablename, "matches": [], "count": 0, "searchFields": cols, "error": "Unauthorized"}
        if resp.status_code >= 400:
            return {"table": tablename, "matches": [], "count": 0, "searchFields": cols, "error": f"{resp.status_code}: {resp.text}"}
        data = resp.json()
        return {"table": tablename, "matches": data or [], "count": len(data or []), "searchFields": cols}
    except Exception as e:
        logger.exception("Error searching table %s", tablename)
        return {"table": tablename, "matches": [], "count": 0, "searchFields": cols, "error": str(e)}


def search_all_tables(search_term: str, is_exact: bool):
    tables = [
        {"table": "world_bank_sanctioned", "columns": ["Firm Name"]},
        {"table": "sebi_circulars", "columns": ["title"]},
        {"table": "nse_under_liquidations", "columns": ["Name of the Company"]},
        {"table": "nse_suspended", "columns": ["Company Name"]},
        {"table": "nse_banned_debared", "columns": ["banned Entity", "PAN"]},
        {"table": "disqualified_directors", "columns": ["CIN", "Company Name", "DIN", "Director Name"]},
        {"table": "ibbi_nclt_orders", "columns": ["Case Title"]},
        {"table": "indian_local_politicians", "columns": ["Candidate Name"]},
        {"table": "ibbi_supreme_court_orders", "columns": ["Case Title"]},
        {"table": "ibbi_orders", "columns": ["Case Title"]},
        {"table": "ibbi_nclat_orders", "columns": ["Case Title"]},
        {"table": "ibbi_high_courts_orders", "columns": ["Case Title"]},
        {"table": "directors_struckoff", "columns": ["CIN", "Company Name", "PAN", "DIN", "Director Name"]},
        {"table": "euro_sanction", "columns": ["name"]},
        {"table": "esma_sanctions", "columns": ["Name"]},
        {"table": "delisted_under_liquidations_nse", "columns": ["Name of the Company"]},
        {"table": "defaulting_clients_nse", "columns": ["Name of the Defaulting client", "PanofClient"]},
        {"table": "defaulting_clients_ncdex", "columns": ["Name of the Defaulting client", "PANof theClient"]},
        {"table": "defaulting_clients_mcx", "columns": ["Name of the Defaulting client", "PANoftheClient"]},
        {"table": "defaulting_clients_bse", "columns": ["Name of  Defaulting Client", "Pan of Client"]},
        {"table": "crip_nse_cases", "columns": ["Name of the Company"]},
        {"table": "consolidatedLegacyByPRN", "columns": ["ENTITY"]},
        {"table": "banned by  Competent Authorities India", "columns": ["Entity/Individual Name", "PAN No."]},
        {"table": "banned _list_uapa", "columns": ["Organisation"]},
        {"table": "SEBI_DEACTIVATED", "columns": ["Name of the Entity", "PANNo."]},
        {"table": "GLOBAL_SDN", "columns": ["Name"]},
        {"table": "Archive SEBI DEBARRED entities", "columns": ["Name of the clients", "PAN No."]},
        {"table": "Defaulting_Client_Database nse_", "columns": ["Name of the Defaulting client", "Pan of Client"]},
        {"table": "Companies_IBC_Moratorium_Debt", "columns": ["Company Name"]},
    ]
    results = [search_table_via_functions(t["table"], t["columns"], search_term, is_exact) for t in tables]
    return results


def send_whatsapp_reply(to: str, message: str):
    if not META_TOKEN or not PHONE_NUMBER_ID:
        logger.error("WhatsApp credentials not set!")
        return False, "WhatsApp credentials not set"
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {META_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "text": {"body": message}}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code >= 400:
            logger.error("Error sending WhatsApp reply: %s %s", resp.status_code, resp.text)
            return False, f"{resp.status_code}: {resp.text}"
        logger.info("WhatsApp reply sent to %s", to)
        return True, None
    except Exception as e:
        logger.exception("Error sending WhatsApp message")
        return False, str(e)


# --- ROUTES ---
@app.route("/whatsapp-webhook", methods=["GET", "POST"])
def whatsapp_webhook():
    if request.method == "GET":
        if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.verify_token") == VERIFY_TOKEN:
            return request.args.get("hub.challenge"), 200
        return "Verification failed", 403

    data = request.get_json(silent=True)
    logger.info("Incoming webhook payload: %s", data)
    if not data:
        return "Bad Request", 400

    entry = data.get("entry", [{}])[0]
    changes = entry.get("changes", [{}])[0]
    value = changes.get("value", {})

    if "statuses" in value:
        return "Status update", 200

    messages = value.get("messages")
    if not messages:
        return "No messages", 200

    message = messages[0]
    from_number = message.get("from")
    message_type = message.get("type")
    if message_type != "text":
        return "Non-text message", 200

    text = message.get("text", {}).get("body", "").strip()
    if not text:
        return "Empty message", 200

    auth_header = request.headers.get("Authorization")
    user, err = (None, None)
    if auth_header:
        user, err = check_user(auth_header)
        if err:
            logger.info("User auth failed or unauthorized: %s", err)
            user = None
    user_id = user.get("id") if user and isinstance(user, dict) and user.get("id") else "public"

    if user and user_id != "public":
        count_data, err = increment_query_count(user_id)
        if err:
            msg = f"❌ Query error: {err}"
            send_whatsapp_reply(from_number, msg)
            return jsonify({"error": err}), 500
        try:
            if not count_data.get("was_successful", True):
                msg = f"❌ Query limit exceeded ({count_data.get('current_queries_used')}/{count_data.get('current_query_limit')})"
                send_whatsapp_reply(from_number, msg)
                return jsonify({"error": "Query limit exceeded"}), 429
        except Exception:
            logger.debug("RPC response did not follow expected shape; continuing")

    search_type = "exact" if request.args.get("searchType") == "exact" else "partial"
    results = search_all_tables(text, is_exact=(search_type == "exact"))
    total_matches = sum(r.get("count", 0) for r in results)

    reply_lines = [f"🔍 Sanction Check for \"{text}\":", f"Total Matches: {total_matches}"]
    for r in results:
        if r.get("count", 0) > 0:
            reply_lines.append(f"\nTable: {r.get('table')} ({r.get('count')} matches)")
            for m in r.get("matches", [])[:5]:
                reply_lines.append(f"  • {repr(m)[:300]}")
            if r.get("count", 0) > 5:
                reply_lines.append(f"  ...and {r.get('count') - 5} more")

    reply_text = "\n".join(reply_lines)
    success, send_err = send_whatsapp_reply(from_number, reply_text)
    if not success:
        logger.error("Failed to send reply: %s", send_err)
        return jsonify({"error": send_err}), 500

    return "OK", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "supabase_service_role_set": bool(SUPABASE_SERVICE_ROLE_KEY),
        "supabase_anon_set": bool(SUPABASE_ANON_KEY),
        "whatsapp_configured": bool(META_TOKEN and PHONE_NUMBER_ID),
        "time": datetime.utcnow().isoformat() + "Z"
    })


# --- MAIN ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Flask app on port %s", port)
    app.run(host="0.0.0.0", port=port)
