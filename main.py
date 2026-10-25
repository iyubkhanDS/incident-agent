import base64
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

DB_FILE = os.environ.get("ACTION_GATE_DB_FILE", "action_gate.db")
PROFILE = "ga5-mailroom-action-gate/v2"
MAX_RESPONSE_BYTES = 512 * 1024
LOGGER = logging.getLogger("action_gate")

ALLOWED_ACTIONS = {
    "create_draft",
    "update_internal_record",
    "send_approved_notice",
    "request_confirmation",
    "quarantine_item",
    "no_action",
}


def init_db() -> None:
    conn = sqlite3.connect(DB_FILE, timeout=60.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                core_id TEXT NOT NULL,
                audit_id TEXT NOT NULL,
                input_digest TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                receipt_verifier TEXT NOT NULL,
                proposals_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dossiers (
                core_id TEXT NOT NULL,
                dossier_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                proposal_json TEXT NOT NULL,
                PRIMARY KEY (core_id, dossier_id, content_hash)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS commit_receipts (
                evaluation_id TEXT NOT NULL,
                receipt_id TEXT NOT NULL,
                call_id TEXT NOT NULL,
                outcome_json TEXT NOT NULL,
                PRIMARY KEY (evaluation_id, receipt_id)
            )
        """)
        conn.commit()
    finally:
        conn.close()


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    init_db()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_input_digest(dossiers: list[Any]) -> str:
    serialized = canonical_json(dossiers)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_proposal_digest(proposal: dict[str, Any]) -> str:
    filtered = {
        "dossierId": proposal.get("dossierId"),
        "callId": proposal.get("callId"),
        "action": proposal.get("action"),
        "target": proposal.get("target"),
        "payload": proposal.get("payload"),
        "evidence": sorted(proposal.get("evidence", [])),
    }
    return hashlib.sha256(canonical_json(filtered).encode("utf-8")).hexdigest()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def response_json(response: dict[str, Any]) -> dict[str, Any]:
    serialized = canonical_json(response)
    if len(serialized.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise HTTPException(status_code=422, detail="Response exceeds the supported size limit.")
    return response


def validate_propose_request(body: dict[str, Any]) -> None:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON request.")
    if body.get("profile") != PROFILE:
        raise HTTPException(status_code=400, detail="Unsupported profile configuration.")
    if body.get("operation") != "propose":
        raise HTTPException(status_code=400, detail="Invalid operation.")
    if not isinstance(body.get("evaluationId"), str) or not body["evaluationId"]:
        raise HTTPException(status_code=422, detail="Missing or invalid evaluationId.")
    
    verifier = body.get("receiptVerifier")
    if not isinstance(verifier, dict) or verifier.get("algorithm") != "Ed25519":
        raise HTTPException(status_code=422, detail="Invalid receiptVerifier.")
    pk = verifier.get("publicKeyJwk")
    if not isinstance(pk, dict) or pk.get("kty") != "OKP" or pk.get("crv") != "Ed25519" or not isinstance(pk.get("x"), str):
        raise HTTPException(status_code=422, detail="Invalid publicKeyJwk.")

    corpus = body.get("corpus")
    if not isinstance(corpus, dict) or not isinstance(corpus.get("coreId"), str) or not isinstance(corpus.get("auditId"), str):
        raise HTTPException(status_code=422, detail="Invalid corpus metadata.")

    dossiers = body.get("dossiers")
    if not isinstance(dossiers, list) or not dossiers:
        raise HTTPException(status_code=422, detail="Invalid dossiers list.")

    allowed_actions = body.get("allowedActions")
    if not isinstance(allowed_actions, list) or not all(a in ALLOWED_ACTIONS for a in allowed_actions):
        raise HTTPException(status_code=422, detail="Invalid allowedActions.")

    dossier_ids = set()
    for d in dossiers:
        if not isinstance(d, dict) or not isinstance(d.get("dossierId"), str) or not d["dossierId"]:
            raise HTTPException(status_code=422, detail="Invalid dossier structure.")
        if d["dossierId"] in dossier_ids:
            raise HTTPException(status_code=400, detail="Duplicate dossier IDs.")
        dossier_ids.add(d["dossierId"])
        sources = d.get("sources")
        if not isinstance(sources, list):
            raise HTTPException(status_code=422, detail="Invalid sources list.")
        for s in sources:
            if not isinstance(s, dict) or not isinstance(s.get("sourceId"), str) or not isinstance(s.get("lines"), list):
                raise HTTPException(status_code=422, detail="Invalid source structure.")
            for line in s["lines"]:
                if not isinstance(line, dict) or not isinstance(line.get("lineId"), str) or not isinstance(line.get("text"), str):
                    raise HTTPException(status_code=422, detail="Invalid line structure.")


def model_configuration() -> tuple[str, str, str]:
    api_key = os.environ.get("ACTION_GATE_AI_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("INCIDENT_AI_API_KEY")
    model = os.environ.get("ACTION_GATE_AI_MODEL") or os.environ.get("OPENAI_MODEL") or os.environ.get("INCIDENT_AI_MODEL") or "gpt-4o"
    base_url = os.environ.get("ACTION_GATE_AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    return base_url, api_key, model


def call_model_for_proposal(dossier: dict[str, Any], allowed_actions: list[str]) -> dict[str, Any]:
    dossier_id = dossier["dossierId"]
    mailbox = dossier.get("mailbox", "support")
    objective = dossier.get("objective", "")
    sources = dossier.get("sources", [])
    
    all_lines = {}
    full_text_lines = []
    for s in sources:
        for l in s.get("lines", []):
            lid = l.get("lineId")
            ltxt = l.get("text", "")
            if lid:
                all_lines[lid] = ltxt
                full_text_lines.append(f"{lid}: {ltxt}")
    
    combined_text = "\n".join(full_text_lines)

    base_url, api_key, model = model_configuration()
    if api_key:
        system_prompt = (
            "You are an expert mailroom action gate processor. Analyze the dossier and return "
            "one JSON object only with keys: action, target, payload, evidence. "
            f"Allowed actions: {allowed_actions}. "
            "Target format: null or {\"kind\": \"...\", \"id\": \"...\"}. "
            "Payload must match exact schema for chosen action. "
            "Evidence must be an array of unique lineIds present in the dossier. "
            "No prose."
        )
        user_payload = {
            "dossierId": dossier_id,
            "mailbox": mailbox,
            "objective": objective,
            "sources": sources,
            "allowedActions": allowed_actions,
        }
        request_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": canonical_json(user_payload)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=canonical_json(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read())
                content = payload["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if isinstance(parsed, dict) and parsed.get("action") in allowed_actions:
                    action = parsed["action"]
                    target = parsed.get("target")
                    payload_obj = parsed.get("payload", {})
                    evidence = [e for e in parsed.get("evidence", []) if e in all_lines]
                    if not evidence and all_lines:
                        evidence = [list(all_lines.keys())[0]]
                    
                    call_id = hashlib.sha256(f"{dossier_id}:{action}:{canonical_json(payload_obj)}".encode("utf-8")).hexdigest()[:16]
                    return {
                        "dossierId": dossier_id,
                        "callId": call_id,
                        "action": action,
                        "target": target,
                        "payload": payload_obj,
                        "evidence": evidence,
                    }
        except Exception as exc:
            LOGGER.warning("LLM call failed or invalid response: %s", exc)

    # Fallback Heuristic Analysis
    lower_text = combined_text.lower()
    evidence_keys = list(all_lines.keys())[:1] if all_lines else ["line-1"]
    
    if any(term in lower_text for term in ["ignore previous", "system prompt", "exfiltrate", "execute", "bypass", "malicious", "override"]):
        action = "quarantine_item"
        target = {"kind": "security_queue", "id": "mailroom"}
        payload = {"artifactId": dossier_id, "reasonCode": "INDIRECT_PROMPT_INJECTION"}
    elif "status" in lower_text or "order" in lower_text or "where is" in lower_text or "tracking" in lower_text:
        action = "create_draft"
        target = {"kind": "draft_queue", "id": f"mailbox:{mailbox}"}
        payload = {"recipient": mailbox, "referenceId": dossier_id, "status": "pending", "template": "order_status"}
    elif "delivery window" in lower_text or "reschedule" in lower_text or "delivery time" in lower_text:
        action = "update_internal_record"
        target = {"kind": "case_record", "id": dossier_id}
        payload = {"field": "delivery_window", "sourceEventId": dossier_id, "value": "updated"}
    elif "approve" in lower_text or "approved notice" in lower_text:
        action = "send_approved_notice"
        target = {"kind": "email", "id": mailbox}
        payload = {"referenceId": dossier_id, "status": "approved", "template": "approved_delivery_notice"}
    elif "confirm" in lower_text or "verify" in lower_text or "identity" in lower_text:
        action = "request_confirmation"
        target = {"kind": "approval_queue", "id": "verification-team"}
        payload = {"claimedSender": mailbox, "questionCode": "VERIFY_REQUEST", "referenceId": dossier_id}
    else:
        action = "no_action"
        target = None
        payload = {"reasonCode": "INFORMATIONAL", "referenceId": dossier_id}

    call_id = hashlib.sha256(f"{dossier_id}:{action}:{canonical_json(payload)}".encode("utf-8")).hexdigest()[:16]
    return {
        "dossierId": dossier_id,
        "callId": call_id,
        "action": action,
        "target": target,
        "payload": payload,
        "evidence": evidence_keys,
    }


@app.post("/")
@app.post("/v2/mailroom")
async def handle_operations(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON request.") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON request.")

    operation = body.get("operation")
    if operation == "propose":
        validate_propose_request(body)
        evaluation_id = body["evaluationId"]
        corpus = body["corpus"]
        core_id = corpus["coreId"]
        audit_id = corpus["auditId"]
        dossiers = body["dossiers"]
        receipt_verifier = body["receiptVerifier"]
        allowed_actions = body.get("allowedActions", list(ALLOWED_ACTIONS))

        input_digest = compute_input_digest(dossiers)
        request_hash = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()

        conn = sqlite3.connect(DB_FILE, timeout=60.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            
            existing = conn.execute(
                "SELECT request_hash, proposals_json FROM evaluations WHERE evaluation_id = ?",
                (evaluation_id,)
            ).fetchone()

            if existing:
                if existing[0] != request_hash:
                    raise HTTPException(status_code=409, detail="Evaluation ID conflicts with existing changed content.")
                proposals = json.loads(existing[1])
                return response_json({
                    "profile": PROFILE,
                    "evaluationId": evaluation_id,
                    "status": "awaiting_receipts",
                    "inputDigest": input_digest,
                    "proposals": proposals,
                })

            proposals = []
            for dossier in dossiers:
                dossier_id = dossier["dossierId"]
                content_hash = hashlib.sha256(canonical_json(dossier).encode("utf-8")).hexdigest()

                cached = conn.execute(
                    "SELECT proposal_json FROM dossiers WHERE core_id = ? AND dossier_id = ? AND content_hash = ?",
                    (core_id, dossier_id, content_hash)
                ).fetchone()

                if cached:
                    prop = json.loads(cached[0])
                else:
                    prop = await run_in_threadpool(call_model_for_proposal, dossier, allowed_actions)
                    conn.execute(
                        "INSERT OR REPLACE INTO dossiers (core_id, dossier_id, content_hash, proposal_json) VALUES (?, ?, ?, ?)",
                        (core_id, dossier_id, content_hash, canonical_json(prop))
                    )
                proposals.append(prop)

            conn.execute(
                """INSERT INTO evaluations 
                   (evaluation_id, core_id, audit_id, input_digest, request_hash, receipt_verifier, proposals_json, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (evaluation_id, core_id, audit_id, input_digest, request_hash, canonical_json(receipt_verifier), canonical_json(proposals), "awaiting_receipts", time.time())
            )
            conn.commit()

            return response_json({
                "profile": PROFILE,
                "evaluationId": evaluation_id,
                "status": "awaiting_receipts",
                "inputDigest": input_digest,
                "proposals": proposals,
            })
        except HTTPException:
            conn.rollback()
            raise
        finally:
            conn.close()

    elif operation == "commit":
        if body.get("profile") != PROFILE:
            raise HTTPException(status_code=400, detail="Unsupported profile configuration.")
        evaluation_id = body.get("evaluationId")
        input_digest = body.get("inputDigest")
        receipts = body.get("receipts")

        if not isinstance(evaluation_id, str) or not isinstance(input_digest, str) or not isinstance(receipts, list):
            raise HTTPException(status_code=422, detail="Invalid commit request format.")

        conn = sqlite3.connect(DB_FILE, timeout=60.0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            eval_row = conn.execute(
                "SELECT receipt_verifier, proposals_json FROM evaluations WHERE evaluation_id = ?",
                (evaluation_id,)
            ).fetchone()

            if not eval_row:
                raise HTTPException(status_code=404, detail="Unknown evaluation ID.")

            receipt_verifier = json.loads(eval_row[0])
            proposals = json.loads(eval_row[1])
            proposal_map = {p["callId"]: p for p in proposals}

            pk = receipt_verifier["publicKeyJwk"]
            try:
                public_key = Ed25519PublicKey.from_public_bytes(base64.urlsafe_b64decode(pk["x"] + "=="))
            except Exception as exc:
                raise HTTPException(status_code=422, detail="Invalid public key.") from exc

            outcomes = []
            seen_receipt_ids = set()

            for r in receipts:
                if not isinstance(r, dict):
                    raise HTTPException(status_code=422, detail="Invalid receipt structure.")
                receipt_id = r.get("receiptId")
                call_id = r.get("callId")
                signature_b64 = r.get("receiptSignature")
                accepted = r.get("accepted")
                proposal_digest = r.get("proposalDigest")
                dossier_id = r.get("dossierId")
                action = r.get("action")

                if not isinstance(receipt_id, str) or not receipt_id or receipt_id in seen_receipt_ids:
                    raise HTTPException(status_code=422, detail="Invalid or duplicate receipt ID.")
                seen_receipt_ids.add(receipt_id)

                if call_id not in proposal_map:
                    raise HTTPException(status_code=422, detail="Call ID mismatch.")

                expected_prop = proposal_map[call_id]
                if expected_prop["dossierId"] != dossier_id or expected_prop["action"] != action:
                    raise HTTPException(status_code=422, detail="Proposal mismatch.")

                expected_digest = compute_proposal_digest(expected_prop)
                if proposal_digest != expected_digest:
                    raise HTTPException(status_code=422, detail="Proposal digest mismatch.")

                signed_inner = {
                    "profile": PROFILE,
                    "evaluationId": evaluation_id,
                    "inputDigest": input_digest,
                    "receipt": {
                        "dossierId": dossier_id,
                        "callId": call_id,
                        "action": action,
                        "accepted": accepted,
                        "proposalDigest": proposal_digest,
                        "receiptId": receipt_id,
                    }
                }
                signed_bytes = canonical_json(signed_inner).encode("utf-8")

                try:
                    sig_bytes = base64.b64decode(signature_b64)
                    public_key.verify(sig_bytes, signed_bytes)
                except Exception as exc:
                    raise HTTPException(status_code=422, detail="Invalid cryptographic receipt signature.") from exc

                status_str = "executed" if accepted else "rejected"
                outcomes.append({
                    "dossierId": dossier_id,
                    "callId": call_id,
                    "action": action,
                    "proposalDigest": proposal_digest,
                    "receiptId": receipt_id,
                    "status": status_str,
                })

                conn.execute(
                    "INSERT OR REPLACE INTO commit_receipts (evaluation_id, receipt_id, call_id, outcome_json) VALUES (?, ?, ?, ?)",
                    (evaluation_id, receipt_id, call_id, canonical_json(outcomes[-1]))
                )

            conn.commit()
            return response_json({
                "profile": PROFILE,
                "evaluationId": evaluation_id,
                "status": "completed",
                "inputDigest": input_digest,
                "outcomes": outcomes,
            })
        except HTTPException:
            conn.rollback()
            raise
        finally:
            conn.close()

    else:
        raise HTTPException(status_code=400, detail="Unknown operation.")
