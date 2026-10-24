# Incident agent

## Run

Install dependencies and start the ASGI app:

```sh
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Configure an OpenAI-compatible chat-completions provider with `INCIDENT_AI_API_KEY`
and `INCIDENT_AI_MODEL`. `INCIDENT_AI_BASE_URL` defaults to
`https://api.openai.com/v1`; `INCIDENT_AI_TIMEOUT_SECONDS` defaults to `45`.
Provider HTTP errors and timeouts are logged without logging credentials,
incident prompts, or provider response bodies.
Set `INCIDENT_DB_FILE` to the persistent SQLite file path used by the host.

For public HTTPS, deploy behind a host or reverse proxy that terminates TLS and
forwards requests without redirecting the incident routes. The app does not
provision a public hostname or TLS certificate.

## Routes

- `POST /v2/incidents`
- `POST /v2/incidents/{runId}/receipts`
- `GET /v2/incidents/{runId}`
