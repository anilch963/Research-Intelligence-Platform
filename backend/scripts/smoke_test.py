"""End-to-end smoke test against a running backend.

    uv run uvicorn app.main:app --port 8000      # terminal 1
    uv run python scripts/smoke_test.py          # terminal 2
"""

import sys

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
QUESTION = "What does error code TL-1042 mean?"


def main() -> int:
    with httpx.Client(base_url=BASE_URL, timeout=60) as client:
        health = client.get("/api/health").raise_for_status().json()
        print(f"health: {health}")
        if health["chunks_indexed"] == 0:
            print("FAIL: index is empty - run `uv run python -m app.ingest`")
            return 1

        for mode in ("vector", "hybrid"):
            resp = client.post("/api/query", json={"question": QUESTION, "retriever": mode})
            resp.raise_for_status()
            body = resp.json()
            print(f"\n[{mode}] {body['answer']}")
            print("  sources:", [s["source"] for s in body["sources"]])

        # CORS preflight as the Vite dev server would send it
        preflight = client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )
        allowed = preflight.headers.get("access-control-allow-origin")
        print(f"\nCORS allow-origin for :5173 -> {allowed}")
        if allowed != "http://localhost:5173":
            print("FAIL: CORS not configured for the frontend")
            return 1

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
