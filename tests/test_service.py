"""Spec 8 -- the two service surfaces, both driven off the stub."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from service.api import create_app
from service.cli import main
from service.context import build_context


@pytest.fixture
def ctx(tmp_path):
    c = build_context(engine="videa", stub=True, projects_dir=tmp_path)
    yield c
    asyncio.run(c.aclose())


@pytest.fixture
def client(tmp_path):
    app = create_app(engine="videa", stub=True, projects_dir=tmp_path)
    with TestClient(app) as tc:
        yield tc


# --- API ---------------------------------------------------------------------

def test_health_reports_producer_availability(client):
    body = client.get("/health").json()
    assert body["engine"] == "videa"
    assert body["llm"]["stub"] is True
    assert set(body["producers"]) == {"writer", "reasoner", "coder"}
    assert "musica" in body["engines_available"]


def test_project_crud(client):
    created = client.post("/projects", json={"name": "Lighthouse"})
    assert created.status_code == 201
    pid = created.json()["id"]

    assert client.get(f"/projects/{pid}").json()["name"] == "Lighthouse"
    assert any(p["id"] == pid for p in client.get("/projects").json())


def test_missing_project_is_404(client):
    assert client.get("/projects/nope").status_code == 404


def test_run_submission_returns_immediately_with_a_run_id(client):
    pid = client.post("/projects", json={"name": "P"}).json()["id"]
    accepted = client.post(f"/projects/{pid}/runs", json={"goal": "write the opening"})

    assert accepted.status_code == 202
    body = accepted.json()
    assert body["run_id"].startswith("run_")
    assert body["poll"] == f"/projects/{pid}/runs/{body['run_id']}"

    # The run id is pollable -- it is not a placeholder.
    for _ in range(200):
        detail = client.get(body["poll"])
        if detail.status_code == 200 and detail.json()["status"] not in ("pending", "running"):
            break
    assert detail.status_code == 200
    assert detail.json()["id"] == body["run_id"]
    assert detail.json()["status"] == "done"
    assert detail.json()["tasks"][0]["verdict"]["judged"] == "artifact"


def test_run_status_is_pollable_before_completion(client):
    pid = client.post("/projects", json={"name": "P"}).json()["id"]
    rid = client.post(f"/projects/{pid}/runs", json={"goal": "g"}).json()["run_id"]
    listed = client.get(f"/projects/{pid}/runs").json()
    assert [r["id"] for r in listed] == [rid]


def test_unknown_run_is_404(client):
    pid = client.post("/projects", json={"name": "P"}).json()["id"]
    assert client.get(f"/projects/{pid}/runs/run_nope").status_code == 404


def test_bible_endpoint_returns_entries_and_rendered_context(client):
    pid = client.post("/projects", json={"name": "P"}).json()["id"]
    poll = client.post(f"/projects/{pid}/runs", json={"goal": "write a lighthouse film"}).json()["poll"]
    for _ in range(200):
        if client.get(poll).json()["status"] == "done":
            break
    body = client.get(f"/projects/{pid}/bible").json()
    assert body["entries"]
    assert "lighthouse" in body["context"]


def test_empty_goal_is_rejected(client):
    pid = client.post("/projects", json={"name": "P"}).json()["id"]
    assert client.post(f"/projects/{pid}/runs", json={"goal": ""}).status_code == 422


def test_app_uses_lifespan_not_deprecated_event_handlers(tmp_path):
    """Structural, not a source grep: no startup/shutdown handlers are
    registered, and a lifespan context is."""
    app = create_app(engine="videa", stub=True, projects_dir=tmp_path)
    assert app.router.on_startup == []
    assert app.router.on_shutdown == []
    assert app.router.lifespan_context is not None


# --- CLI ---------------------------------------------------------------------

def test_cli_health(tmp_path, capsys):
    code = main(["--stub", "--projects-dir", str(tmp_path), "health"])
    out = capsys.readouterr().out
    assert code == 0
    assert "videa" in out and "stub" in out
    assert "writer" in out


def test_cli_health_json(tmp_path, capsys):
    main(["--stub", "--json", "--projects-dir", str(tmp_path), "health"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["stub"] is True and payload["engine"] == "videa"


def test_cli_run_creates_a_project_and_reports_verdicts(tmp_path, capsys):
    code = main(["--stub", "--projects-dir", str(tmp_path), "run", "write the opening scene"])
    out = capsys.readouterr().out
    assert code == 0
    assert "[ok]" in out
    assert "judged:" in out
    assert "1 done, 0 failed, 0 deferred" in out


def test_cli_run_json_output(tmp_path, capsys):
    main(["--stub", "--json", "--projects-dir", str(tmp_path), "run", "write it"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "done"
    assert payload["tasks"][0]["verdict"]["judged"] == "artifact"


def test_cli_projects_and_show(tmp_path, capsys):
    main(["--stub", "--json", "--projects-dir", str(tmp_path), "run", "write it"])
    run_id = json.loads(capsys.readouterr().out)["id"]

    main(["--stub", "--json", "--projects-dir", str(tmp_path), "projects"])
    projects = json.loads(capsys.readouterr().out)
    assert len(projects) == 1
    pid = projects[0]["id"]

    code = main(["--stub", "--projects-dir", str(tmp_path), "show", run_id, "--project", pid])
    assert code == 0
    assert run_id in capsys.readouterr().out


def test_cli_bible(tmp_path, capsys):
    main(["--stub", "--json", "--projects-dir", str(tmp_path), "run", "a lighthouse film"])
    capsys.readouterr()
    main(["--stub", "--json", "--projects-dir", str(tmp_path), "projects"])
    pid = json.loads(capsys.readouterr().out)[0]["id"]

    main(["--stub", "--projects-dir", str(tmp_path), "bible", "--project", pid])
    assert "lighthouse" in capsys.readouterr().out


def test_cli_show_unknown_run_is_an_error(tmp_path, capsys):
    main(["--stub", "--json", "--projects-dir", str(tmp_path), "run", "write it"])
    capsys.readouterr()
    main(["--stub", "--json", "--projects-dir", str(tmp_path), "projects"])
    pid = json.loads(capsys.readouterr().out)[0]["id"]
    assert main(["--stub", "--projects-dir", str(tmp_path), "show", "nope", "--project", pid]) == 1


def test_cli_unknown_engine_is_an_error(tmp_path, capsys):
    assert main(["--engine", "fabrica", "--stub", "--projects-dir", str(tmp_path), "health"]) == 1
    assert "unknown engine" in capsys.readouterr().err


def test_cli_engine_selection(tmp_path, capsys):
    main(["--engine", "musica", "--stub", "--json", "--projects-dir", str(tmp_path), "health"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "musica"
    assert set(payload["producers"]) == {"composer", "arranger", "mixer"}


def test_stub_is_never_selected_implicitly(tmp_path, capsys):
    """Spec 5.4: without --stub the CLI talks to a real host and says so when
    it cannot reach one."""
    code = main(["--projects-dir", str(tmp_path), "--json",
                 "--config", _no_host_config(tmp_path), "health"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["stub"] is False
    assert payload["llm_client"] == "ollama"
    assert payload["llm_reachable"] is False
    assert code == 1   # honest non-zero, not a silent stub success


def _no_host_config(tmp_path) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"llm": {"host": "http://127.0.0.1:9"}}))
    return str(path)
