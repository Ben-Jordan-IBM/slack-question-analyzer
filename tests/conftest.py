import pytest


@pytest.fixture(autouse=True)
def isolated_topic_bank(tmp_path, monkeypatch):
    """Every test gets its own topic bank — never the repo's real one.

    Seeding is pointed at a nonexistent file so tests with mocked embedding
    providers don't try to embed the 150 real seed questions.
    """
    monkeypatch.setenv('TOPIC_BANK_PATH', str(tmp_path / 'topic_bank.json'))
    monkeypatch.setenv('SEED_TOPICS_PATH', str(tmp_path / 'no_seeds.json'))
    # The repo ships a real taxonomy.json; tests opt in to routing explicitly
    monkeypatch.setenv('TAXONOMY_PATH', str(tmp_path / 'no_taxonomy.json'))
    # No Ollama in the test environment: the fail-fast embedding probe
    # would spend real network retries before every LLM-extraction test
    monkeypatch.setenv('EMBEDDING_PREFLIGHT', 'off')


@pytest.fixture(autouse=True)
def pinned_generation_model(monkeypatch):
    """The default chat model depends on the host's RAM — pin it so tests
    behave identically on every machine. Tests of the sizing logic itself
    delete this env var."""
    monkeypatch.setenv('OLLAMA_GENERATION_MODEL', 'llama3.2')


@pytest.fixture(autouse=True)
def no_chat_backoff(monkeypatch):
    """Chat retries must not sleep between attempts in tests."""
    from slack_question_analyzer.group_labeler import GroupLabeler
    monkeypatch.setattr(GroupLabeler, 'CHAT_BACKOFF_SECONDS', 0)
