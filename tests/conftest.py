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


@pytest.fixture(autouse=True)
def pinned_generation_model(monkeypatch):
    """The default chat model depends on the host's RAM — pin it so tests
    behave identically on every machine. Tests of the sizing logic itself
    delete this env var."""
    monkeypatch.setenv('OLLAMA_GENERATION_MODEL', 'llama3.2')
