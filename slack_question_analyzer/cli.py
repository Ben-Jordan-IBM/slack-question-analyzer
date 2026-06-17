"""
Command-line interface for the Slack Question Analyzer.
"""

import sys
import logging
import click
from pathlib import Path
from .analyzer import QuestionAnalyzer
from .inputs import load_input_files


@click.group()
@click.version_option(version='2.40.1')
@click.option('--verbose', '-v', is_flag=True, help='Show debug-level logs')
def cli(verbose):
    """
    Slack Question Analyzer - AI-powered question grouping and ranking.

    Analyzes Slack questions and groups similar ones together using AI embeddings.
    Supports Ollama (local), Azure OpenAI, and standard OpenAI.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(message)s')


@cli.command()
@click.argument('input_files', type=click.Path(exists=True), nargs=-1, required=True)
@click.option('--output', '-o', type=click.Path(),
              help='Output file path (.json, .csv, or .md — format inferred from extension)')
@click.option('--provider', '-p', type=click.Choice(['ollama', 'azure', 'openai']),
              help='AI provider to use (default: from .env or ollama)')
@click.option('--threshold', '-t', type=click.FloatRange(0.0, 1.0),
              help='Similarity threshold (0-1)')
@click.option('--no-summary', is_flag=True, help='Skip printing summary to console')
@click.option('--no-cache', is_flag=True, help='Disable the persistent embedding cache')
@click.option('--no-labels', is_flag=True, help='Skip LLM-generated topic labels')
def analyze(input_files, output, provider, threshold, no_summary, no_cache, no_labels):
    """
    Analyze questions from one or more Slack content files.

    INPUT_FILES: .json/.txt/.csv files and/or .zip archives (e.g. a zipped
    Slack export); everything is merged and analyzed as a single corpus.

    Examples:
        slack-analyzer analyze slack_content.txt -o results.json
        slack-analyzer analyze slack-export.zip -o report.md
        slack-analyzer analyze week1.json week2.json -o combined.md
    """
    try:
        # Initialize analyzer
        click.echo(f"Initializing analyzer with provider: {provider or 'from .env (default: ollama)'}")
        analyzer = QuestionAnalyzer(provider=provider, use_disk_cache=not no_cache,
                                    threshold=threshold,
                                    label_groups=False if no_labels else None)

        # Set default output path if not provided
        if not output:
            input_path = Path(input_files[0])
            output = input_path.parent / f"{input_path.stem}_analysis.json"

        # Run analysis
        click.echo(f"\nAnalyzing: {', '.join(input_files)}")
        contents = load_input_files(input_files)
        results = analyzer.analyze_contents(contents)
        analyzer.save_results(results, str(output))

        # Print summary unless disabled
        if not no_summary:
            analyzer.print_summary(results)

        click.echo(f"\nAnalysis complete! Results saved to: {output}")

    except FileNotFoundError as e:
        click.echo(f"Error: File not found - {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def setup():
    """
    Setup wizard to configure the analyzer.

    Creates a .env file with your preferred AI provider settings.
    """
    click.echo("=== Slack Question Analyzer Setup ===\n")

    # Choose provider
    provider = click.prompt(
        'Choose AI provider',
        type=click.Choice(['ollama', 'azure', 'openai']),
        default='ollama'
    )

    env_content = [
        f"AI_PROVIDER={provider}",
        "",
        "# Similarity threshold: unset = auto (recommended); a number pins it",
        "# SIMILARITY_THRESHOLD=0.85",
        ""
    ]

    if provider == 'ollama':
        from .model_defaults import default_generation_model, total_ram_gb
        click.echo("\nOllama Configuration (Local & Free)")
        ollama_url = click.prompt('Ollama URL', default='http://localhost:11434')
        ollama_model = click.prompt('Embedding model', default='nomic-embed-text')
        ram = total_ram_gb()
        if ram:
            click.echo(f"Detected {ram:.0f}GB RAM — suggesting a chat model sized for this machine.")
        generation_model = click.prompt(
            'Chat model for LLM features (topic labels, summaries, etc.)',
            default=default_generation_model())

        env_content.extend([
            "# Ollama Configuration",
            f"OLLAMA_URL={ollama_url}",
            f"OLLAMA_MODEL={ollama_model}",
            f"OLLAMA_GENERATION_MODEL={generation_model}",
        ])

        click.echo("\nMake sure Ollama is running and the models are pulled:")
        click.echo(f"   ollama pull {ollama_model}")
        click.echo(f"   ollama pull {generation_model}")

    elif provider == 'azure':
        click.echo("\nAzure OpenAI Configuration")
        api_key = click.prompt('Azure OpenAI API Key', hide_input=True)
        endpoint = click.prompt('Azure OpenAI Endpoint')
        deployment = click.prompt('Embedding Deployment Name')
        chat_deployment = click.prompt(
            'Chat Deployment Name for LLM features (leave empty to skip)',
            default='', show_default=False)
        api_version = click.prompt('API Version', default='2024-02-15-preview')

        env_content.extend([
            "# Azure OpenAI Configuration",
            f"AZURE_OPENAI_API_KEY={api_key}",
            f"AZURE_OPENAI_ENDPOINT={endpoint}",
            f"AZURE_OPENAI_DEPLOYMENT_NAME={deployment}",
            f"AZURE_OPENAI_API_VERSION={api_version}",
            "EMBEDDING_MODEL=text-embedding-ada-002",
        ])
        if chat_deployment:
            env_content.append(f"AZURE_OPENAI_CHAT_DEPLOYMENT={chat_deployment}")

    else:  # openai
        click.echo("\nOpenAI Configuration")
        api_key = click.prompt('OpenAI API Key', hide_input=True)
        chat_model = click.prompt('Chat model for LLM features', default='gpt-4o-mini')

        env_content.extend([
            "# OpenAI Configuration",
            f"OPENAI_API_KEY={api_key}",
            "EMBEDDING_MODEL=text-embedding-ada-002",
            f"CHAT_MODEL={chat_model}",
        ])

    env_content.extend([
        "",
        "# Optional LLM features: 'auto' (when the model is available), 'on', or 'off'",
        "GROUP_LABELS=auto",
        "LLM_VERIFY_GROUPS=auto",
        "LLM_EXTRACTION=auto",
        "LLM_ANSWER_DETECTION=auto",
        "EXECUTIVE_SUMMARY=auto",
    ])

    # Write .env file (UTF-8 explicitly: Windows defaults to a legacy codepage)
    env_path = Path('.env')
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(env_content))

    click.echo(f"\nConfiguration saved to {env_path}")
    click.echo("\nYou're all set! Run 'slack-analyzer analyze <input_file>' to start analyzing.")


@cli.command()
def doctor():
    """
    Check that everything needed for analysis is installed and reachable.

    Run this on a new machine (and send the output when asking for help).
    """
    import os
    import requests
    from dotenv import load_dotenv
    load_dotenv()

    failures = 0

    def check(ok, label, fix=''):
        nonlocal failures
        mark = 'OK  ' if ok else 'FAIL'
        click.echo(f"[{mark}] {label}")
        if not ok:
            failures += 1
            if fix:
                click.echo(f"       fix: {fix}")

    check(sys.version_info >= (3, 10),
          f"Python {sys.version_info.major}.{sys.version_info.minor} (need 3.10+)",
          'Install Python 3.10 or newer from https://python.org')

    try:
        import numpy   # noqa: F401  (import probe: is the dependency present?)
        import sklearn  # noqa: F401
        import flask    # noqa: F401
        check(True, 'Python dependencies installed')
    except ImportError as e:
        check(False, f'Python dependencies ({e.name} missing)',
              'pip install -e .')

    provider = os.getenv('AI_PROVIDER', 'ollama')
    click.echo(f"[INFO] Provider: {provider}")

    if provider == 'ollama':
        from .model_defaults import default_generation_model, FALLBACK_GENERATION_MODEL
        url = os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')
        embed_model = os.getenv('OLLAMA_MODEL', 'nomic-embed-text')
        gen_model = default_generation_model()
        try:
            names = [m.get('name', '') for m in
                     requests.get(f"{url}/api/tags", timeout=3).json().get('models', [])]
            check(True, f'Ollama reachable at {url}')
            has = lambda m: any(n == m or n.startswith(m + ':') for n in names)  # noqa: E731
            check(has(embed_model), f"Embedding model '{embed_model}' downloaded",
                  f'ollama pull {embed_model}')
            if has(gen_model):
                check(True, f"Chat model '{gen_model}' downloaded (topic labels enabled)")
            elif (not os.getenv('OLLAMA_GENERATION_MODEL')
                  and has(FALLBACK_GENERATION_MODEL)):
                check(True, f"Chat model '{FALLBACK_GENERATION_MODEL}' downloaded "
                            f"(used instead of '{gen_model}', which isn't pulled)")
            else:
                click.echo(f"[WARN] Chat model '{gen_model}' not downloaded — topic "
                           f"labels/summaries will fall back to keywords")
                click.echo(f"       fix: ollama pull {gen_model}")
        except requests.RequestException:
            check(False, f'Ollama reachable at {url}',
                  'Install from https://ollama.com/download, then start it (ollama serve)')
    elif provider == 'azure':
        check(bool(os.getenv('AZURE_OPENAI_API_KEY') and os.getenv('AZURE_OPENAI_ENDPOINT')),
              'Azure OpenAI credentials configured',
              'Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env')
    else:
        check(bool(os.getenv('OPENAI_API_KEY')), 'OpenAI API key configured',
              'Set OPENAI_API_KEY in .env')

    if not os.getenv('DOMAIN_CONTEXT'):
        click.echo("[TIP ] Set DOMAIN_CONTEXT in .env (e.g. 'a webMethods MFT support "
                   "Slack channel') — it makes AI topic names noticeably sharper")

    try:
        cache_dir = Path(os.getenv('EMBEDDING_CACHE_DIR', '.embedding_cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe = cache_dir / '.doctor-probe'
        probe.write_text('ok', encoding='utf-8')
        probe.unlink()
        check(True, 'Working directory is writable (caches, analyses)')
    except OSError:
        check(False, 'Working directory is writable',
              'Run from a folder you can write to')

    click.echo()
    if failures:
        click.echo(f"{failures} problem(s) found — fix the items above and re-run "
                   f"'slack-analyzer doctor'.")
        sys.exit(1)
    click.echo("All good! Start the app with: python api_server.py")


@cli.command(name='eval')
@click.argument('fixture_files', type=click.Path(exists=True), nargs=-1)
@click.option('--provider', '-p', type=click.Choice(['ollama', 'azure', 'openai']),
              help='AI provider to use (default: from .env or ollama)')
def eval_fixture(fixture_files, provider):
    """
    Score the pipeline against labeled regression fixtures.

    With no arguments, runs EVERY fixture in fixtures/. Run after each
    prompt, anchor, or threshold change — question fixtures score routing
    and grouping; transcript fixtures run the full pipeline end-to-end
    (extraction included) against an answer key. Measurable, not guessed.
    """
    from .evaluation import (load_fixture, evaluate, format_report,
                             evaluate_transcript, format_transcript_report)

    if not fixture_files:
        fixture_files = sorted(str(p) for p in Path('fixtures').glob('*.json'))
        if not fixture_files:
            click.echo('No fixtures found in fixtures/', err=True)
            sys.exit(2)

    analyzer = QuestionAnalyzer(provider=provider, use_disk_cache=True)
    any_failed = False
    for i, fixture_file in enumerate(fixture_files):
        if i:
            click.echo('')
        try:
            fixture = load_fixture(fixture_file)
            click.echo(f"=== Evaluating {fixture_file} ===")
            if fixture.get('type') == 'transcript':
                result = evaluate_transcript(analyzer, fixture)
                click.echo(format_transcript_report(result))
                if result['failed']:
                    any_failed = True
            else:
                result = evaluate(analyzer, fixture)
                click.echo(format_report(result))
                if (result['routing_mismatches'] or result['missed_pairs']
                        or result['wrong_pairs']
                        or result.get('integrity_violations')):
                    any_failed = True
        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(2)
    if any_failed:
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
def validate(input_file):
    """
    Validate input file format and show statistics.

    INPUT_FILE: Path to file containing Slack messages/questions
    """
    try:
        from .question_extractor import QuestionExtractor

        click.echo(f"Validating: {input_file}\n")

        extractor = QuestionExtractor()
        questions = []
        for content in load_input_files([input_file]):
            questions.extend(extractor.parse_slack_content(content))

        click.echo("File is valid!")
        click.echo("\nStatistics:")
        click.echo(f"  Total questions found: {len(questions)}")

        if questions:
            click.echo("\nSample questions:")
            for i, q in enumerate(questions[:5], 1):
                click.echo(f"  {i}. {q['text'][:80]}...")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
