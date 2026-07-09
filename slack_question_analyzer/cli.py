"""
Command-line interface for the Slack Question Analyzer.
"""

import json
import sys
import logging
import click
from pathlib import Path
from . import __version__
from .analyzer import QuestionAnalyzer
from .inputs import load_input_files, default_data_path

# Redirected Windows consoles default to cp1252, and Slack text is full of
# emoji — an encoding error must never kill a run (or doctor's own output)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(errors='replace')
        except (OSError, ValueError):
            pass


@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Show debug-level logs')
def cli(verbose):
    """
    Slack Question Analyzer - AI-powered question grouping and ranking.

    Analyzes Slack questions and groups similar ones together using AI
    embeddings — entirely on local Ollama: free, private, nothing leaves
    your machine.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(message)s')


@cli.command()
@click.argument('input_files', type=click.Path(exists=True), nargs=-1, required=True)
@click.option('--output', '-o', type=click.Path(),
              help='Output file path (.json, .csv, or .md — format inferred from extension)')
@click.option('--threshold', '-t', type=click.FloatRange(0.0, 1.0),
              help='Similarity threshold (0-1)')
@click.option('--no-summary', is_flag=True, help='Skip printing summary to console')
@click.option('--no-cache', is_flag=True, help='Disable the persistent embedding cache')
@click.option('--no-labels', is_flag=True, help='Skip LLM-generated topic labels')
def analyze(input_files, output, threshold, no_summary, no_cache, no_labels):
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
        click.echo("Initializing analyzer (local Ollama)")
        analyzer = QuestionAnalyzer(use_disk_cache=not no_cache,
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

    Creates a .env file with your local Ollama settings — everything runs
    on your machine; no data ever leaves it.
    """
    click.echo("=== Slack Question Analyzer Setup ===\n")

    env_content = [
        "# Similarity threshold: unset = auto (recommended); a number pins it",
        "# SIMILARITY_THRESHOLD=0.85",
        ""
    ]

    from .model_defaults import default_generation_model, total_ram_gb
    click.echo("Ollama Configuration (local & free)")
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

    # The pipeline's data files are CWD-relative: run from the wrong folder
    # and routing/seeds silently disable while everything "works". Say what
    # was checked, from where.
    click.echo(f"[INFO] Working directory: {Path.cwd()}")
    env_tax = os.getenv('TAXONOMY_PATH')
    tax_path = Path(env_tax) if env_tax else default_data_path('taxonomy.json')
    if tax_path.is_file():
        try:
            with open(tax_path, 'r', encoding='utf-8') as f:
                tax_data = json.load(f)
            # Shape errors must FAIL with a pointer, never crash the
            # diagnostic command they exist to be diagnosed by
            buckets = (tax_data.get('buckets', [])
                       if isinstance(tax_data, dict) else None)
            valid = bool(buckets) and isinstance(buckets, list) and all(
                isinstance(b, dict) and b.get('id') and b.get('name')
                and b.get('anchor') for b in buckets)
            check(valid, f'Taxonomy valid ({tax_path}, '
                         f'{len(buckets or [])} buckets)',
                  "the file must be an object with a 'buckets' list, and "
                  "every bucket needs 'id', 'name', and 'anchor'")
        except json.JSONDecodeError as e:
            check(False, f'Taxonomy valid ({tax_path})',
                  f'JSON error at line {e.lineno}, column {e.colno}: {e.msg} '
                  f'— routing is silently DISABLED until this parses')
    else:
        click.echo(f"[WARN] No taxonomy at {tax_path.resolve()} — the category "
                   f"funnel is off. If that's unexpected, run doctor from the "
                   f"slack-question-analyzer folder.")
    env_seeds = os.getenv('SEED_TOPICS_PATH')
    seeds_path = Path(env_seeds) if env_seeds else default_data_path('seed_topics.json')
    if seeds_path.is_file():
        try:
            with open(seeds_path, 'r', encoding='utf-8') as f:
                seeds = json.load(f)
            valid = isinstance(seeds, list) and all(
                isinstance(s, dict) and s.get('topic') and s.get('question')
                for s in seeds)
            count_label = len(seeds) if isinstance(seeds, list) else 0
            check(valid, f'Seed topics valid ({seeds_path}, {count_label} topics)',
                  "the file must be a list, and every seed needs 'topic' "
                  "and 'question'")
        except json.JSONDecodeError as e:
            check(False, f'Seed topics valid ({seeds_path})',
                  f'JSON error at line {e.lineno}, column {e.colno}: {e.msg}')

    provider = os.getenv('AI_PROVIDER', 'ollama').strip().lower()
    if provider != 'ollama':
        check(False, f"AI_PROVIDER '{provider}' is not supported",
              "This tool runs entirely on local Ollama — remove AI_PROVIDER "
              "from .env (cloud providers were dropped by design)")

    from .model_defaults import default_generation_model, FALLBACK_GENERATION_MODEL
    url = os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')
    embed_model = os.getenv('OLLAMA_MODEL', 'nomic-embed-text')
    gen_model = default_generation_model()
    try:
        names = [m.get('name', '') for m in
                 requests.get(f"{url}/api/tags", timeout=3).json().get('models', [])]
        check(True, f'Ollama reachable at {url}')
        try:
            version = requests.get(f"{url}/api/version",
                                   timeout=3).json().get('version', '')
            nums = tuple(int(p) for p in version.split('.')[:2])
            if len(nums) == 2 and nums < (0, 5):
                click.echo(f"[WARN] Ollama {version} is old — structured "
                           f"output needs 0.5+; update from ollama.com")
        except (requests.RequestException, ValueError):
            pass
        has = lambda m: any(n == m or n.startswith(m + ':') for n in names)  # noqa: E731
        check(has(embed_model), f"Embedding model '{embed_model}' downloaded",
              f'ollama pull {embed_model}')
        if has(embed_model):
            # A listed model can still be corrupt: prove it embeds
            click.echo('[....] Probing the embedding model (can take up '
                       'to a minute while it loads)...')
            try:
                r = requests.post(f"{url}/api/embeddings",
                                  json={'model': embed_model,
                                        'prompt': 'doctor probe'},
                                  timeout=60)
                r.raise_for_status()
                ok = bool(r.json().get('embedding'))
                check(ok, 'Embedding round-trip works',
                      f'ollama pull {embed_model} (re-download the model)')
            except requests.RequestException as e:
                check(False, 'Embedding round-trip works',
                      f'embedding request failed ({e}) — try: '
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
@click.option('--json', 'json_out', type=click.Path(), default=None,
              help='Also write the full per-check results to this JSON file '
                   '(usable later as a --compare baseline)')
@click.option('--compare', 'baseline_path',
              type=click.Path(exists=True), default=None,
              help='Diff this run against a saved --json baseline: prints '
                   'newly failing and newly passing checks')
@click.option('--runs', default=1, show_default=True,
              help='Repeat the whole suite N times and report per-check '
                   'flip rates (LLM nondeterminism). Combine with --no-cache '
                   'or flips will be hidden by cached responses.')
@click.option('--no-cache', is_flag=True,
              help='Bypass the disk caches (embeddings + LLM) for this run')
def eval_fixture(fixture_files, json_out, baseline_path, runs, no_cache):
    """
    Score the pipeline against labeled regression fixtures.

    With no arguments, runs EVERY fixture in fixtures/. Run after each
    prompt, anchor, or threshold change — question fixtures score routing
    and grouping; transcript fixtures run the full pipeline end-to-end
    (extraction included) against an answer key. Measurable, not guessed.

    A single fixture runs fast: slack-analyzer eval fixtures/mft_synthetic_7.json
    """
    from datetime import datetime, timezone
    from .evaluation import (load_fixture, evaluate, format_report,
                             evaluate_transcript, format_transcript_report,
                             checks_from_question_result, format_scoreboard,
                             diff_baseline, format_diff, flip_report,
                             format_flip_report)
    from . import __version__

    if not fixture_files:
        fixture_files = sorted(str(p) for p in Path('fixtures').glob('*.json'))
        if not fixture_files:
            click.echo('No fixtures found in fixtures/', err=True)
            sys.exit(2)

    import os as _os
    # --no-cache must bypass BOTH disk caches: use_disk_cache only reaches
    # the embedding cache, while the LLM cache is env-gated — leaving it on
    # would replay run 1's verdicts and make every --runs read "not flaky"
    saved_llm_cache = _os.environ.get('LLM_CACHE')
    if no_cache:
        _os.environ['LLM_CACHE'] = 'off'
    try:
        any_error = False
        run_entries = []
        for run in range(max(1, runs)):
            first_run = run == 0
            if not first_run:
                click.echo(f"\n=== Run {run + 1}/{runs} ===")
            # A fresh analyzer per run: grouping must not carry state
            # (threshold, caches) from one run into the next
            analyzer = QuestionAnalyzer(use_disk_cache=not no_cache)
            entries = []
            for i, fixture_file in enumerate(fixture_files):
                if i and first_run:
                    click.echo('')
                try:
                    fixture = load_fixture(fixture_file)
                    if first_run:
                        click.echo(f"=== Evaluating {fixture_file} ===")
                    if fixture.get('type') == 'transcript':
                        result = evaluate_transcript(analyzer, fixture)
                        if first_run:
                            click.echo(format_transcript_report(result))
                        checks = result['checks']
                    else:
                        result = evaluate(analyzer, fixture)
                        if first_run:
                            click.echo(format_report(result))
                        checks = checks_from_question_result(fixture, result)
                except Exception as e:
                    # One broken fixture must not hide the others' results
                    any_error = True
                    click.echo(f"Error evaluating {fixture_file}: {e}", err=True)
                    checks = [{'name': 'fixture ran to completion', 'ok': False,
                               'detail': str(e), 'axis': 'error'}]
                entries.append({'fixture': fixture_file, 'checks': checks})
            run_entries.append(entries)
    finally:
        if no_cache:
            if saved_llm_cache is None:
                _os.environ.pop('LLM_CACHE', None)
            else:
                _os.environ['LLM_CACHE'] = saved_llm_cache

    entries = run_entries[0]
    board = format_scoreboard(entries)
    if runs > 1:
        board = board.replace('=== Scoreboard ===',
                              f'=== Scoreboard (run 1 of {runs}) ===')
    click.echo('\n' + board)

    if runs > 1:
        click.echo('\n' + format_flip_report(flip_report(run_entries)))
        if not no_cache:
            click.echo('  (disk caches were ON — flips may be hidden; '
                       're-run with --no-cache for a true stability read)')

    if baseline_path:
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
        click.echo('\n' + format_diff(diff_baseline(baseline, entries)))

    if json_out:
        payload = {
            'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'app_version': __version__,
            'fixtures': entries,
            'totals': {
                'passed': sum(1 for e in entries for c in e['checks'] if c['ok']),
                'checks': sum(len(e['checks']) for e in entries),
            },
        }
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        click.echo(f"\nWrote {json_out} (use it as --compare baseline for the "
                   f"next round)")

    if any_error:
        sys.exit(2)
    # A failure in ANY run fails the command — with --runs, run 2's flaky
    # failure is exactly the signal, and exiting 0 on it would hide it
    if any(not c['ok'] for run in run_entries for e in run
           for c in e['checks']):
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
