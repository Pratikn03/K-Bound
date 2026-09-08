"""Synthetic portability boundaries; never invoke TeX, Pandoc, or rendering."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'docs/research/kbound/scripts' / 'word_export.py'
spec = importlib.util.spec_from_file_location('portable_word', SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def inventory(root):
    return {str(p.relative_to(root)): ('link', os.readlink(p)) if p.is_symlink()
            else ('file', p.read_bytes()) if p.is_file() else ('directory', None)
            for p in root.rglob('*')}


def fake_tools(root):
    root.mkdir()
    for name in ('pandoc', 'latexpand', 'pdftex', 'pdftoppm'):
        path = root / name
        path.write_text('#!/bin/sh\nexit 99\n')
        path.chmod(0o755)
    return {name: str(root / name) for name in ('pandoc', 'latexpand', 'pdftex', 'pdftoppm')}


def required_assets(root):
    for name in ('paper/figures/decision_flow.tex', 'figures/fig_frontier_schematic.png'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic required asset')


def test_tools_discovered_from_path_and_explicit_override_wins(tmp_path, monkeypatch):
    discovered = fake_tools(tmp_path / 'discovered')
    chosen = fake_tools(tmp_path / 'chosen')
    monkeypatch.setenv('PATH', str(tmp_path / 'discovered'))
    assert hasattr(exporter, 'resolve_tools'), 'portable executable discovery is missing'
    resolved = exporter.resolve_tools({'pandoc': chosen['pandoc']})
    assert resolved == {**discovered, 'pandoc': chosen['pandoc']}


def test_missing_explicit_tool_does_not_fall_back_to_path(tmp_path, monkeypatch):
    fake_tools(tmp_path / 'bin')
    monkeypatch.setenv('PATH', str(tmp_path / 'bin'))
    assert hasattr(exporter, 'resolve_tools'), 'portable executable discovery is missing'
    with pytest.raises((ValueError, RuntimeError), match='executable'):
        exporter.resolve_tools({'pandoc': str(tmp_path / 'missing')})


@pytest.mark.parametrize('kind', ['directory', 'nonexecutable'])
def test_invalid_tool_is_rejected(tmp_path, monkeypatch, kind):
    selected = fake_tools(tmp_path / 'bin')
    invalid = tmp_path / 'bad'
    if kind == 'directory':
        invalid.mkdir()
    else:
        invalid.write_text('not executable')
        invalid.chmod(0o644)
        if os.access(invalid, os.X_OK):
            pytest.skip('fixture filesystem does not enforce non-executable file modes')
    assert hasattr(exporter, 'resolve_tools'), 'portable executable discovery is missing'
    with pytest.raises((ValueError, RuntimeError), match='executable'):
        exporter.resolve_tools({**selected, 'pandoc': str(invalid)})


@pytest.mark.parametrize('kind', ['existing_workdir', 'existing_output', 'workdir_link',
    'output_link', 'source_link', 'source_parent_link', 'source_asset_link', 'input_path_symlink',
    'lexical_parent_before_dotdot', 'outside_workdir', 'workdir_inside_source'])
def test_cli_rejects_unsafe_paths_before_any_write(tmp_path, kind):
    source_root = tmp_path / 'source'
    source_root.mkdir()
    source = source_root / 'kbound_short_main.tex'
    source.write_text('synthetic source: must never reach conversion')
    required_assets(source_root)
    workdir = tmp_path / 'fresh'
    output = workdir / 'main.docx'
    outside = tmp_path / 'outside'
    outside.mkdir()
    if kind == 'existing_workdir':
        workdir.mkdir()
    elif kind == 'existing_output':
        workdir.mkdir()
        output.write_bytes(b'accepted document')
    elif kind == 'workdir_link':
        workdir.symlink_to(outside, target_is_directory=True)
    elif kind == 'output_link':
        workdir.mkdir()
        output.symlink_to(outside / 'missing.docx')
    elif kind == 'source_link':
        actual = source_root / 'actual.tex'
        source.rename(actual)
        source.symlink_to(actual)
    elif kind == 'source_parent_link':
        link = tmp_path / 'source-link'
        link.symlink_to(source_root, target_is_directory=True)
        source = link / source.name
    elif kind == 'source_asset_link':
        asset = source_root / 'figures/fig_frontier_schematic.png'
        asset.unlink()
        asset.symlink_to(outside / 'missing.png')
    elif kind == 'input_path_symlink':
        (outside / 'input.tex').write_text('external source')
        link = tmp_path / 'input-link'
        link.symlink_to(outside, target_is_directory=True)
        source.write_text(r'\input{../input-link/input}')
    elif kind == 'lexical_parent_before_dotdot':
        link = tmp_path / 'link'
        link.symlink_to(outside, target_is_directory=True)
        workdir = link / '..' / 'fresh'
        output = workdir / 'main.docx'
    elif kind == 'outside_workdir':
        output = outside / 'main.docx'
    elif kind == 'workdir_inside_source':
        workdir = source_root / 'generated-output'
        output = workdir / 'main.docx'
    tools = fake_tools(tmp_path / 'bin')
    tool_args = [arg for name, path in tools.items() for arg in ('--'+name, path)]
    before = inventory(tmp_path)
    result = subprocess.run([sys.executable, str(SCRIPT), '--source', str(source),
        '--role', 'main', '--output', str(output), '--workdir', str(workdir), *tool_args],
        capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode != 0
    assert 'unrecognized arguments' not in result.stderr
    if kind in ('existing_workdir', 'existing_output'):
        assert 'FileExistsError' in result.stderr
    elif kind == 'outside_workdir':
        assert 'directly inside the fresh workdir' in result.stderr
    elif kind == 'workdir_inside_source':
        assert 'workdir must be outside the source tree' in result.stderr
    else:
        assert 'symlink paths are not allowed' in result.stderr
    assert inventory(tmp_path) == before


@pytest.mark.parametrize('role,driver', [('main', 'kbound_short_main.tex'),
    ('combined', 'kbound_main_with_appendix.tex')])
def test_cli_routes_explicit_tool_and_records_failure_only_in_claimed_workdir(tmp_path, role, driver):
    source = tmp_path / 'source' / driver
    generated = source.parent / 'paper/generated/fake_numbers.tex'
    generated.parent.mkdir(parents=True)
    generated.write_text(r'\newcommand{\FakeMetric}{0.1}')
    source.write_text(r'\input{paper/generated/fake_numbers}')
    required_assets(source.parent)
    source_before = inventory(source.parent)
    workdir = tmp_path / 'fresh'
    tools = fake_tools(tmp_path / 'bin')
    Path(tools['latexpand']).write_text('#!'+sys.executable+'\n'
        'import json, os, sys\n'
        f'with open({str(workdir / "invocation.json")!r}, "x") as f: json.dump([os.getcwd(), sys.argv[1:]], f)\n'
        'sys.stderr.write("synthetic converter failure")\nsys.exit(7)\n')
    tool_args = [arg for name, path in tools.items() for arg in ('--'+name, path)]
    result = subprocess.run([sys.executable, str(SCRIPT), '--source', str(source),
        '--role', role, '--output', str(workdir / 'main.docx'), '--workdir', str(workdir), *tool_args],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert json.loads((workdir / 'invocation.json').read_text()) == [str(source.parent), ['--fatal', str(source)]]
    failure = json.loads((workdir / 'main.failure.json').read_text())
    assert failure['error_type'] == 'RuntimeError'
    assert 'conversion failed (7)' in failure['message']
    assert not (workdir / 'main.docx').exists()
    assert inventory(source.parent) == source_before


def test_synthetic_figure_tools_receive_unchanged_compile_and_raster_arguments(tmp_path):
    root = tmp_path / 'source'
    figure = root / 'paper/figures/decision_flow.tex'
    figure.parent.mkdir(parents=True)
    figure.write_text('synthetic TikZ input; no TeX process will run')
    workdir = tmp_path / 'work'
    workdir.mkdir()
    tools = fake_tools(tmp_path / 'bin')
    for name in ('pdftex', 'pdftoppm'):
        Path(tools[name]).write_text('#!'+sys.executable+'\n'
            'import json, sys\n'
            f'with open({str(workdir / (name+".json"))!r}, "x") as f: json.dump(sys.argv[1:], f)\n')
    result = exporter.render_tikz(root, workdir / 'main.docx', tools)
    wrapper = workdir / 'main-figure/current-flow.tex'
    assert result == wrapper.with_suffix('.png')
    assert json.loads((workdir / 'pdftex.json').read_text()) == [
        '-fmt=pdflatex', '-interaction=nonstopmode', '-halt-on-error',
        '-output-directory='+str(wrapper.parent), str(wrapper)]
    assert json.loads((workdir / 'pdftoppm.json').read_text()) == [
        '-png', '-singlefile', '-r', '180', str(wrapper.with_suffix('.pdf')), str(wrapper.with_suffix(''))]


@pytest.mark.parametrize('role,driver', [('main', 'kbound_short_main.tex'),
    ('combined', 'kbound_main_with_appendix.tex')])
def test_safe_preflight_supports_both_relocated_roles_without_writes(tmp_path, role, driver):
    source = tmp_path / 'source' / driver
    source.parent.mkdir()
    source.write_text('synthetic')
    required_assets(source.parent)
    workdir = tmp_path / 'fresh'
    before = inventory(tmp_path)
    assert hasattr(exporter, 'preflight_paths'), 'fresh isolated workdir preflight is missing'
    assert exporter.preflight_paths(source, role, workdir / 'paper.docx', workdir) == (
        source.resolve(), (workdir / 'paper.docx').resolve(), workdir.resolve())
    assert inventory(tmp_path) == before


def test_missing_tools_fail_without_creating_workdir(tmp_path, monkeypatch):
    source = tmp_path / 'source' / 'kbound_short_main.tex'
    source.parent.mkdir()
    source.write_text('synthetic')
    required_assets(source.parent)
    workdir = tmp_path / 'fresh'
    monkeypatch.setenv('PATH', '')
    before = inventory(tmp_path)
    result = subprocess.run([sys.executable, str(SCRIPT), '--source', str(source),
        '--role', 'main', '--output', str(workdir / 'main.docx'), '--workdir', str(workdir)],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'executable' in result.stderr
    assert inventory(tmp_path) == before


@pytest.mark.parametrize('target_kind', ['absolute', 'parent_relative'])
def test_external_tex_input_is_rejected_before_any_workdir_write(tmp_path, target_kind):
    source = tmp_path / 'source/kbound_short_main.tex'
    source.parent.mkdir()
    required_assets(source.parent)
    external = tmp_path / 'external.tex'
    external.write_text('external dependency')
    target = str(external) if target_kind == 'absolute' else '../external'
    source.write_text(r'\input{'+target+'}')
    tools = fake_tools(tmp_path / 'bin')
    tool_args = [arg for name, path in tools.items() for arg in ('--'+name, path)]
    before = inventory(tmp_path)
    workdir = tmp_path / 'new'
    result = subprocess.run([sys.executable, str(SCRIPT), '--source', str(source), '--role', 'main',
        '--workdir', str(workdir), '--output', str(workdir / 'main.docx'), *tool_args],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert inventory(tmp_path) == before
    assert 'outside manuscript root' in result.stderr


def test_snapshot_hashes_only_selected_literal_closure_and_required_assets(tmp_path):
    source = tmp_path / 'source/kbound_short_main.tex'
    source.parent.mkdir()
    required_assets(source.parent)
    child = source.parent / 'sections/child.tex'
    child.parent.mkdir()
    child.write_text('active child')
    source.write_text(r'\input{sections/child}'+'\n% '+r'\input{unopened}')
    historical = source.parent / 'historical.pdf'
    historical.write_bytes(b'historical bytes not build input')
    (source.parent / 'kbound_main_with_appendix.tex').write_text('other role')
    first = exporter.snapshot(source)
    assert set(first) == {'kbound_short_main.tex', 'sections/child.tex',
        'paper/figures/decision_flow.tex', 'figures/fig_frontier_schematic.png'}
    assert first['sections/child.tex'] == hashlib.sha256(b'active child').hexdigest()
    historical.write_bytes(b'changed unrelated bytes')
    (source.parent / 'unrelated-link.pdf').symlink_to(source.parent / 'not-present.pdf')
    assert exporter.snapshot(source) == first
    workdir = tmp_path / 'new'
    exporter.preflight_paths(source, 'main', workdir / 'main.docx', workdir)
    child.write_text('changed active child')
    assert exporter.snapshot(source) != first


def test_owner_relative_parent_input_inside_root_remains_supported(tmp_path):
    source = tmp_path / 'source/kbound_short_main.tex'
    source.parent.mkdir()
    required_assets(source.parent)
    (source.parent / 'nested').mkdir()
    source.write_text(r'\input{nested/child}')
    (source.parent / 'nested/child.tex').write_text(r'\input{../sibling}')
    (source.parent / 'sibling.tex').write_text('active sibling')
    assert 'sibling.tex' in exporter.snapshot(source)


def test_ci_executes_word_export_and_publication_regressions():
    import shlex
    import yaml

    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / '.github/workflows/kbound-ci.yml').read_text())
    job = workflow['jobs']['kbound-research-tests']
    required = {'tests/test_word_export.py', 'tests/test_word_portability.py',
        'tests/test_build_docx_pipeline.py', 'tests/test_kbound_publication_package.py'}
    executed = set()
    assert 'if' not in job and not job.get('continue-on-error')
    for step in job['steps']:
        if 'if' in step or step.get('continue-on-error'):
            continue
        for line in step.get('run', '').replace('\\\n', ' ').splitlines():
            if not line.lstrip().startswith('pytest '):
                continue
            args = shlex.split(line, comments=True)
            if args and args[0] == 'pytest' and '--collect-only' not in args:
                if any(option in args for option in ('-k', '-m', '||', ';', '--ignore', '--deselect')):
                    continue
                executed.update(required.intersection(args))
    assert required <= executed, f'Word regressions are not executed by CI: {required - executed}'
