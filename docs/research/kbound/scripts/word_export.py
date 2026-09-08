"""Source-explicit Word export into a new isolated work directory.

Content/layout verification is separate; DOCX ZIP timestamps are not normalized.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape, unescape

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('legacy_word_export', BASE / 'build_docx.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
TOOL_NAMES = ('pandoc', 'latexpand', 'pdftex', 'pdftoppm')

def resolve_tools(overrides=None):
    overrides = overrides or {}
    unknown = set(overrides) - set(TOOL_NAMES)
    if unknown:
        raise ValueError(f'unknown executable override: {sorted(unknown)}')
    resolved = {}
    for name in TOOL_NAMES:
        requested = str(overrides.get(name) or name)
        found = shutil.which(requested)
        if not found or not Path(found).is_file() or not os.access(found, os.X_OK):
            raise RuntimeError(f'required executable not found or not executable: {requested}')
        resolved[name] = str(Path(found).resolve())
    return resolved

def reject_symlink_path(path):
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f'symlink paths are not allowed: {component}')

def preflight_paths(source, role, output, workdir):
    for path in (source, output, workdir):
        reject_symlink_path(path)
    source = select_source(source, role)
    output, workdir = output.resolve(), workdir.resolve()
    if not source.is_file():
        raise ValueError(f'source must be a regular file: {source}')
    if output.suffix.lower() != '.docx' or output.parent != workdir:
        raise ValueError('output must be a .docx file directly inside the fresh workdir')
    if workdir.is_relative_to(source.parent):
        raise ValueError('workdir must be outside the source tree')
    require_fresh_output(workdir)
    require_fresh_output(output)
    if not workdir.parent.is_dir():
        raise ValueError('workdir parent must be an existing directory')
    active_source_paths(source)
    return source, output, workdir

def active_source_paths(source):
    """Literal input/include closure plus assets directly consumed by export.

    This is not a general TeX interpreter: it follows the same literal input
    graph as macro discovery and does not hash unreferenced archival files.
    """
    reject_symlink_path(source)
    source = source.resolve()
    root = source.parent
    assets = {root / 'paper/figures/decision_flow.tex',
              root / 'figures/fig_frontier_schematic.png'}
    for path in assets:
        reject_symlink_path(path)
        if not path.is_file():
            raise ValueError(f'required source asset is missing or not a regular file: {path}')
    pending, visited = [source, root / 'paper/figures/decision_flow.tex'], set()
    while pending:
        owner = pending.pop()
        if owner in visited:
            continue
        visited.add(owner)
        text = legacy.strip_tex_comments(owner.read_text(encoding='utf-8'))
        for target in legacy.INPUT_PATTERN.findall(text):
            relative = Path(target.strip())
            if not relative.suffix:
                relative = relative.with_suffix('.tex')
            for base in (root, owner.parent):
                reject_symlink_path(base / relative)
            dependency = legacy._resolve_tex_input(owner, target, root)
            if not dependency.is_relative_to(root):
                raise ValueError(f'TeX dependency is outside manuscript root: {dependency}')
            pending.append(dependency)
    return tuple(sorted(visited | assets))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def select_source(source, role):
    expected = {'main':'kbound_short_main.tex','combined':'kbound_main_with_appendix.tex'}
    if role not in expected or source.name != expected[role]:
        raise ValueError('source driver does not match requested role')
    return source.resolve()

def require_fresh_output(path):
    reject_symlink_path(path)
    if path.exists():
        raise FileExistsError(path)

def require_clean_process(returncode, stdout, stderr):
    if returncode:
        raise RuntimeError(f'conversion failed ({returncode}): {stderr}')
    if stderr.strip():
        raise RuntimeError(f'conversion warning: {stderr}')
    return stdout

def source_contract(processed, ast, expected_figures=3):
    counts = {'tables':0,'figures':0,'math':0}
    kinds = {'Table':'tables','Image':'figures','Math':'math'}
    def walk(obj):
        if isinstance(obj, dict):
            if obj.get('t') in kinds:
                counts[kinds[obj['t']]] += 1
            for value in obj.values():
                walk(value)
        elif isinstance(obj,list):
            for value in obj:
                walk(value)
    walk(ast)
    tables = len(re.findall(r'\\begin\{(?:tabular\*?|longtable|tabularx)\}', legacy.strip_tex_comments(processed)))
    if tables != counts['tables']:
        raise ValueError(f'table parse loss: source={tables}, parsed={counts["tables"]}')
    if counts['figures'] != expected_figures:
        raise ValueError(f'expected {expected_figures} figures, found {counts["figures"]}')
    if counts['math'] < 1:
        raise ValueError('no native math parsed')
    return counts

def validate_counts(observed, expected):
    for key in ('tables','figures','math'):
        if observed[key] != expected[key]:
            raise ValueError(f'{key} preservation failed: expected {expected[key]}, found {observed[key]}')

def package_counts(path):
    with ZipFile(path) as archive:
        xml = ET.fromstring(archive.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
        return {'tables':len(xml.findall('.//w:tbl',ns)), 'figures':len(xml.findall('.//w:drawing',ns)), 'math':len(xml.findall('.//m:oMath',ns))}

def validate_media(path, expected):
    with ZipFile(path) as archive:
        observed=sorted(hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if name.startswith('word/media/') and not name.endswith('/'))
    if observed!=sorted(expected):
        raise ValueError('embedded media bytes differ from current source figures')
    return observed

def sanitize_image_descriptions(source, output):
    """Sanitize path-valued drawing descriptions and remove stale optional stats."""
    reject_symlink_path(source)
    require_fresh_output(output)
    changed=0
    local_path=re.compile(r'^(?:/|~/|file:/|[A-Za-z]:[\\/]|\.\.?/|figures/|paper/)')
    description=re.compile(r'\bdescr="([^"]*)"')
    with ZipFile(source) as original, ZipFile(output,'x') as clean:
        clean.comment=original.comment
        for info in original.infolist():
            data=original.read(info.filename)
            if info.filename=='word/document.xml':
                text=data.decode('utf-8')
                def drawing(match):
                    nonlocal changed
                    block=match.group(0)
                    primary=re.search(r'<wp:docPr\b[^>]*>',block)
                    primary_description=description.search(primary.group(0)) if primary else None
                    meaningful=unescape(primary_description.group(1),{'&quot;':'"'}) if primary_description else ''
                    def properties(properties_match):
                        nonlocal changed
                        tag=properties_match.group(0)
                        value=description.search(tag)
                        if value is None or not local_path.match(unescape(value.group(1))):
                            return tag
                        if not meaningful or local_path.match(meaningful):
                            raise ValueError('path-valued image description lacks meaningful figure alt text')
                        changed+=1
                        replacement=escape(meaningful,{'"':'&quot;'})
                        return tag[:value.start(1)]+replacement+tag[value.end(1):]
                    return re.sub(r'<(?:pic:cNvPr|wp:docPr)\b[^>]*>',properties,block)
                data=re.sub(r'<w:drawing\b[^>]*>.*?</w:drawing>',drawing,text,flags=re.DOTALL).encode('utf-8')
            elif info.filename=='docProps/app.xml':
                data=re.sub(rb'<(Pages|Words|Characters|CharactersWithSpaces|Lines|Paragraphs)\b[^>]*>[^<]*</\1>',b'',data)
            clean.writestr(info,data)
    return changed

def role_references(text, role):
    text=re.sub(r'^\s*\\(?:newcommand|providecommand)\{\\(?:KBSuppRef|KBMainRef|KBMainEqRef)\}\[[23]\].*$', '', text, flags=re.MULTILINE)
    pattern=re.compile(r'\\(KBSuppRef|KBMainRef|KBMainEqRef)\b')
    while match:=pattern.search(text):
        cursor=match.end()
        args=[]
        for _ in range(2 if match.group(1)=='KBMainEqRef' else 3):
            while cursor<len(text) and text[cursor].isspace():
                cursor+=1
            if cursor>=len(text) or text[cursor]!='{':
                raise ValueError('malformed companion reference')
            end=legacy.find_brace_end(text,cursor)
            args.append(text[cursor+1:end])
            cursor=end+1
        if match.group(1)=='KBMainEqRef':
            replacement=r'Eq.~\eqref{'+args[0]+'}'
        else:
            replacement=args[2] if role=='main' and match.group(1)=='KBSuppRef' else args[0]+r'~\ref{'+args[1]+'}'
        text=text[:match.start()]+replacement+text[cursor:]
    return text

def layout_conditionals(text):
    pattern=r'\\ifdefined\\(?:KBoundTMLRLayout|KBoundSingleColumnLayout)\b(.*?)\\else(.*?)\\fi'
    return re.sub(pattern,lambda match:match.group(2),text,flags=re.DOTALL)

def normalize_word_tables(text):
    # Short captions serve the TeX list of tables; Word needs the full caption.
    text=re.sub(r'\\caption\[[^\]]*\]\s*\{',lambda _:r'\caption{',text)
    known=r'\begin{tabular}{@{}p{0.18\textwidth}rrrrrccp{0.26\textwidth}@{}}'
    if known in text:
        raise ValueError('misdeclared action-table column count; correct the shared source')
    return text

def format_algorithm_numbering(doc):
    active=False
    for paragraph in doc.paragraphs:
        if paragraph.style and paragraph.style.name.startswith('Heading'):
            active=paragraph.text.startswith('Algorithm ')
        if active and paragraph._p.pPr is not None and paragraph._p.pPr.numPr is not None:
            level=paragraph._p.pPr.numPr.ilvl
            indent=0.5*(1+(int(level.val) if level is not None else 0))
            paragraph.paragraph_format.left_indent=legacy.Inches(indent)
            paragraph.paragraph_format.first_line_indent=legacy.Inches(-0.25)
            paragraph.paragraph_format.tab_stops.add_tab_stop(legacy.Inches(indent))
            # LO occasionally collapses the implicit label tab beside OMML.
            # A visible nonbreaking text-space guarantees a separator while
            # retaining native numbering and leaving all existing runs intact.
            gap=paragraph.add_run('\u00a0')._r
            paragraph._p.remove(gap)
            paragraph._p.insert(1,gap)

def replace_tikz(text, image):
    pattern=r'\\resizebox\{[^}]+\}\{!\}\{\s*\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}\s*\}'
    text,count=re.subn(pattern,lambda _:r'\includegraphics[width=0.98\textwidth]{'+str(image)+'}',text,flags=re.DOTALL)
    if count!=1:
        raise ValueError(f'expected one actual TikZ figure, found {count}')
    return text

def render_tikz(root, output, tools):
    directory=output.parent/(output.stem+'-figure')
    directory.mkdir(exist_ok=False)
    original=root/'paper/figures/decision_flow.tex'
    wrapper=directory/'current-flow.tex'
    wrapper.write_text(r'\documentclass[tikz,border=4pt]{standalone}'+'\n'+r'\usepackage{amsmath,amssymb}\usetikzlibrary{arrows.meta,positioning}\begin{document}'+'\n'+original.read_text()+'\n'+r'\end{document}')
    command=[tools['pdftex'],'-fmt=pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={directory}',str(wrapper)]
    proc=subprocess.run(command,cwd=root,capture_output=True,text=True)
    (directory/'compile.txt').write_text(proc.stdout+'\n'+proc.stderr)
    if proc.returncode:
        raise RuntimeError('actual TikZ source rendering failed; see figure compile.txt')
    rendered=subprocess.run([tools['pdftoppm'],'-png','-singlefile','-r','180',str(wrapper.with_suffix('.pdf')),str(wrapper.with_suffix(''))],capture_output=True,text=True)
    require_clean_process(rendered.returncode,rendered.stdout,rendered.stderr)
    return wrapper.with_suffix('.png')

def snapshot(source):
    root = source.resolve().parent
    return {str(path.relative_to(root)):digest(path) for path in active_source_paths(source)}

def build(source, role, output, *, workdir, tools=None):
    source, output, workdir = preflight_paths(source, role, output, workdir)
    tools = resolve_tools(tools)
    # Claim an absent directory only after all static path/tool checks pass.
    # This is not a concurrent-path-race or whole-conversion atomicity guarantee.
    workdir.mkdir(exist_ok=False)
    try:
        return _build(source, role, output, tools)
    except Exception as exc:
        with output.with_suffix('.failure.json').open('x', encoding='utf-8') as failure:
            json.dump({'error_type':type(exc).__name__, 'message':str(exc)}, failure, indent=2)
            failure.write('\n')
        raise

def _build(source, role, output, tools):
    root=source.parent
    before=snapshot(source)
    legacy.ROOT=root
    legacy.SOURCE=source
    legacy.GENERATED_DIR=root/'paper/generated'
    macros=legacy.generated_macros(legacy.discover_generated_macro_sources(source,generated_dir=legacy.GENERATED_DIR,search_root=root))
    resources=':'.join(str(root/p) for p in ('','figures','paper','paper/generated','paper/sections'))
    flat=subprocess.run([tools['latexpand'],'--fatal',str(source)],cwd=root,capture_output=True,text=True)
    flattened=require_clean_process(flat.returncode,flat.stdout,flat.stderr)
    actual_flow=render_tikz(root,output,tools)
    selected=normalize_word_tables(replace_tikz(layout_conditionals(role_references(flattened,role)),actual_flow))
    processed,references,required_values=legacy.preprocess_with_metadata(selected,macros=macros)
    prepared=output.with_suffix('.prepared.tex')
    require_fresh_output(prepared)
    prepared.write_text(processed)
    common=[tools['pandoc'],str(prepared),'--from','latex',f'--resource-path={resources}','--standalone']
    parsed=subprocess.run([*common,'--to','json'],cwd=root,capture_output=True,text=True)
    ast=json.loads(require_clean_process(parsed.returncode,parsed.stdout,parsed.stderr))
    contract=source_contract(processed,ast,expected_figures=2)
    receipt=dict(schema='kbound-staged-word-v1',role=role,source=str(source),source_snapshot=before,prepared_sha256=digest(prepared),expected=contract,references=references,exporter_sha256=digest(__file__),legacy_exporter_sha256=digest(BASE/'build_docx.py'),tools=tools)
    output.with_suffix('.inputs.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='word-convert-',dir=output.parent) as temporary:
        raw=Path(temporary)/'raw.docx'
        proc=subprocess.run([*common,'--output',str(raw)],cwd=root,capture_output=True,text=True)
        require_clean_process(proc.returncode,proc.stdout,proc.stderr)
        validate_counts(package_counts(raw),contract)
        styled=Path(temporary)/'styled.docx'
        legacy.postprocess(raw,styled)
        doc=legacy.Document(styled)
        format_algorithm_numbering(doc)
        formatted=Path(temporary)/'formatted.docx'
        doc.save(formatted)
        sanitized_descriptions=sanitize_image_descriptions(formatted,output)
    validate_counts(package_counts(output),contract)
    media=validate_media(output,[digest(actual_flow),digest(root/'figures/fig_frontier_schematic.png')])
    legacy.validate_docx(output,references,required_values,expected_tables=contract['tables'],expected_figures=2)
    if snapshot(source)!=before:
        raise RuntimeError('source snapshot changed during conversion; candidate is not promotable')
    receipt.update(output_sha256=digest(output),observed=package_counts(output),embedded_media_sha256=media,source_unchanged=True,sanitized_image_descriptions=sanitized_descriptions,render_qa='pending')
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--role',choices=('main','combined'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True,help='absent directory outside the source tree; parent must exist')
    for name in TOOL_NAMES:
        parser.add_argument('--'+name,help='executable path or name; default: discover on PATH')
    args=parser.parse_args()
    tools={name:getattr(args,name) for name in TOOL_NAMES if getattr(args,name)}
    print(json.dumps(build(args.source,args.role,args.output,workdir=args.workdir,tools=tools),indent=2))

if __name__=='__main__':
    main()
