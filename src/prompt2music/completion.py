"""Shell completion scripts generated from the argparse tree."""

from __future__ import annotations

import argparse


def _children(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:  # argparse has no public API for this
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _options(parser: argparse.ArgumentParser) -> list[str]:
    return [opt for action in parser._actions for opt in action.option_strings
            if opt.startswith("--")]


def _tree(parser: argparse.ArgumentParser) -> dict[str, list[str]]:
    table: dict[str, list[str]] = {"": sorted(_children(parser)) + _options(parser)}
    for name, child in _children(parser).items():
        grand = _children(child)
        table[name] = sorted(grand) + _options(child)
        for sub, leaf in grand.items():
            table[f"{name} {sub}"] = _options(leaf)
    return table


def render_completion(parser: argparse.ArgumentParser, shell: str) -> str:
    tree = _tree(parser)
    if shell in ("bash", "zsh"):
        cases = []
        for key in sorted(tree, key=lambda k: -len(k.split())):
            words = " ".join(tree[key])
            pattern = '""' if key == "" else f'"{key}"*'
            cases.append(f'    {pattern}) opts="{words}" ;;')
        body = "\n".join(cases)
        script = f"""_prompt2music() {{
  local cur="${{COMP_WORDS[COMP_CWORD]}}" path="" n=0 w
  for w in "${{COMP_WORDS[@]:1:COMP_CWORD-1}}"; do
    [[ $w == -* ]] && continue
    path="${{path:+$path }}$w"; n=$((n+1)); [[ $n -ge 2 ]] && break
  done
  local opts=""
  case "$path" in
{body}
  esac
  COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
  [[ ${{#COMPREPLY[@]}} -eq 0 ]] && COMPREPLY=( $(compgen -f -- "$cur") )
}}
complete -F _prompt2music prompt2music aimusic
"""
        if shell == "zsh":
            script = "autoload -U +X bashcompinit && bashcompinit\n" + script
        return script
    lines = []
    for exe in ("prompt2music", "aimusic"):
        lines.append(f"complete -c {exe} -f")
        top = [k for k in tree[""] if not k.startswith("-")]
        lines.append(f'complete -c {exe} -n "__fish_use_subcommand" -a "{" ".join(top)}"')
        for cmd in top:
            subs = [k for k in tree.get(cmd, []) if not k.startswith("-")]
            if subs:
                lines.append(f'complete -c {exe} -n "__fish_seen_subcommand_from {cmd}" '
                             f'-a "{" ".join(subs)}"')
    return "\n".join(lines) + "\n"
