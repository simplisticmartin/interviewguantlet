"""Static analysis of candidate-submitted code.

**This module never executes candidate code.** Not with ``exec``, not with
``subprocess``, not in a thread. Candidate code is treated as hostile input
(spec section 17), and the only safe place to run it is an ephemeral, network-isolated,
resource-capped container - which is phase 4 of the roadmap and is not implemented yet.

What this *does* do is real and useful today: parse the submission, report syntax
validity, and extract structural signals (does it define a function, how deeply are
loops nested, does it handle empty input) that the interviewer uses to choose its next
question. When a candidate's solution has a triple-nested loop, the interviewer asking
"what's the complexity of what you just wrote?" is a genuinely good interview move, and
it needs no sandbox.

``executed`` is always ``False`` and every consumer surfaces that, so nothing in the
product implies tests were run when they were not.
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field

SUPPORTED_LANGUAGES = ("python", "java", "javascript", "typescript")

_JAVA_METHOD = re.compile(
    r"\b(?:public|private|protected|static|final|synchronized|\s)*"
    r"[\w<>\[\],\s]+\s+(\w+)\s*\([^)]*\)\s*\{",
)
_JS_FUNCTION = re.compile(
    r"\bfunction\s+(\w+)\s*\(|\b(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\("
)
_LOOP_KEYWORD = re.compile(r"\b(for|while)\b")
_EMPTY_GUARD = re.compile(
    r"(\.(is)?[Ee]mpty\(\)|len\([^)]*\)\s*==\s*0|==\s*null|is None|\.length\s*===?\s*0|"
    r"if\s+not\s+\w+)",
)


@dataclass(slots=True)
class CodeCheckResult:
    language: str
    syntax_ok: bool
    executed: bool = False
    execution_note: str = (
        "Not executed. Sandboxed execution runs in isolated containers (roadmap phase 4); "
        "this is static analysis only."
    )
    errors: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    max_loop_depth: int = 0
    line_count: int = 0
    has_empty_input_guard: bool = False
    uses_recursion: bool = False
    interviewer_signals: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def check_code(code: str, language: str | None) -> CodeCheckResult:
    """Analyse a submission. Returns a result even for unknown languages."""
    normalised = (language or _guess_language(code) or "unknown").lower()
    body = _strip_fences(code)

    if normalised == "python":
        result = _check_python(body)
    elif normalised in {"java", "javascript", "typescript"}:
        result = _check_braced(body, normalised)
    else:
        result = CodeCheckResult(
            language=normalised,
            syntax_ok=True,
            errors=[],
            line_count=len(body.splitlines()),
        )
        result.errors.append(f"No static analyser for language '{normalised}'.")

    result.has_empty_input_guard = bool(_EMPTY_GUARD.search(body))
    result.interviewer_signals = _signals(result)
    return result


def _strip_fences(code: str) -> str:
    fenced = re.match(r"^\s*```[a-zA-Z]*\n(.*?)```\s*$", code, re.DOTALL)
    return fenced.group(1) if fenced else code


def _guess_language(code: str) -> str | None:
    if re.search(r"\b(public\s+class|System\.out\.println|import java\.)", code):
        return "java"
    if re.search(r"\b(def |import |print\()", code):
        return "python"
    if re.search(r"\b(const |let |=>|console\.log)", code):
        return "javascript"
    return None


def _check_python(code: str) -> CodeCheckResult:
    result = CodeCheckResult(language="python", syntax_ok=True, line_count=len(code.splitlines()))
    try:
        # ast.parse only builds a syntax tree; it does not evaluate anything.
        tree = ast.parse(code)
    except SyntaxError as exc:
        result.syntax_ok = False
        result.errors.append(f"Line {exc.lineno}: {exc.msg}")
        return result

    function_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function_names.append(node.name)
    result.functions = function_names
    result.max_loop_depth = _python_loop_depth(tree)
    result.uses_recursion = any(
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id in set(function_names)
        for inner in ast.walk(tree)
    )
    return result


def _python_loop_depth(tree: ast.AST, depth: int = 0) -> int:
    best = depth
    for child in ast.iter_child_nodes(tree):
        child_depth = depth + 1 if isinstance(child, ast.For | ast.While | ast.AsyncFor) else depth
        best = max(best, _python_loop_depth(child, child_depth))
    return best


def _check_braced(code: str, language: str) -> CodeCheckResult:
    """Brace/paren balance plus structural extraction for C-family languages.

    A full parser for Java and TypeScript is out of scope; balance checking catches the
    overwhelmingly common submission error (an unclosed block) without pretending to
    compile anything.
    """
    result = CodeCheckResult(language=language, syntax_ok=True, line_count=len(code.splitlines()))

    pairs = {"}": "{", ")": "(", "]": "["}
    stack: list[str] = []
    in_string: str | None = None
    escaped = False
    index = 0
    while index < len(code):
        char = code[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            index += 1
            continue
        if char in {'"', "'", "`"}:
            in_string = char
        elif code.startswith("//", index):
            index = code.find("\n", index)
            if index == -1:
                break
            continue
        elif code.startswith("/*", index):
            closing = code.find("*/", index)
            index = closing + 2 if closing != -1 else len(code)
            continue
        elif char in {"{", "(", "["}:
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                result.syntax_ok = False
                result.errors.append(f"Unbalanced '{char}' at offset {index}.")
                break
            stack.pop()
        index += 1

    if result.syntax_ok and stack:
        result.syntax_ok = False
        result.errors.append(f"{len(stack)} unclosed block(s): missing {' '.join(stack)}.")

    if language == "java":
        result.functions = [match.group(1) for match in _JAVA_METHOD.finditer(code)]
    else:
        result.functions = [
            name for match in _JS_FUNCTION.finditer(code) for name in match.groups() if name
        ]

    result.max_loop_depth = _braced_loop_depth(code)
    result.uses_recursion = any(
        len(re.findall(rf"\b{re.escape(name)}\s*\(", code)) > 1 for name in result.functions
    )
    return result


def _braced_loop_depth(code: str) -> int:
    """Track brace depth, recording how many loop headers are simultaneously open."""
    depth = 0
    best = 0
    loop_depths: list[int] = []
    for line in code.splitlines():
        if _LOOP_KEYWORD.search(line):
            loop_depths.append(depth)
            best = max(best, len(loop_depths))
        depth += line.count("{") - line.count("}")
        loop_depths = [entry for entry in loop_depths if entry < depth]
    return best


def _signals(result: CodeCheckResult) -> list[str]:
    """Concrete things worth asking about, derived from the structure alone."""
    signals: list[str] = []
    if not result.syntax_ok:
        signals.append("Submission does not parse - ask them to walk through it line by line.")
    if result.max_loop_depth >= 2:
        signals.append(
            f"Loop nesting depth {result.max_loop_depth} - ask for the complexity of what "
            "they wrote, not the algorithm they intended."
        )
    if not result.has_empty_input_guard:
        signals.append(
            "No visible empty/null input guard - ask what assumptions they are making "
            "about the input."
        )
    if result.uses_recursion:
        signals.append("Recursive - ask about stack depth on large inputs.")
    if not result.functions and result.syntax_ok:
        signals.append("No named function defined - ask how they would expose this as an API.")
    return signals
