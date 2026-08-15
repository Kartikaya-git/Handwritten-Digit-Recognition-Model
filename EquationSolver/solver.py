import ast
import operator

from classes import SYMBOL_DISPLAY

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


class ParseError(ValueError):
    pass


def symbols_to_expression(symbol_classes):
    """Turn an ordered list of predicted class names (e.g.
    ['1','2','add','7']) into an expression string like '12+7'.

    Consecutive digits are grouped into one number and normalized through
    int() so leading zeros (e.g. '05') don't break Python's literal grammar."""
    parts = []
    digit_buffer = []

    def flush_digits():
        if digit_buffer:
            parts.append(str(int("".join(digit_buffer))))
            digit_buffer.clear()

    for cls in symbol_classes:
        if cls.isdigit():
            digit_buffer.append(cls)
        elif cls in SYMBOL_DISPLAY:
            flush_digits()
            parts.append(SYMBOL_DISPLAY[cls])
        else:
            raise ParseError(f"unrecognized symbol class: {cls}")
    flush_digits()

    expr = "".join(parts)
    if not expr:
        raise ParseError("empty equation")
    return expr


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ParseError(f"disallowed expression element: {ast.dump(node)}")


def solve_expression(expr):
    """Safely evaluate an arithmetic expression string (no eval())."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ParseError(f"could not parse '{expr}': {e}") from e
    try:
        return _safe_eval(tree)
    except ZeroDivisionError as e:
        raise ParseError(f"division by zero in '{expr}'") from e


def solve_symbols(symbol_classes):
    expr = symbols_to_expression(symbol_classes)
    result = solve_expression(expr)
    return expr, result
