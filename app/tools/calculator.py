import ast
import operator


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str):
    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Invalid constant")

        if isinstance(node, ast.BinOp):
            operation = OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("Operator not allowed")
            return operation(evaluate(node.left), evaluate(node.right))

        if isinstance(node, ast.UnaryOp):
            operation = OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("Operator not allowed")
            return operation(evaluate(node.operand))

        raise ValueError("Expression not allowed")

    return evaluate(tree.body)
