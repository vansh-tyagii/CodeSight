# backend/ast_behavior_parser.py
import ast
import re

def extract_io_signature(node):
    """Extracts roughly f(type, type) -> type"""
    args = [a.arg for a in node.args.args]
    returns = "unknown"
    if node.returns and isinstance(node.returns, ast.Name):
        returns = node.returns.id
    return f"f({', '.join(args)}) -> {returns}"

def profile_operations(node):
    """Detects behavior profiling (arithmetic, string, loops, etc.)"""
    profile = {"arithmetic_heavy": False, "search_sort_pattern": False, "data_manipulation": False}
    math_ops = 0
    loops = 0
    comparisons = 0
    
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp):
            math_ops += 1
        elif isinstance(child, (ast.For, ast.While)):
            loops += 1
        elif isinstance(child, ast.Compare):
            comparisons += 1
        elif isinstance(child, ast.Subscript): # List/Dict slicing
            profile["data_manipulation"] = True
            
    if math_ops > 3: profile["arithmetic_heavy"] = True
    if loops > 0 and comparisons > 0: profile["search_sort_pattern"] = True
    
    return [k for k, v in profile.items() if v]

def generate_behavior_proxy(source_code):
    try:
        tree = ast.parse(source_code)
        behavior_tags = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                behavior_tags.append(f"Signature: {extract_io_signature(node)}")
                ops = profile_operations(node)
                if ops:
                    behavior_tags.append(f"Profile: {', '.join(ops)}")
        return " | ".join(behavior_tags) if behavior_tags else "Profile: Structural script"
    except SyntaxError:
        return "Profile: Unparseable snippet"