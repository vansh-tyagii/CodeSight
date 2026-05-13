import ast
import re

NETWORK_LIBS = {"requests", "httpx", "aiohttp", "urllib", "boto3", "socket", "urllib3"}
DB_LIBS = {"sqlalchemy", "pymongo", "sqlite3", "psycopg2", "redis", "pymysql", "motor", "asyncpg"}

OPERATOR_MAP = {
    ast.Add: 'addition', ast.Sub: 'subtraction', ast.Mult: 'multiplication',
    ast.Div: 'division', ast.Pow: 'exponentiation', ast.Mod: 'modulo',
    ast.Eq: 'equality', ast.NotEq: 'inequality', ast.In: 'membership', ast.And: 'logical_and'
}

class AdvancedCodeParser(ast.NodeVisitor):
    def __init__(self, source_code, file_path="unknown.py"):
        self.source_code = source_code
        self.file_path = file_path
        self.chunks = []
        self.scope_stack = []
        
        self.symbol_table = {}
        self.inferred_types = {} 
        self.file_dependencies = set()

    def visit_Import(self, node):
        for alias in node.names:
            self.symbol_table[alias.asname or alias.name] = alias.name
            self.file_dependencies.add(alias.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        for alias in node.names:
            resolved_name = f"{module}.{alias.name}" if module else alias.name
            self.symbol_table[alias.asname or alias.name] = resolved_name
            if module: self.file_dependencies.add(module.split('.')[0])
        self.generic_visit(node)

    def resolve_call_name(self, node):
        if isinstance(node.func, ast.Name):
            base = node.func.id
            return self.symbol_table.get(base, base)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                base = node.func.value.id
                if base in self.inferred_types:
                    base_type = self.inferred_types[base]
                    return f"{base_type}.{node.func.attr}"
                resolved_base = self.symbol_table.get(base, base)
                return f"{resolved_base}.{node.func.attr}"
            else:
                return f"complex_obj.{node.func.attr}"
        return "unknown_call"

    def clean_docstring(self, doc):
        return re.sub(r'\s+', ' ', doc).strip() if doc else ""

    def extract_io_signature(self, node):
        args = [a.arg for a in node.args.args]
        returns = "unknown"
        if hasattr(node, 'returns') and node.returns:
            if hasattr(ast, 'unparse'): returns = ast.unparse(node.returns)
            elif isinstance(node.returns, ast.Name): returns = node.returns.id
        return f"({', '.join(args)}) -> {returns}"

    def extract_decorators(self, node):
        decs = []
        for d in node.decorator_list:
            if hasattr(ast, 'unparse'): decs.append(ast.unparse(d))
        return decs

    def build_nested_cfg(self, node_body):
        if not isinstance(node_body, list):
            node_body = [node_body]
            
        flow = []
        for stmt in node_body:
            if isinstance(stmt, ast.If):
                if_flow = self.build_nested_cfg(stmt.body)
                else_flow = self.build_nested_cfg(stmt.orelse) if stmt.orelse else ""
                branch = f"IF[{if_flow}]"
                if else_flow: branch += f" -> ELSE[{else_flow}]"
                flow.append(branch)
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                flow.append(f"LOOP[{self.build_nested_cfg(stmt.body)}]")
            elif isinstance(stmt, ast.Try):
                try_flow = self.build_nested_cfg(stmt.body)
                catch_flows = " | ".join([self.build_nested_cfg(h.body) for h in stmt.handlers])
                flow.append(f"TRY[{try_flow}] -> EXCEPT[{catch_flows}]")
            elif isinstance(stmt, ast.Return):
                flow.append("RETURN")
            elif isinstance(stmt, ast.Break):
                flow.append("BREAK")
            elif isinstance(stmt, ast.Continue):
                flow.append("CONTINUE")
                
        return " -> ".join(flow) if flow else "SEQ"

    def analyze_block(self, node):
        calls = []
        operations = set()
        metrics = {"complexity": 1, "max_depth": 0, "loops": 0, "branches": 0}

        def traverse(n, depth):
            # STRICT SCOPE BOUNDARY: Do not bleed into nested functions or classes
            if depth > 0 and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return
                
            metrics["max_depth"] = max(metrics["max_depth"], depth)

            if isinstance(n, ast.Assign):
                if isinstance(n.value, ast.Call):
                    call_name = self.resolve_call_name(n.value)
                    for target in n.targets:
                        if isinstance(target, ast.Name):
                            self.inferred_types[target.id] = call_name 

            elif isinstance(n, ast.Call):
                calls.append(self.resolve_call_name(n))

            elif isinstance(n, ast.BinOp) and type(n.op) in OPERATOR_MAP:
                operations.add(OPERATOR_MAP[type(n.op)])
            elif isinstance(n, ast.Compare):
                operations.add("comparisons")

            elif isinstance(n, ast.If):
                metrics["complexity"] += 1
                metrics["branches"] += 1
            elif isinstance(n, (ast.For, ast.AsyncFor, ast.While)):
                metrics["complexity"] += 1
                metrics["loops"] += 1
            elif isinstance(n, ast.Try):
                metrics["complexity"] += len(n.handlers)
            elif isinstance(n, ast.ExceptHandler):
                metrics["branches"] += 1

            for child in ast.iter_child_nodes(n):
                traverse(child, depth + 1)

        traverse(node, 0)
        cfg_string = self.build_nested_cfg(getattr(node, 'body', []))
        return list(set(calls)), list(operations), cfg_string, metrics

    def detect_explicit_patterns(self, node_name, calls, cfg_string, metrics):
        patterns = []
        calls_str = " ".join(calls).lower()

        if node_name in calls:
            patterns.append("recursion")

        has_network = any(call.split('.')[0] in NETWORK_LIBS for call in calls)
        has_db = any(call.split('.')[0] in DB_LIBS for call in calls)

        if metrics["loops"] > 0 and "TRY[" in cfg_string and "EXCEPT[" in cfg_string and (has_network or has_db):
            patterns.append("retry_logic")

        if metrics["loops"] > 0 and any(ml in calls_str for ml in ["backward", "step", "zero_grad", "fit", "predict"]):
            patterns.append("ml_training_loop")

        # STRICT DB CRUD: Prevent false positives on 'session'
        db_actions = ["commit", "execute", "rollback", "flush"]
        if has_db or any(action in calls_str for action in db_actions):
            patterns.append("database_crud")

        return patterns

    def process_node(self, node, node_type, name):
        segment = ast.get_source_segment(self.source_code, node)
        if not segment: return

        scoped_name = ".".join(self.scope_stack + [name]) if self.scope_stack else name
        
        chunk_data = {
            "name": scoped_name,
            "type": node_type,
            "file_path": self.file_path,
            "docstring": self.clean_docstring(ast.get_docstring(node)),
            "dependencies": list(self.file_dependencies)
        }

        if node_type in ["function", "async_function", "method", "async_method"]:
            chunk_data["signature"] = self.extract_io_signature(node)
            chunk_data["decorators"] = self.extract_decorators(node)
            
            calls, ops, cfg_string, metrics = self.analyze_block(node)
            patterns = self.detect_explicit_patterns(name, calls, cfg_string, metrics)
            
            chunk_data["executes"] = calls
            chunk_data["operations"] = ops
            chunk_data["cfg_path"] = f"START -> {cfg_string} -> END"
            chunk_data["metrics"] = metrics
            chunk_data["detected_patterns"] = patterns

        self.chunks.append(chunk_data)

    def visit_ClassDef(self, node):
        self.process_node(node, "class", node.name)
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node):
        node_type = "method" if self.scope_stack and self.scope_stack[0].istitle() else "function"
        self.process_node(node, node_type, node.name)
        
        self.scope_stack.append(node.name)
        self.generic_visit(node) 
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node):
        node_type = "async_method" if self.scope_stack and self.scope_stack[0].istitle() else "async_function"
        self.process_node(node, node_type, node.name)
        
        self.scope_stack.append(node.name)
        self.generic_visit(node) 
        self.scope_stack.pop()

if __name__ == "__main__":
    test_code = """
import requests

def outer_func():
    client = requests.Session()
    
    def inner_retry_func():
        for i in range(3):
            try:
                res = client.get("https://api.github.com")
                if res.status_code == 200:
                    return res.json()
            except Exception:
                pass
        return None
"""
    try:
        tree = ast.parse(test_code)
        parser = AdvancedCodeParser(test_code, "test_file.py")
        parser.visit(tree)
        
        import json
        for chunk in parser.chunks:
            print(json.dumps(chunk, indent=2))
            
    except Exception as e:
        print(f"Parser failed: {e}")