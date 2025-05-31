# generate_code_summary.py
# Version 1.3:
# - Modified to include the FULL docstring for modules, classes, and functions.
# - Docstrings are now formatted within Markdown code blocks (```text ... ```).
# - Removed MAX_DOCSTRING_SUMMARY_LENGTH and first-line summarization logic.
# - Retains other logic from v1.2 (path handling, type hint formatting, etc.).

"""
A Python script to analyze a given set of Python files and directories,
extracting information about modules, classes, and functions.

This script uses Python's `ast` (Abstract Syntax Tree) module to parse
source code without executing it. It extracts:
- Full module-level docstrings.
- Class definitions including their names, base classes, and full docstrings.
- Function and method definitions including their names, parameters (with type hints
  and basic default value representation), return type hints, and full docstrings.

The output is formatted as Markdown and saved to a specified file (default: codebase_summary.md),
which can then be used as context for Large Language Models (LLMs) or for human
developers to understand the codebase structure in detail.
"""

import ast
import os
from typing import List, Tuple, Optional, Any, Union

# --- Configuration ---
OUTPUT_FILENAME: str = "codebase_summary.md" 
PATHS_TO_ANALYZE: List[str] = [ 
    "app.py",
    "modules"
]
EXCLUDE_FILES: List[str] = ["__init__.py", "generate_code_summary.py"]
# MAX_DOCSTRING_SUMMARY_LENGTH removed

# --- Helper Functions (format_type_hint, extract_arg_info, get_default_value_str remain the same as v1.2) ---

def format_type_hint(node: Optional[ast.expr]) -> str:
    """
    Attempts to reconstruct a type hint string from an AST annotation node.
    Handles simple names, attributes, subscripted generics, and basic union types.
    Uses `ast.unparse` (Python 3.9+) as a fallback.

    Args:
        node: The AST node representing the type annotation (e.g., ast.Name, ast.Subscript).

    Returns:
        A string representation of the type hint, or "Any" if unparseable.
    """
    if node is None:
        return "" 

    if hasattr(ast, 'unparse'):
        try:
            return ast.unparse(node)
        except Exception:
            pass 

    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{format_type_hint(node.value)}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        value_str = format_type_hint(node.value)
        slice_val_str: str
        if isinstance(node.slice, ast.Tuple): 
            slice_val_str = ", ".join(format_type_hint(elt) for elt in node.slice.elts)
        elif isinstance(node.slice, (ast.Name, ast.Constant, ast.Subscript, ast.Attribute, ast.BinOp)):
             slice_val_str = format_type_hint(node.slice)
        elif hasattr(node.slice, 'value') and isinstance(node.slice.value, (ast.Name, ast.Constant, ast.Subscript, ast.Attribute, ast.BinOp)): 
            slice_val_str = format_type_hint(node.slice.value)
        else:
            slice_val_str = "..." 
        return f"{value_str}[{slice_val_str}]"
    elif isinstance(node, ast.Constant) and node.value is None:
        return "None"
    elif isinstance(node, ast.NameConstant) and node.value is None: 
        return "None"
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr): 
        left = format_type_hint(node.left)
        right = format_type_hint(node.right)
        return f"{left} | {right}"
    else:
        return "Any" 

def extract_arg_info(arg_node: ast.arg) -> str:
    """
    Extracts an argument's name and its type hint string.

    Args:
        arg_node: The ast.arg node for the parameter.

    Returns:
        A string like "param_name: ParamType".
    """
    name: str = arg_node.arg
    type_hint_str: str = ""
    if arg_node.annotation:
        type_hint_str = f": {format_type_hint(arg_node.annotation)}"
    return f"{name}{type_hint_str}"

def get_default_value_str(default_node: Optional[ast.expr]) -> str:
    """
    Tries to get a string representation of a default argument value.
    Uses ast.unparse if available, otherwise simple constants.
    """
    if default_node is None:
        return ""
    
    if hasattr(ast, 'unparse'):
        try:
            return f" = {ast.unparse(default_node)}"
        except Exception:
            return " = <complex_default>"
    elif isinstance(default_node, ast.Constant):
        return f" = {default_node.value!r}" 
    elif isinstance(default_node, ast.NameConstant): 
        return f" = {default_node.value}"
    elif isinstance(default_node, ast.Num): 
         return f" = {default_node.n!r}"
    elif isinstance(default_node, ast.Str): 
         return f" = {default_node.s!r}"
    else:
        return " = <default_value>"

def analyze_python_file(filepath: str, project_root_dir: str) -> List[str]:
    """
    Analyzes a single Python file to extract module, class, and function details
    including their full docstrings.

    Args:
        filepath: The absolute path to the Python file to analyze.
        project_root_dir: The absolute path to the project's root directory.

    Returns:
        A list of strings, formatted in Markdown, summarizing the file's structure.
    """
    output_lines: List[str] = []
    try:
        with open(filepath, "r", encoding="utf-8") as source_file:
            source_code: str = source_file.read()
            tree: ast.Module = ast.parse(source_code, filename=filepath)
    except Exception as e:
        relative_filepath_for_error = os.path.relpath(filepath, start=project_root_dir).replace(os.sep, '/')
        return [f"## File: {relative_filepath_for_error}", f"Error parsing file: {e}\n\n---\n"]

    relative_filepath = os.path.relpath(filepath, start=project_root_dir).replace(os.sep, '/')
    output_lines.append(f"## File: {relative_filepath}")

    module_docstring: Optional[str] = ast.get_docstring(tree, clean=True)
    output_lines.append("Module Docstring:")
    if module_docstring:
        output_lines.append(f"```text\n{module_docstring.strip()}\n```\n")
    else:
        output_lines.append("[No docstring provided]\n")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name: str = node.name
            args_info_list: List[str] = []
            num_args = len(node.args.args)
            num_defaults = len(node.args.defaults)
            defaults_start_index = num_args - num_defaults

            for i, arg in enumerate(node.args.args):
                arg_str = extract_arg_info(arg)
                if i >= defaults_start_index:
                    default_val_str = get_default_value_str(node.args.defaults[i - defaults_start_index])
                    arg_str += default_val_str
                args_info_list.append(arg_str)
            
            if node.args.vararg: args_info_list.append(f"*{extract_arg_info(node.args.vararg)}")
            if node.args.kwarg: args_info_list.append(f"**{extract_arg_info(node.args.kwarg)}")
            args_str: str = ", ".join(args_info_list)
            
            return_hint_str: str = ""
            if node.returns: return_hint_str = f" -> {format_type_hint(node.returns)}"
            
            func_type_prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            output_lines.append(f"### {func_type_prefix} {func_name}({args_str}){return_hint_str}")
            
            docstring: Optional[str] = ast.get_docstring(node, clean=True)
            output_lines.append("Docstring:")
            if docstring:
                output_lines.append(f"```text\n{docstring.strip()}\n```")
            else:
                output_lines.append("[No docstring provided]")
            output_lines.append("")

        elif isinstance(node, ast.ClassDef):
            class_name: str = node.name
            base_classes_str: str = ""
            if node.bases:
                base_classes_str = f"({', '.join(format_type_hint(base) for base in node.bases)})"
            
            output_lines.append(f"### class {class_name}{base_classes_str}")
            
            class_docstring: Optional[str] = ast.get_docstring(node, clean=True)
            output_lines.append("Docstring:")
            if class_docstring:
                output_lines.append(f"```text\n{class_docstring.strip()}\n```")
            else:
                output_lines.append("[No docstring provided]")

            for class_node_item in node.body:
                if isinstance(class_node_item, (ast.FunctionDef, ast.AsyncFunctionDef)): # Methods
                    method_name: str = class_node_item.name
                    method_args_info_list: List[str] = []
                    
                    m_num_args = len(class_node_item.args.args)
                    m_num_defaults = len(class_node_item.args.defaults)
                    m_defaults_start_index = m_num_args - m_num_defaults

                    for i, arg in enumerate(class_node_item.args.args):
                        arg_str = extract_arg_info(arg)
                        if i >= m_defaults_start_index:
                           default_val_str = get_default_value_str(class_node_item.args.defaults[i - m_defaults_start_index])
                           arg_str += default_val_str
                        method_args_info_list.append(arg_str)
                    
                    if class_node_item.args.vararg: method_args_info_list.append(f"*{extract_arg_info(class_node_item.args.vararg)}")
                    if class_node_item.args.kwarg: method_args_info_list.append(f"**{extract_arg_info(class_node_item.args.kwarg)}")
                    
                    method_args_str: str = ", ".join(method_args_info_list)
                    method_return_hint_str: str = ""
                    if class_node_item.returns:
                        method_return_hint_str = f" -> {format_type_hint(class_node_item.returns)}"
                    
                    method_type_prefix = "async def" if isinstance(class_node_item, ast.AsyncFunctionDef) else "def"
                    output_lines.append(f"  #### {method_type_prefix} {method_name}({method_args_str}){method_return_hint_str}")
                    
                    method_docstring: Optional[str] = ast.get_docstring(class_node_item, clean=True)
                    output_lines.append(f"  Docstring:") # Added for methods
                    if method_docstring:
                        output_lines.append(f"  ```text\n  {method_docstring.strip().replace(chr(10), chr(10) + '  ')}\n  ```") # Indent multiline method docstrings
                    else:
                        output_lines.append("  [No docstring provided]")
            output_lines.append("") # Extra newline after class methods processed
    
    output_lines.append("---\n") 
    return output_lines

def main() -> None:
    """
    Main execution function for the script.
    Identifies Python files and generates a Markdown summary including full docstrings.
    """
    project_root_dir: str = os.getcwd() 
    
    all_summary_lines: List[str] = [f"# Codebase Summary (v{SCRIPT_VERSION})\n"]

    for path_item_str in PATHS_TO_ANALYZE:
        current_path_to_analyze: str = os.path.join(project_root_dir, path_item_str)

        if os.path.isfile(current_path_to_analyze) and current_path_to_analyze.endswith(".py"):
            if os.path.basename(current_path_to_analyze) not in EXCLUDE_FILES:
                all_summary_lines.extend(analyze_python_file(current_path_to_analyze, project_root_dir))
        elif os.path.isdir(current_path_to_analyze):
            for root, dirnames, filenames in os.walk(current_path_to_analyze):
                dirnames[:] = [d for d in dirnames if d not in ['venv', '.venv', 'env', '__pycache__']]
                for filename in filenames:
                    if filename.endswith(".py") and filename not in EXCLUDE_FILES:
                        filepath: str = os.path.join(root, filename)
                        all_summary_lines.extend(analyze_python_file(filepath, project_root_dir))
        else:
            print(f"Warning: Configured path item '{path_item_str}' ('{current_path_to_analyze}') "
                  "is not a valid file or directory, or not a Python file. Skipping.")

    output_filepath: str = os.path.join(project_root_dir, OUTPUT_FILENAME)
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            for line in all_summary_lines:
                f.write(line + "\n")
        print(f"Codebase summary successfully saved to: {output_filepath}")
    except IOError as e:
        print(f"Error: Could not write summary to file '{output_filepath}'. Error: {e}")
        print("\n--- Summary Output to Console (due to file write error) ---")
        for line in all_summary_lines:
            print(line)

SCRIPT_VERSION: str = "1.3" 

if __name__ == "__main__":
    print(f"Running Codebase Summary Generator v{SCRIPT_VERSION}...")
    main()
# end of generate_code_summary.py
