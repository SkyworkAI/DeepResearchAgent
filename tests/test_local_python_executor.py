import unittest

from src.tools.executor.local_python_executor import (
    BASE_PYTHON_TOOLS,
    DEFAULT_MAX_LEN_OUTPUT,
    InterpreterError,
    evaluate_python_code,
)

# It's good practice to define a small, fixed list for default authorized_imports in tests
# unless a test specifically needs to modify it.
TEST_DEFAULT_AUTHORIZED_IMPORTS = [
    "math",
    "subprocess",
    "os",
    "os.path",
]  # Example, can be empty if preferred for stricter tests


class TestPythonInterpreterSandbox(unittest.TestCase):
    def setUp(self):
        # These are defaults for the tools/state available during evaluation.
        # Tests can override state or custom_tools if needed.
        self.static_tools = BASE_PYTHON_TOOLS.copy()
        self.custom_tools = {}
        # self.state is not defined here, as evaluate_python_code takes state as an argument
        # and it's better to pass a fresh state for each test call to avoid interference.

    def _evaluate(self, code, authorized_imports=None, state=None):
        if authorized_imports is None:
            authorized_imports = list(TEST_DEFAULT_AUTHORIZED_IMPORTS)  # Use a copy

        current_state = state if state is not None else {}

        # evaluate_python_code returns (result, is_final_answer)
        return evaluate_python_code(
            code,
            static_tools=self.static_tools,
            custom_tools=self.custom_tools,  # Pass along self.custom_tools
            state=current_state,  # Pass along current_state
            authorized_imports=authorized_imports,
            max_print_outputs_length=DEFAULT_MAX_LEN_OUTPUT,
        )

    # === Import Tests ===
    def test_import_disallowed_module_direct(self):
        with self.assertRaisesRegex(InterpreterError, "Import of os is not allowed"):
            self._evaluate("import os", authorized_imports=[])

    def test_import_disallowed_module_from(self):
        with self.assertRaisesRegex(InterpreterError, "Import from os is not allowed"):
            self._evaluate("from os import path", authorized_imports=[])

    def test_import_allowed_module(self):
        result, _ = self._evaluate(
            "import math; x = math.sqrt(4)", authorized_imports=["math"]
        )
        self.assertEqual(result, 2.0)

    def test_import_submodule_allowed_implicitly(self):
        # If 'collections' is allowed, 'collections.abc' should be usable via attribute access.
        # The get_safe_module ensures submodules are also checked if they were explicitly imported.
        # This test checks if 'collections.abc' can be accessed if 'collections' is authorized.
        # The updated get_safe_module will try to check 'collections.abc' when 'collections' is processed.
        # So, 'collections.abc' must also be in authorized_imports or match a wildcard like 'collections.*'
        # For this test, let's authorize both specifically.
        result, _ = self._evaluate(
            "import collections; c = collections.abc.Callable",
            authorized_imports=["collections", "collections.abc"],
        )
        # Check that 'c' is indeed the Callable type from collections.abc
        import collections.abc as abc_module

        self.assertIs(result, abc_module.Callable)

    def test_import_only_specific_submodule_denies_parent_access(self):
        # Allow "os.path" but try to use "os.listdir()" -> should fail on "os" not being fully allowed for that.
        # This tests the precision of check_import_authorized.
        # If only "os.path" is authorized, "import os" should fail.
        with self.assertRaisesRegex(InterpreterError, "Import of os is not allowed"):
            self._evaluate("import os; os.listdir('.')", authorized_imports=["os.path"])

    def test_import_authorized_submodule_directly(self):
        result, _ = self._evaluate(
            "import os.path; x = os.path.basename('/a/b')",
            authorized_imports=["os.path"],
        )
        self.assertEqual(result, "b")

    def test_import_from_authorized_submodule(self):
        result, _ = self._evaluate(
            "from os.path import basename; x = basename('/a/b')",
            authorized_imports=["os.path"],
        )
        self.assertEqual(result, "b")

    # === Dangerous Function Call Tests ===
    def test_call_dangerous_builtin_function_eval(self):
        with self.assertRaisesRegex(
            InterpreterError, "Forbidden access to function: eval"
        ):
            self._evaluate("eval('1+1')")

    def test_call_dangerous_builtin_function_exec(self):
        with self.assertRaisesRegex(
            InterpreterError, "Forbidden access to function: exec"
        ):
            self._evaluate("exec('a=1')")

    def test_call_dangerous_os_function_system_via_import(self):
        # This relies on 'os' module itself being blocked from import.
        with self.assertRaisesRegex(InterpreterError, "Import of os is not allowed"):
            self._evaluate("import os; os.system('echo hello')")

    def test_call_dangerous_function_if_module_was_somehow_allowed(self):
        # If 'os' was authorized, safer_eval should still block 'os.system' if it's in DANGEROUS_FUNCTIONS
        # This tests the defense in depth of safer_eval.
        with self.assertRaisesRegex(
            InterpreterError, "Forbidden access to function: system"
        ):
            self._evaluate(
                "import os; os.system('echo hello')", authorized_imports=["os"]
            )

    def test_call_allowed_builtin_function(self):
        result, _ = self._evaluate("len([1,2,3])")
        self.assertEqual(result, 3)

    def test_call_function_returned_by_tool_if_dangerous(self):
        # Mocking state to contain a dangerous function
        current_state = {"my_dangerous_func": eval}
        with self.assertRaisesRegex(
            InterpreterError, "Forbidden access to function: eval"
        ):
            self._evaluate(
                "my_dangerous_func('1+1')", state=current_state, authorized_imports=[]
            )

    # === Dunder Attribute Access Tests ===
    def test_access_disallowed_dunder_directly_on_dict(self):
        with self.assertRaisesRegex(
            InterpreterError, "Forbidden access to dunder attribute: __dict__"
        ):
            self._evaluate("x = {}; x.__dict__")

    def test_access_disallowed_dunder_directly_on_module(self):
        # math.__loader__ is an example.
        with self.assertRaisesRegex(
            InterpreterError, "Forbidden access to dunder attribute: __loader__"
        ):
            self._evaluate("import math; math.__loader__", authorized_imports=["math"])

    def test_access_disallowed_dunder_via_getattr(self):
        # getattr is nodunder_getattr in BASE_PYTHON_TOOLS
        with self.assertRaisesRegex(
            InterpreterError, "Forbidden access to dunder attribute: __subclasses__"
        ):
            self._evaluate("x = type(0); getattr(x, '__subclasses__')")

    def test_allowed_dunder_method_indirectly_len(self):
        result, _ = self._evaluate("x = [1,2]; len(x)")
        self.assertEqual(result, 2)

    def test_allowed_dunder_method_indirectly_getitem(self):
        result, _ = self._evaluate("x = [10,20]; x[1]")
        self.assertEqual(result, 20)

    # === AST Node Behavior Tests ===
    def test_assign_to_static_tool_name_blocked(self):
        with self.assertRaisesRegex(InterpreterError, "Cannot assign to name 'len'"):
            self._evaluate("len = lambda x: x")

    def test_lambda_executes_in_sandbox_blocks_import(self):
        with self.assertRaisesRegex(InterpreterError, "Import of sys is not allowed"):
            self._evaluate("f = lambda: __import__('sys'); f()", authorized_imports=[])

    def test_def_function_executes_in_sandbox_blocks_import(self):
        code = """
def my_func():
    import shutil # Disallowed
    return shutil.disk_usage('.')
my_func()
"""
        with self.assertRaisesRegex(
            InterpreterError, "Import of shutil is not allowed"
        ):
            self._evaluate(code, authorized_imports=[])

    def test_class_def_executes_in_sandbox_blocks_import_in_init(self):
        code = """
class MyClass:
    def __init__(self):
        import subprocess # Disallowed
        self.name = subprocess.call('echo')
    def get_name(self):
        return self.name
x = MyClass()
x.get_name()
"""
        with self.assertRaisesRegex(
            InterpreterError, "Import of subprocess is not allowed"
        ):
            self._evaluate(code, authorized_imports=[])

    def test_class_def_executes_in_sandbox_blocks_import_in_method(self):
        code = """
class MyClassMethod:
    def do_bad_stuff(self):
        import _thread # Disallowed
        return _thread.get_ident()
x = MyClassMethod()
x.do_bad_stuff()
"""
        with self.assertRaisesRegex(
            InterpreterError, "Import of _thread is not allowed"
        ):
            self._evaluate(code, authorized_imports=[])

    def test_unsupported_ast_node_global_keyword(self):
        code = """
x = 0
def f():
    global x # ast.Global node
    x = 1
"""
        with self.assertRaisesRegex(InterpreterError, "Global is not supported"):
            self._evaluate(code)

    def test_unsupported_ast_node_nonlocal_keyword(self):
        code = """
def f():
    x = 1
    def g():
        nonlocal x # ast.Nonlocal node
        x = 2
    g()
"""
        with self.assertRaisesRegex(InterpreterError, "Nonlocal is not supported"):
            self._evaluate(code)

    def test_comprehension_sandbox_import(self):
        with self.assertRaisesRegex(InterpreterError, "Import of os is not allowed"):
            self._evaluate(
                "[__import__('os') for i in range(1)]", authorized_imports=[]
            )

    def test_try_except_sandbox_import(self):
        code = """
try:
    x = 1
except Exception:
    import os
else:
    import sys
finally:
    import subprocess
"""
        # The first import attempt (os) should be caught.
        with self.assertRaisesRegex(InterpreterError, "Import of os is not allowed"):
            self._evaluate(code, authorized_imports=[])


if __name__ == "__main__":
    unittest.main()
