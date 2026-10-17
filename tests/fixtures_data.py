"""Synthetic fixture constants (not real competition data)."""

CORE_SRC = """\
import os


def helper(x):
    return x + 1


class Engine:
    def run(self, x):
        return helper(x)

    async def arun(self, x):
        return await fetch(x)


async def fetch(x):
    return x
"""

# Gold patch: edits helper() body (old line 5) and inserts a line inside Engine.run.
# Built line by line because blank context lines in a diff are a single space " ", which
# editors and linters like to strip from triple-quoted strings.
GOLD_PATCH = (
    "\n".join(
        [
            "diff --git a/pkg/core.py b/pkg/core.py",
            "index 1111111..2222222 100644",
            "--- a/pkg/core.py",
            "+++ b/pkg/core.py",
            "@@ -4,7 +4,7 @@ import os",
            " def helper(x):",
            "-    return x + 1",
            "+    return x + 2",
            " ",
            " ",
            " class Engine:",
            "@@ -9,4 +9,5 @@ class Engine:",
            "     def run(self, x):",
            "+        x = int(x)",
            "         return helper(x)",
            " ",
            "diff --git a/tests/test_core.py b/tests/test_core.py",
            "--- a/tests/test_core.py",
            "+++ b/tests/test_core.py",
            "@@ -1,2 +1,3 @@",
            " def test_helper():",
            "-    assert True",
            "+    assert helper(1) == 3",
            "+",
            "diff --git a/pkg/new_module.py b/pkg/new_module.py",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/pkg/new_module.py",
            "@@ -0,0 +1,2 @@",
            "+def brand_new():",
            "+    return 1",
        ]
    )
    + "\n"
)

#: base_commit of the synthetic task; real files are named f"{repo_short}_{base_commit}".
COMMIT = "a" * 40
