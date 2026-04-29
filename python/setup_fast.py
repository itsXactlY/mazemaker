#!/usr/bin/env python3
"""
Build Cython extension for Neural Memory hot paths.

Usage:
    python setup_fast.py build_ext --inplace

NOTE: this is a BUILD-TIME script, not a runtime module. The hermes
plugin loader enumerates every .py in the plugin dir and tried to
import this file at startup, hitting `No module named 'setuptools'`
on lean venvs. Guard the imports so the file loads cleanly even when
setuptools/Cython/numpy aren't installed — it just becomes a no-op
at import-time. Build-time invocation (`python setup_fast.py ...`)
still surfaces the real ImportError when the deps are missing, which
is the right behaviour there.
"""

try:
    from setuptools import setup, Extension
    from Cython.Build import cythonize
    import numpy
except ImportError:
    # Loaded by the hermes plugin enumerator without build deps. Skip the
    # rest — there's nothing for a runtime importer to use here. The build
    # script still fails loudly when invoked as __main__ via the guard
    # below, which is the only path that actually NEEDS these imports.
    if __name__ == "__main__":
        raise
    setup = Extension = cythonize = numpy = None  # type: ignore[assignment]

if __name__ == "__main__" or setup is not None:
    extensions = [
        Extension(
            "fast_ops",
            sources=["fast_ops.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-march=native"],
        ),
    ]

    setup(
        name="neural-memory-fast-ops",
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
            },
        ),
    )
