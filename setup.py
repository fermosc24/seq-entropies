from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name="seq_entropies",
    version="0.1.0",
    description="Fast sequence entropy and complexity estimators (LZ76, ZL77, etc.)",
    author="Your Name",
    author_email="your.email@example.com",
    packages=["seq_entropies"],
    ext_modules=cythonize(
        [
            Extension(
                "seq_entropies.lz",
                sources=["seq_entropies/lz.pyx"],
                include_dirs=[numpy.get_include()],
                language="c",
            )
        ]
    ),
    install_requires=[
        "numpy>=1.18",
        "Cython>=0.29"
    ],
    python_requires=">=3.7",
    zip_safe=False,
)

