import collections
import re
import argparse
import subprocess
import shlex
import cgen

parameter_re = re.compile(r"([\w\d_:]+(<.*>)?)\s+mutable\s+([\w\d_]+)(\{([^}]*)\})?")

preprocessor = "clang -E -I include"

"""
auto create_from_cfg<Type>( json ){

}
"""

if __name__ == "__main__":
    files = ["include/ginkgo/core/solver/cg.hpp"]

    factories = {}

    Parameter = collections.namedtuple("Parameter", ["type", "name", "default"])

    for name in files:
        cp = subprocess.run(shlex.split(f"{preprocessor} {name}"), stdout=subprocess.PIPE, universal_newlines=True,
                            stderr=subprocess.DEVNULL)
        begin = re.search("class Cg", cp.stdout).start()
        end = re.search("\n};", cp.stdout[begin:]).start()
        region = cp.stdout[begin:(begin + end)]

        factory = []
        for match in parameter_re.finditer(region):
            factory.append(Parameter(match.group(1), match.group(3), match.group(5)))
        factories["Cg"] = factory
    print(factories)
