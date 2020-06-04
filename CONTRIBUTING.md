# Contributing guidelines

We are glad that you are interested in contributing to Ginkgo. Please have a
look at our coding guidelines before proposing a pull request.

## Table of Contents

[Most Important stuff](#most-important-stuff)

[Project Structure](#project-structure)
 * [Extended header files](#extended-header-files)
 * [Using library classes](#using-library-classes)

[Git related](#git-related)
 * [Our git Workflow](#our-git-workflow)
 * [Writing good commit messages](#writing-good-commit-messages)
 * [Creating, Reviewing and Merging Pull
   Requests](#creating-reviewing-and-merging-pull-requests)

[Code Style](#code-style)
 * [Automatic code formatting](#automatic-code-formatting)
 * [Naming Scheme](#naming-scheme)
 * [Whitespace](#whitespace)
 * [Include statement grouping](#include-statement-grouping)
 * [Other Code Formatting not handled by
   ClangFormat](#other-code-formatting-not-handled-by-clangformat)
 * [CMake coding style](#cmake-coding-style)
 
[Helper Scripts](#helper-scripts)
 * [Create a new algorithm](#create-a-new-algorithm)
 * [Converting CUDA code to HIP code](#converting-cuda-code-to-hip-code)

[Writing Tests](#writing-tests)
 * [Testing know-how](#testing-know-how)
 * [Some general rules](#some-general-rules)
 * [Writing tests for kernels](#writing-tests-for-kernels)

[Documentation style](#documentation-style)
 * [Developer targeted notes](#developer-targeted-notes)
 * [Whitespaces](#whitespaces)
 * [Documenting examples](#documenting-examples)
 
[Other general programming comments](#other-general-programming-comments)
 * [C++ standard stream objects](#c++-standard-stream-objects)
 * [Warnings](#warnings)
 * [Avoiding circular dependencies](#avoiding-circular-dependencies)


## Most important stuff (A TL;DR)

* `GINKGO_DEVEL_TOOLS` needs to be set to `on` to commit. This requires
  `clang-format` to be installed. See [Automatic code
  formatting](#automatic-code-formatting) for more details. Once installed, you
  can run `make format` in your `build/` folder to automatically format your
  modified files. As `make format` unstages your files post-formatting, you must
  stage the files again once you have verified that `make format` has done the
  appropriate formatting, before committing the files.

* See [Our git workflow](#our-git-workflow) to get a quick overview of our
  workflow.

* See [Creating, Reviewing and Merging Pull
  Requests](#creating-reviewing-and-merging-pull-requests) on how to create a
  Pull request.


## Project structure

Ginkgo is divided into a `core` module with common functionalities independent
of the architecture, and several kernel modules (`reference`, `omp`, `cuda`,
`hip`) which contain low-level computational routines for each supported
architecture.

### Extended header files

Some header files from the core module have to be extended to include special
functionality for specific architectures. An example of this is
`core/base/math.hpp`, which has a GPU counterpart in `cuda/base/math.hpp`. For
such files you should always include the version from the module you are working
on, and this file will internally include its `core` counterpart.

### Using library classes

You can use and call functions of existing classes inside a kernel (that are
defined and not just declared in a header file), however, you are not allowed to
create new instances of a polymorphic class inside a kernel (or in general
inside any kernel module like cuda/hip/omp/reference) as this creates circular
dependencies between the `core` and the backend library. With this in mind, our
CI contains a job which checks if such a circular dependency exists.

For example, when creating a new matrix class `AB` by combining existing classes
`A` and `B`, the `AB::apply()` function composed of kernel invocations to
`A::apply()` and `B::apply()` can only be defined in the core module, it is not
possible to create instances of `A` and `B` inside the `AB` kernel file. An
example for such a class the `hybrid` matrix format.

## Git related

Ginkgo uses git, the distributed version control system to track code changes
and coordinate work among its developers. A general guide to git can be found in
[its extensive documentation](https://git-scm.com/docs).

### Our git workflow

In Ginkgo, we prioritize keeping a clean history over accurate tracking of
commits. `git rebase` is hence our command of choice to make sure that we have a
nice and linear history, especially for pulling the latest changes from the
`develop` branch. More importantly, rebasing upon develop is **required** before
the commits of the PR are merged into the `develop` branch.

### Writing good commit messages

With software sustainability and maintainability in mind, it is important to
write commit messages that are short, clear and informative. Ideally, this would
be the format to prefer:

```sh
Summary of the changes in a sentence, max 50 chars.

More detailed comments:
+ Changes that have been added.
- Changes that been removed.

Related PR: https://github.com/ginkgo-project/ginkgo/pull/<PR-number>
```

You can refer to [this informative
guide](https://chris.beams.io/posts/git-commit/) for more details.

#### Attributing credit

Git has a nice feature where it allows you to add a co-author for your commit,
if you would like to attribute credits for the changes made in the commit. This
can be done by:

```sh
Commit message.

Co-authored-by: Name <email@domain>
```

In the Ginkgo commit history, this is most common associated with suggested
improvements from code reviews.

### Creating, Reviewing and Merging Pull Requests

* The `develop` branch is the default branch to submit PR's to. From time to
  time, we merge the `develop` branch to the `master` branch and create tags on
  the `master` to create new releases of Ginkgo. Therefore, all pull requests
  must be merged into `develop`.
* Please have a look at the labels and make sure to add the relevant labels.
* You can mark the PR as a `WIP` if you are still working on it, `Ready for
  Review` when it is ready for others to review it.
* Assignees to the PR should be the ones responsible for merging that PR. Currently, it is only possible to assign members within the `ginkgo-project`.  
* Each pull request requires at least two approvals before merging.
* PR's created from within the repository will automatically trigger two CI
  pipelines on pushing to the branch from the which the PR has been created. The
  Github Actions pipeline tests our framework on Mac OSX and on Windows
  platforms. Another comprehensive Linux based pipeline is run from a [mirror on
  gitlab](https://gitlab.com/ginkgo-project/ginkgo-public-ci/pipelines) and
  contains additional checks like static analysis and test coverage.
* Once a PR has been approved and the build has passed, one of the reviewers can
  mark the PR as `READY TO MERGE`. At this point the creator/assignee of the PR
  *needs to* verify that the branch is up to date with `develop` and rebase it
  on `develop` if it is not.


## Code style

### Automatic code formatting

Ginkgo uses [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html)
(executable is usually named `clang-format`) and a custom `.clang-format`
configuration file (mostly based on ClangFormat's _Google_ style) to
automatically format your code. __Make sure you have ClangFormat set up and
running properly__ ( you should be able to run `make format` from Ginkgo's build
directory) before committing anything that will end up in a pull request against
`ginkgo-project/ginkgo` repository. In addition, you should __never__ modify the
`.clang-format` configuration file shipped with Ginkgo. E.g. if ClangFormat has
trouble reading this file on your system, you should install a newer version of
ClangFormat, and avoid commenting out parts of the configuration file.

ClangFormat is the primary tool that helps us achieve a uniform look of Ginkgo's
codebase, while reducing the learning curve of potential contributors. However,
ClangFormat configuration is not expressive enough to incorporate the entire
coding style, so there are several additional rules that all contributed code
should follow.

_Note_: To learn more about how ClangFormat will format your code, see existing
files in Ginkgo, `.clang-format` configuration file shipped with Ginkgo, and
ClangFormat's documentation.

### Naming scheme

#### Filenames 

Filenames use `snake_case` and use the following extensions:
*   C++ source files: `.cpp`
*   C++ header files: `.hpp`
*   CUDA source files: `.cu`
*   CUDA header files: `.cuh`
*   HIP source files: `.hip.cpp`
*   HIP header files: `.hip.hpp`
*   Common source files used by both CUDA and HIP: `.hpp.inc`
*   CMake utility files: `.cmake`
*   Shell scripts: `.sh`

_Note:_ A C++ source/header file is considered a `CUDA` file if it contains CUDA
code that is not guarded with `#if` guards that disable this code in non-CUDA
compilers. I.e. if a file can be compiled by a general C++ compiler, it is not
considered a CUDA file.

#### Macros

Macros (both object-like and function-like macros) use `CAPITAL_CASE`. They
have to start with `GKO_` to avoid name clashes (even if they are `#undef`-ed in
the same file!).

#### Variables

Variables use `snake_case`.

#### Constants

Constants use `snake_case`.

#### Functions

Functions use `snake_case`.

#### Structures and classes

Structures and classes which do not experience polymorphic behaviour (i.e. do
not contain virtual methods, nor members which experience polymorphic behaviour)
use `snake_case`.

All other structures and classes use `CamelCase`.

#### Members

All structure / class members use the same naming scheme as they would if they
were not members:
*   methods use the naming scheme for functions
*   data members the naming scheme for variables or constants
*   type members for classes / structures

Additionally, non-public data members end with an underscore (`_`).

#### Namespaces

Namespaces use `snake_case`.

#### Template parameters

* Type template parameters use `CamelCase`, for example `ValueType`.
* Non-type template parameters use `snake_case`, for example `subwarp_size`.

### Whitespace

Spaces and tabs are handled by ClangFormat, but blank lines are only partially
handled (the current configuration doesn't allow for more than 2 blank lines).
Thus, contributors should be aware of the following rules for blank lines:

1.  Top-level statements and statements directly within namespaces are separated
    with 2 blank lines. The first / last statement of a namespace is separated
    by two blank lines from the opening / closing brace of the namespace.
    1.  _exception_: if the first __or__ the last statement in the namespace is another namespace, then no blank lines are required  
        _example_:
        ```c++
        namespace foo {


        struct x {
        };


        }  // namespace foo


        namespace bar {
        namespace baz {


        void f();


        }  // namespace baz
        }  // namespace bar
        ```

    2.  _exception_: in header files whose only purpose is to _declare_ a bunch
        of functions (e.g. the `*_kernel.hpp` files) these declarations can be
        separated by only 1 blank line (note: standard rules apply for all other
        statements that might be present in that file)
    3.  _exception_: "related" statement can have 1 blank line between them.
        "Related" is not a strictly defined adjective in this sense, but is in
        general one of:

        1.  overload of a same function,
        2.  function / class template and it's specializations,
        3.  macro that modifies the meaning or adds functionality to the
            previous / following statement.

        However, simply calling function `f` from function `g` does not imply
        that `f` and `g` are "related".
2.  Statements within structures / classes are separated with 1 blank line.
    There are no blank lines betweeen the first / last statement in the
    structure / class.
    1.  _exception_: there is no blank line between an access modifier (`private`, `protected`, `public`) and the following statement.  
       _example_:
        ```c++
        class foo {
        public:
            int get_x() const noexcept { return x_; }

            int &get_x() noexcept { return x_; }

        private:
            int x_;
        };
        ```

3.  Function bodies cannot have multiple consecutive blank lines, and a single
    blank line can only appear between two logical sections of the function.
4. Unit tests should follow the [AAA](http://wiki.c2.com/?ArrangeActAssert)
   pattern, and a single blank line must appear between consecutive "A"
   sections. No other blank lines are allowed in unit tests.
5.  Enumeration definitions should have no blank lines between consecutive
    enumerators.


### Include statement grouping

In general, all include statements should be present on the top of the file,
ordered in the following groups, with two blank lines between each group:

1. Related header file (e.g. `core/foo/bar.hpp` included in `core/foo/bar.cpp`,
   or in the unit test`core/test/foo/bar.cpp`)
2. Standard library headers (e.g. `vector`)
3. Executor specific library headers (e.g. `omp.h`)
4. System third-party library headers (e.g. `papi.h`)
5. Local third-party library headers
6. Public Ginkgo headers
7. Private Ginkgo headers

_Example_: A file `core/base/my_file.cpp` might have an include list like this:

```c++
#include <ginkgo/core/base/my_file.hpp>


#include <algorithm>
#include <vector>
#include <tuple>


#include <omp.h>


#include <papi.h>


#include "third_party/blas/cblas.hpp"
#include "third_party/lapack/lapack.hpp"


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/my_file_kernels.hpp"
```

#### Main header

This section presents general rules used to define the main header attributed to
the file. In the previous example, this would be ` #include
<ginkgo/core/base/my_file.hpp>`.

General rules:
1. Some fixed main header.
2. components:
  - with `_kernel` suffix looks for the header in the same folder.
  - without `_kernel` suffix looks for the header in `core`.
3. `test/utils`: looks for the header in `core`
4. `core`: looks for the header in `ginkgo`
5. `test` or `base`: looks for the header in `ginkgo/core`
6. others: looks for the header in `core`

_Note_: Please see the detail in the `dev_tools/scripts/config`.


#### Automatic header arrangement

1. `dev_tools/script/format_header.sh` will take care of the group/sorting of
   headers according to this guideline.
2. `make format_header` arranges the header of the modified files in the branch.
3. `make format_header_all` arranges the header of all files.


### Other Code Formatting not handled by ClangFormat

#### Control flow constructs

Single line statements should be avoided in all cases. Use of brackets is
mandatory for all control flow constructs (e.g. `if`, `for`, `while`, ...).

#### Variable declarations

C++ supports declaring / defining multiple variables using a single
_type-specifier_. However, this is often very confusing as references and
pointers exhibit strange behavior:

```c++
template <typename T> using pointer = T *;

int *        x, y;  // x is a pointer, y is not
pointer<int> x, y;  // both x and y are pointers
```

For this reason, __always__ declare each variable on a separate line, with its
own _type-specifier_.

### CMake coding style

#### Whitespaces
All alignment in CMake files should use four spaces.

#### Use of macros vs functions

Macros in CMake do not have a scope. This means that any variable set in this
macro will be available to the whole project. In contrast, functions in CMake
have local scope and therefore all set variables are local only. In general,
wrap all piece of algorithms using temporary variables in a function and use
macros to propagate variables to the whole project.

#### Naming style

All Ginkgo specific variables should be prefixed with a `GINKGO_` and all
functions by `ginkgo_`.


## Helper scripts

To facilitate easy development within Ginkgo and to encourage coders and
scientists who do not want get bogged down by the details of the Ginkgo library,
but rather focus on writing the algorithms and the kernels, Ginkgo provides the
developers with a few helper scripts.

### Create a new algorithm

A `create_new_algorithm.sh` script is available for developers to facilitate
easy addition of new algorithms. The options it provides can be queried with

```sh
./create_new_algorithm.sh --help
```
The main objective of this script is to add files and boiler plate code for the new algorithm using a model and an instance of that model. For example, models can be any one of `factorization`, `matrix`, `preconditioner` or `solver`. For example to create a new solver named `my_solver` similar to `gmres`, you would set the `ModelType` to `solver` and set the `ModelName` to `gmres`. This would duplicate the core algorithm and kernels of the `gmres` algorithm and replace the naming to `my_solver`. Additionally, all the kernels of the new `my_solver` are marked as `GKO_NOT_IMPLEMENTED`. For easy navigation and `.txt` file is created in the folder where the script is run, which lists all the TODO's. These TODO's can also be found in the corresponding files.

### Converting CUDA code to HIP code

This script calls the `hipify` script provided by HIP converting the CUDA syntax
to HIP syntax. Additionally, it also automatically replaces the instances of
CUDA with HIP as appropriate. Hence, this script can be called on a Ginkgo CUDA
file.


## Writing Tests

Ginkgo uses the [GTest framework](https://github.com/google/googletest) for the
unit test framework within Ginkgo. Writing good tests are extremely important to
verify the functionality of the new code and to make sure that none of the
existing code has been broken.

### Testing know-how

* GTest provides a [comprehensive documentation](https://github.com/google/googletest/blob/master/googletest/docs/primer.md) of the functionality available within Gtest.  
* Reduce code duplication with [Testing Fixtures,
  `TEST_F`](https://github.com/google/googletest/blob/master/googletest/docs/primer.md#test-fixtures-using-the-same-data-configuration-for-multiple-tests-same-data-multiple-tests)
* Write templated tests using
  [`TYPED_TEST`](https://github.com/google/googletest/blob/master/googletest/docs/advanced.md#typed-tests).

### Some general rules.

* Unit tests must follow the [KISS
  principle](https://en.wikipedia.org/wiki/KISS_principle).
* Unit tests must follow the [AAA](http://wiki.c2.com/?ArrangeActAssert)
  pattern, and a single blank line must appear between consecutive "A" sections.

### Writing tests for kernels

* Reference kernels, kernels on the `ReferenceExecutor`, are meant to be single
  threaded reference implementations. Therefore, tests for reference kernels
  need to be performed with data that can be as small as possible. For example,
  matrices lesser than 5x5 are acceptable. This allows the reviewers to verify
  the results for exactness with tools such as MATLAB.
* OpenMP, CUDA and HIP kernels have to be tested against the reference kernels.
  Hence data for the tests of these kernels can be generated in the test files
  using helper functions or by using external files to be read through the
  standard input. In particular for CUDA and HIP, the data size should be at
  least bigger than the architecture's warp size to ensure there is no corner
  case in the kernels.


## Documentation style

Documentation uses standard Doxygen.

###  Developer targeted notes
Make use of `@internal` doxygen tag. This can be used for any comment which is
not intended for users, but is useful to better understand a piece of code.

### Whitespaces

#### After named tags such as `@param foo`

The documentation tags which use an additional name should be followed by two
spaces in order to better distinguish the text from the doxygen tag. It is also
possible to use a line break instead.

### Documenting examples

There are two main steps:

1. First, you can just copy over the [`doc/`](https://github.com/ginkgo-project/ginkgo/tree/develop/examples/simple-solver) folder (you can copy it from the example most relevant to you) and adapt your example names and such, then you can modify the actual documentation.  
+ In `tooltip`: A short description of the example.
+ In `short-intro`: The name of the example.
+ In `results.dox`: Run the example and write the output you get.
+ In `kind`: The kind of the example. For different kinds see [the
  documentation](https://ginkgo-project.github.io/ginkgo/doc/develop/Examples.html).
  Examples can be of `basic`, `techniques`, `logging`, `stopping_criteria` or
  `preconditioners`. If your example does not fit any of these categories, feel
  free to create one.
+ In `intro.dox`: You write an explanation of your code with some introduction
  similar to what you see in an existing example most relevant to you.
+ In `builds-on`: You write the examples it builds on.

2. You also need to modify the
   [examples.hpp.in](https://github.com/ginkgo-project/ginkgo/blob/develop/doc/examples/examples.hpp.in)
   file. You add the name of the example in the main section and in the section
   that you specified in the `doc/kind` file in the example documentation.


## Other programming comments

### C++ standard stream objects

These are global objects and are shared inside the same translation unit.
Therefore, whenever its state or formatting is changed (e.g. using `std::hex` or
floating point formatting) inside library code, make sure to restore the state
before returning the control to the user. See this [stackoverflow
question](https://stackoverflow.com/questions/2273330/restore-the-state-of-stdcout-after-manipulating-it)
for examples on how to do it correctly. This is extremely important for header
files.

### Warnings

By default, the `-DGINKGO_COMPILER_FLAGS` is set to `-Wpedantic` and hence
pedantic warnings are emitted by default. Some of these warnings are false
positives and a complete list of the currently known warnings and their
solutions is listed in [Issue
174](https://github.com/ginkgo-project/ginkgo/issues/174). Specifically, when
macros are being used, we have the issue of having `extra ;` warnings, which is
resolved by adding a `static_assert()`. The CI system additionally also has a
step where it compiles for pedantic warnings to be errors.

### Avoiding circular dependencies

To avoid circular dependencies, it is forbidden inside the kernel modules
(`ginkgo_cuda`, `ginkgo_omp`, `ginkgo_reference`) to use functions implemented
only in the `core` module (using functions implemented in the headers is fine).
In practice, what this means is that it is required that any commit to Ginkgo
passes the `no-circular-deps` CI step. For more details, see [this
pipeline](https://gitlab.com/ginkgo-project/ginkgo-public-ci/pipelines/52941979),
where Ginkgo did not abide to this policy and [PR
#278](https://github.com/ginkgo-project/ginkgo/pull/278) which fixed this. Note
that doing so is not enough to guarantee with 100% accuracy that no circular
dependency is present. For an example of such a case, take a look at [this
pipeline](https://gitlab.com/ginkgo-project/ginkgo-public-ci/pipelines/53006772)
where one of the compiler setups detected an incorrect dependency of the `cuda`
module (due to jacobi) on the `core` module.
