# Security Policy

The Ginkgo team takes the security of the library seriously. This document
describes which versions receive security fixes, what we consider a security
issue, and how to report one privately.

## Supported Versions

Security fixes are applied to the `develop` branch and are included in the next
release of Ginkgo. Depending on the severity of the issue, we may also publish a
patch release for the latest release. Older releases do not receive security
fixes, so we recommend always using the latest release.

| Version                              | Supported          |
| ------------------------------------ | ------------------ |
| `develop` branch                     | :white_check_mark: |
| Latest release (see [Releases][rel]) | :white_check_mark: |
| Older releases                       | :x:                |

[rel]: https://github.com/ginkgo-project/ginkgo/releases

## Scope

Ginkgo is a numerical linear algebra library, so most bugs are correctness or
performance issues rather than security issues. Please report an issue privately
if you believe it can be exploited, for example:

* Memory safety bugs, such as out-of-bounds reads or writes, use-after-free, or
  integer overflows in size computations that lead to memory corruption, that
  can be triggered through Ginkgo's public interface.
* Bugs in code that parses external input, such as the matrix readers in
  `gko::read`, `gko::read_raw` and `gko::read_binary`, or the JSON and YAML
  configuration readers in `ginkgo/extensions/config`, which can be exploited
  with a crafted input file.
* Issues in Ginkgo's build system, CI workflows, or release process that could
  allow tampering with the source code, the build, or the released artifacts.

The following are usually not security issues and should be reported through a
[public issue](https://github.com/ginkgo-project/ginkgo/issues/new):

* Wrong results, missing convergence, or loss of accuracy.
* Crashes or excessive resource usage caused by inputs that violate the
  documented preconditions of an interface.
* Vulnerabilities in third-party dependencies (for example vendor libraries
  such as CUDA, ROCm, oneAPI or MPI). Please report these to the respective
  project. If Ginkgo's use of the dependency makes the issue exploitable, let us
  know as well.

If you are unsure whether an issue is security-related, please report it
privately. We would rather receive a report that turns out to be harmless than
miss a real problem.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, use one of the following private channels:

1. **GitHub private vulnerability reporting (preferred):** open the
   [Security tab](https://github.com/ginkgo-project/ginkgo/security) of the
   repository and click "Report a vulnerability", or go directly to the
   [new advisory form](https://github.com/ginkgo-project/ginkgo/security/advisories/new).
   This creates a private advisory visible only to you and the Ginkgo
   maintainers.
2. **Email:** send a message to
   [security@gko-project.org](mailto:security@gko-project.org).

To help us understand and resolve the issue quickly, please include as much of
the following information as you can:

* The affected Ginkgo version or commit hash.
* The executor(s) and backend(s) involved (Reference, OpenMP, CUDA, HIP, SYCL,
  MPI), together with the compiler, operating system, and relevant CMake
  options.
* A description of the issue and its potential impact.
* Steps to reproduce the issue, ideally a minimal program and any input files
  needed to trigger it.
* Any suggestions for a fix, if you have them.

## What to Expect

* We will acknowledge your report within 5 working days.
* We will investigate the issue and keep you informed about our progress. We may
  ask you for additional information.
* Once the issue is confirmed, we will develop a fix in a private fork through a
  GitHub security advisory and, where appropriate, request a CVE identifier.
* After a fix is available, we will publish the security advisory and the fix.
  We will credit you in the advisory unless you prefer to remain anonymous.

We follow a coordinated disclosure process and ask that you do not disclose the
issue publicly until a fix has been released, or until we have agreed on a
disclosure date with you. We aim to release a fix within 90 days of the initial
report.

## Other Questions

For questions about using Ginkgo, please use
[GitHub Discussions](https://github.com/ginkgo-project/ginkgo/discussions). For
bugs and feature requests that are not security-related, please
[create an issue](https://github.com/ginkgo-project/ginkgo/issues/new).
