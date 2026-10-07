# Summary
<!-- (Mandatory) short summary of this PR. -->

# Details
<!-- (Optional) more details about the PR. -->

# Reminders 
<!-- For developers and reviewers because most of them are not checked automatically. 

Ensure you have set up the pre-commit and the corresponding pre-commit hook.  
Use `pre-commit run --from-ref origin/develop --to-ref HEAD` to ensure the branch changes compared to origin/develop are formatted through pre-commit.

Also, check https://github.com/ginkgo-project/ginkgo/wiki/Contributing-guidelines regarding Ginkgo's `naming-scheme`, `Whitespace`, `Other Code Formatting` not handled by ClangFormat. 

Please delete the irrelevant entries in the following.
-->
## Creating/Modifying classes 
- [ ] have entries in test/test_install
- [ ] If it is a LinOpFactory, add the corresponding key in `core/config/registry.cpp:generate_config_map`
- [ ] If any changes are introduced in LinOpFactory parameters, add the changes into the corresponding parse function
## Creating/Modifying functions
- [ ] have the corresponding tests
## Introducing new files
- [ ] ensure they are added in corresponding CMakeLists.txt
## Creating/Modifying kernels
- [ ] have the corresponding entries in core/device_hooks/common_kernels.inc.cpp
- [ ] check whether the changes are also required for other backends
## Being a new contributor introducing new stuff or performance improvement
- [ ] add yourself into contributors.txt with `LastName FirstName <email> affiliation` in alphabetical order
