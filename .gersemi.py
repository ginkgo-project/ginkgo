from gersemi.builtin_commands import builtin_commands

new_target_sources = builtin_commands["target_sources"]
new_target_sources["keyword_preprocessors"] = {
    key: "sort+unique" for key in new_target_sources["multi_value_keywords"]
}

new_target_compile_definitions = builtin_commands["target_compile_definitions"]
new_target_compile_definitions["keyword_preprocessors"] = {
    key: "sort+unique" for key in new_target_compile_definitions["multi_value_keywords"]
}

command_definitions = {
    "target_sources": new_target_sources,
    "target_compile_definitions": new_target_compile_definitions
}
