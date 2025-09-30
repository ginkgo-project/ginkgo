message(STATUS "Fetching external json-schema-validator")
include(FetchContent)
FetchContent_Declare(
    json_schema_validator
    GIT_REPOSITORY https://github.com/pboettch/json-schema-validator.git
    GIT_TAG 2.3.0
)
FetchContent_MakeAvailable(json_schema_validator)
