add_executable(tests
    catch2/catch_amalgamated.cpp
    test_lexer.cpp
    test_parser.cpp
)
target_link_libraries(tests PRIVATE gpu_emu_lib)
add_test(NAME all_tests COMMAND tests)