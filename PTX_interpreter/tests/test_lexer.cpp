#include "catch2/catch_amalgamated.hpp"
#include "ptx_lexer.h"

TEST_CASE("Tokenizer handles directives", "[lexer]") {
    auto tokens = tokenize(".version 7.0\n.target sm_80");
    REQUIRE(tokens[0].type == TokenType::Directive);
    REQUIRE(tokens[0].value == ".version");
    REQUIRE(tokens[2].type == TokenType::Directive);
    REQUIRE(tokens[2].value == ".target");
}

TEST_CASE("Tokenizer handles identifiers", "[lexer]") {
    auto tokens = tokenize("%r0 param_a add_kernel");
    REQUIRE(tokens[0].value == "%r0");
    REQUIRE(tokens[1].value == "param_a");
    REQUIRE(tokens[2].value == "add_kernel");
}

TEST_CASE("Tokenizer strips comments", "[lexer]") {
    auto tokens = tokenize("add.u32 // this is a comment\n%r0");
    // Should only see identifier tokens, no comment content
    for (auto& t : tokens)
        REQUIRE(t.value.find("comment") == std::string::npos);
}

TEST_CASE("Tokenizer handles braces and punctuation", "[lexer]") {
    auto tokens = tokenize("{ .reg .u32 %r<4>; }");
    REQUIRE(tokens[0].type == TokenType::LBrace);
    REQUIRE(tokens[4].type == TokenType::Semicolon);
    REQUIRE(tokens[5].type == TokenType::RBrace);
}