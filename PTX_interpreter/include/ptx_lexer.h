#pragma once
#include <string>
#include <vector>

enum class TokenType {
    Directive,    // .version, .reg, .entry, etc.
    Identifier,   // %r0, param_a, LOOP
    Opcode,       // ld.param.u64, add.u32
    LBrace, RBrace, LBracket, RBracket, LParen, RParen,
    Comma, Semicolon, Colon, At,
    Integer, Float,
    EndOfFile
};

struct Token {
    TokenType type;
    std::string value;
    int line;
};

std::vector<Token> tokenize(const std::string& source);