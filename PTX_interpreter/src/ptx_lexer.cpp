#include "ptx_lexer.h"
#include <sstream>
#include <stdexcept>

std::vector<Token> tokenize(const std::string& source) {
    std::vector<Token> tokens;
    int i = 0, line = 1;
    int n = source.size();

    auto peek = [&](int offset = 0) -> char {
        return (i + offset < n) ? source[i + offset] : '\0';
    };

    while (i < n) {
        // Skip whitespace
        if (isspace(peek())) {
            if (peek() == '\n') line++;
            i++; continue;
        }

        // Skip line comments
        if (peek() == '/' && peek(1) == '/') {
            while (i < n && peek() != '\n') i++;
            continue;
        }

        // Skip block comments
        if (peek() == '/' && peek(1) == '*') {
            i += 2;
            while (i < n && !(peek() == '*' && peek(1) == '/')) {
                if (peek() == '\n') line++;
                i++;
            }
            i += 2; continue;
        }

        // Single-char tokens
        char c = peek();
        if (c == '{') { tokens.push_back({TokenType::LBrace, "{", line}); i++; continue; }
        if (c == '}') { tokens.push_back({TokenType::RBrace, "}", line}); i++; continue; }
        if (c == '[') { tokens.push_back({TokenType::LBracket, "[", line}); i++; continue; }
        if (c == ']') { tokens.push_back({TokenType::RBracket, "]", line}); i++; continue; }
        if (c == '(') { tokens.push_back({TokenType::LParen, "(", line}); i++; continue; }
        if (c == ')') { tokens.push_back({TokenType::RParen, ")", line}); i++; continue; }
        if (c == ',') { tokens.push_back({TokenType::Comma, ",", line}); i++; continue; }
        if (c == ';') { tokens.push_back({TokenType::Semicolon, ";", line}); i++; continue; }
        if (c == '@') { tokens.push_back({TokenType::At, "@", line}); i++; continue; }

        // Directives: start with '.'
        if (c == '.') {
            std::string val;
            val += c; i++;
            while (i < n && (isalnum(peek()) || peek() == '_')) {
                val += peek(); i++;
            }
            tokens.push_back({TokenType::Directive, val, line});
            continue;
        }

        // Identifiers and opcodes: letters, digits, '_', '%', '<', '>'
        if (isalpha(c) || c == '%' || c == '_') {
            std::string val;
            while (i < n && (isalnum(peek()) || peek() == '_' || peek() == '%'
                              || peek() == '<' || peek() == '>')) {
                val += peek(); i++;
            }
            // Check if next non-space char is ':' -> it's a label
            // Dot-separated opcode continuation (e.g. "ld" followed by ".param")
            // We'll handle opcode joining in the parser for simplicity
            if (peek() == ':') {
                tokens.push_back({TokenType::Colon, ":", line});
            }
            tokens.push_back({TokenType::Identifier, val, line});
            continue;
        }

        // Numbers
        if (isdigit(c) || (c == '-' && isdigit(peek(1)))) {
            std::string val;
            val += c; i++;
            while (i < n && (isdigit(peek()) || peek() == '.' || peek() == 'x'
                              || isxdigit(peek()))) {
                val += peek(); i++;
            }
            tokens.push_back({TokenType::Integer, val, line});
            continue;
        }

        throw std::runtime_error("Unexpected character '" + std::string(1, c)
                                 + "' at line " + std::to_string(line));
    }

    tokens.push_back({TokenType::EndOfFile, "", line});
    return tokens;
}