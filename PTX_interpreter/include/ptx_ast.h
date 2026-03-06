#pragma once
#include <string>
#include <vector>

enum class PTXType {
    U8, U16, U32, U64,
    S8, S16, S32, S64,
    F16, F32, F64,
    Pred,
    Unknown
};

struct PTXRegDecl {
    PTXType type;
    std::string baseName;  // "%r"
    int count;             // 4 from ".reg .u32 %r<4>"
};

struct PTXParam {
    PTXType type;
    std::string name;      // "param_a"
};

struct PTXInstruction {
    std::string label;     // "LOOP" if this instruction is a label target, else ""
    std::string predicate; // "@%p0" or ""
    std::string opcode;    // "ld.param.u64"
    std::string dest;      // "%rd0" or ""
    std::vector<std::string> sources; // operands
};

struct PTXKernel {
    std::string name;
    std::vector<PTXParam> params;
    std::vector<PTXRegDecl> regDecls;
    std::vector<PTXInstruction> instructions;
};

struct PTXModule {
    std::string version;   // "7.0"
    std::string target;    // "sm_80"
    std::vector<PTXKernel> kernels;
};

// PTXModule
// ├── version: "7.0"
// ├── target: "sm_80"
// └── kernels[]
//     └── PTXKernel
//         ├── name: "add_kernel"
//         ├── params[]
//         │   ├── PTXParam { U64, "param_a" }
//         │   └── PTXParam { U64, "param_b" }
//         ├── regDecls[]
//         │   └── PTXRegDecl { U32, "%r", 4 }
//         └── instructions[]
//             ├── PTXInstruction { "", "", "ld.param.u64", "%rd0", ["param_a"] }
//             └── PTXInstruction { "", "", "add.u32", "%r2", ["%r0", "%r1"] }