
# Our goal is to write something that parses valid 8086 assembly code and outputs the corresponding machine code.
# We will use the following instructions:
# MOV, ADD, sub, MUL, DIV, AND, OR, XOR, cmp, JMP, JE, JNE, JG, JGE, JL, JLE, CALL, RET, INT, NOP
# We will also use the following registers:
# AX, BX, CX, DX, SI, DI, SP, BP
# We will also use the following memory addressing modes:
# [register], [register+register], [register+constant], [constant]
# We will also use the following data types:
# db, dw, dd
# We will also use the following directives: 
# ORG, END
# We will also use the following comments:
# ;
# We will also use the following labels:
# label:
# We will also use the following stack operations:
# PUSH, POP
# We will also use the following conditional jumps:
# JE, JNE, JG, JGE, JL, JLE
# We will also use the following interrupts:
# INT 21h
# We will also use the following segment registers:
# CS, DS, SS, ES
# We will also use the following segment override prefixes:
# CS:, DS:, SS:, ES:
# We will also use the following segment override prefixes:
# CS:, DS:, SS:, ES: 
import argparse
import subprocess

from functools import wraps
from typing import List, Optional, Tuple

from itertools import count

gen = (f"label{n}" for n in count())

def debugBytesStr(line):
    bitStr = ''.join(f'{byte:08b}' for byte in line)
    print(bitStr)

def debugBytesNum(i):
    print(f'{i:08b}')

OP83_TABLE = {
    0: "add",
    1: "or",
    2: "adc",
    3: "sbb",
    4: "and",
    5: "sub",
    6: "xor",
    7: "cmp",
}


MATH_ACC_OPCODES = {
    0x04: ('add', 'al'),
    0x05: ('add', 'ax'),
    0x2C: ('sub', 'al'),
    0x2D: ('sub', 'ax'),
    0x3C: ('cmp', 'al'),
    0x3D: ('cmp', 'ax'),

}

MATH_REG_OPCODES = {
    # mnemonic, w, d
    0x00: ('add', 0, 0),
    0x01: ('add', 1, 0),
    0x02: ('add', 0, 1),
    0x03: ('add', 1, 1),

    0x28: ('sub', False, 0),
    0x29: ('sub', True,  0),
    0x2A: ('sub', False, 1),
    0x2B: ('sub', True,  1),

    0x38: ('cmp', False, 0),
    0x39: ('cmp', True,  0),
    0x3A: ('cmp', False, 1),
    0x3B: ('cmp', True,  1),
}

MATH_IMM_RM_OPCODES = {
    0x80: { "commands": {0: 'add', 5: 'sub', 7: 'cmp'}, "modifier": "byte", "bytes": 1},
    0x81: { "commands": {0: 'add', 5: 'sub', 7: 'cmp'}, "modifier": "word", "bytes": 2},
    0x83: { "commands": {0: 'add', 5: 'sub', 7: 'cmp'}, "modifier": "word", "bytes": 1},
}

COND_JUMP_OPCODES = {
    0x70: 'jo',
    0x71: 'jno',
    0x72: 'jb',
    0x73: 'jnb',
    0x74: 'je',
    0x75: 'jne',
    0x76: 'jbe',
    0x77: 'ja',
    0x78: 'js',
    0x79: 'jns',
    0x7A: 'jp',
    0x7B: 'jnp',
    0x7C: 'jl',
    0x7D: 'jnl',
    0x7E: 'jle',
    0x7F: 'jg',
    0xE0: 'loopnz',
    0xE1: 'loopz',
    0xE2: 'loop',
    0xE3: 'jcxz',
}

class ASMParser:
    # This is a global counter and you may eventually want it not to be that.  
    _label_gen = (f"label{n}" for n in count())

    @classmethod
    def next_label(cls):
        return next(cls._label_gen)

    @classmethod
    def getRegisterName(cls, register : int, word: bool):
        reg_table_16 = ['ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di']
        reg_table_8  = ['al', 'cl', 'dl', 'bl', 'ah', 'ch', 'dh', 'bh']
        registers = reg_table_16 if word else reg_table_8
        return registers[register]

    @classmethod
    def getRegisterNameMemory(cls, register : int, displacement : Optional[int] = None, force_zero : bool = False) -> str:
        if displacement is None: 
            displacement = 0
        reg_table = ['bx + si', 'bx + di', 'bp + si', 'bp + di', 'si', 'di', 'bp', 'bx']
        reg = reg_table[register]
        if displacement != 0 or force_zero:
            if displacement > 0:
                reg += f" + {displacement}"
            elif displacement < 0:
                reg += f" - {-displacement}"
            elif "+" in reg:
                pass
            else:
                reg += " + 0"
        return "[" + reg + "]"

    def __init__(self, asm_bytes : List[bytes], bits : int):
        self.asm_bytes = asm_bytes
        self.bits = bits
        self.pc = 0
        self.labels = {}
    
    def write(self, filename):
        print(f"Lines: {len(self.lines)}")
        print(f"Labels: {len(self.labels)}")
        with open(filename, 'w') as f:
            f.write(f"bits {self.bits}\n")
            for linenum, line in self.lines.items():
                if self.labels.get(linenum):
                    f.write(f"{self.labels[linenum]}:\n")
                f.write(line + "\n")

    def parse(self):
        lines = {}
        try: 
            while self.pc < len(self.asm_bytes):
                old_pc = self.pc
                line = self.parseInstruction()
                lines[old_pc] = line
        except Exception as e:
            print(f"Error at line {self.pc}: {e}")
        finally: 
            print("HERE")
            self.lines = lines

    def parseInstruction(self) -> str:
        # opcode is the first byte sort of
        if self.asm_bytes[self.pc] >> 2 == 0b100010 or self.asm_bytes[self.pc] >> 1 == 0b1100011:
            return self.parseMOV()
        elif self.asm_bytes[self.pc] >> 2 == 0b101000:
            return self.parseMOVValue()
        elif self.asm_bytes[self.pc] >> 4 == 0b1011:
            return self.parseMOVDirect()
        elif self.asm_bytes[self.pc] in MATH_REG_OPCODES or self.asm_bytes[self.pc] in MATH_IMM_RM_OPCODES or self.asm_bytes[self.pc] in MATH_ACC_OPCODES:
            return self.parseMath()
        elif self.asm_bytes[self.pc] == 0x90:
            self.pc += 1
            return "nop"
        elif self.asm_bytes[self.pc] in COND_JUMP_OPCODES:
            return self.parseJump()
        else:
            print("Invalid opcode %s which is %s in hex and %s in binary at position %d" % 
                             (self.asm_bytes[self.pc], 
                              hex(self.asm_bytes[self.pc]), 
                              bin(self.asm_bytes[self.pc]), 
                              self.pc))
            return ""
        
    def getBytes(self, num_bytes : int) -> int:
        ret = 0
        for i in range(num_bytes-1, -1, -1):
            ret = (ret << 8) | self.asm_bytes[self.pc + i]
        self.pc += num_bytes
        return ret
    
    def getIntegerFromBytes(self, num_bytes : int, signed : bool = True) -> int:
        if num_bytes == 0:
            return 0
        ret = self.getBytes(num_bytes)
        if signed:
            sign_bit = 1 << (num_bytes * 8 - 1)
            ret = (ret ^ sign_bit) - sign_bit
        return ret

    def getModRM(self) -> Tuple[int, int, int]:
        modrm = self.getBytes(1)
        mod = modrm >> 6
        reg = (modrm >> 3) & 0x7
        rm = modrm & 0x7
        return mod, reg, rm
    
    def parseDirectMemory(self, mod : int, reg : int, rm : int, word : bool, 
                          lam : callable, modifier : Optional[str] = None, 
                          src_lambda : Optional[callable] = None, 
                          force_zero: bool = False):
        if src_lambda is None: 
            src_lambda = lambda: self.getRegisterName(reg, word) 
        if modifier is None:
            modifier = ""
        elif not modifier.endswith(" "):
            modifier += " "
        if mod == 0b11:
            src, dest = src_lambda(), self.getRegisterName(rm, word)
        else: 
            displacement = self.getIntegerFromBytes(mod, signed = True)
            dest = modifier
            if mod == 0b00 and rm == 0b110:
                # pull two bytes from the instruction
                direct_memory_value = self.getIntegerFromBytes(2)
                dest += f"[{direct_memory_value}]"
            else:
                dest += self.getRegisterNameMemory(rm, displacement, force_zero=force_zero)
            src = lam()
        return src, dest
    
    def parser(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instruction = self.getBytes(1)
            # print("Instruction: ", instruction, "is", hex(instruction), "in hex")
            # call the real function — should return (mnemonic, src, dest, d)
            mnemonic, src, dest, d = fn(self, instruction)
            if d:
                src, dest = dest, src

            if dest:
                return f"{mnemonic} {dest}, {src}"
            return f"{mnemonic} {src}"
        return wrapper
    
    @parser
    def parseJump(self, instruction : int) -> str:
        jump = self.getIntegerFromBytes(1, signed=True)
        #NASM refuses to do offsets approporiately so we need to make fake labels.  
        label = self.labels.get(self.pc + jump)
        if not label:
            label = self.labels[self.pc + jump] = self.next_label()
        return COND_JUMP_OPCODES[instruction], label, None, None

    @parser
    def parseMath(self, instruction : int) -> Tuple[str, str, str, bool]:
        if instruction in MATH_ACC_OPCODES:
            mnemonic, reg = MATH_ACC_OPCODES[instruction]
            imm = self.getIntegerFromBytes(2 if reg == "ax" else 1)
            return (mnemonic, imm, reg, False)

        d = False
        mod, reg, rm = self.getModRM()
        if instruction in MATH_REG_OPCODES:
            mnemonic, word, d = MATH_REG_OPCODES[instruction]
            src, dest = self.parseDirectMemory(mod, reg, rm, word, lambda: self.getRegisterName(reg, word), force_zero = True)
        elif instruction == 0x83:
            modifier = MATH_IMM_RM_OPCODES[instruction]["modifier"]
            mnemonic = OP83_TABLE.get(reg, f"OP{reg}")
            src, dest = self.parseDirectMemory(mod, reg, rm, True, lambda: self.getIntegerFromBytes(1, signed=True), modifier, lambda : self.getIntegerFromBytes(1), force_zero = True)
        elif instruction in MATH_IMM_RM_OPCODES:
            mnemonic = MATH_IMM_RM_OPCODES[instruction]["commands"][reg]
            num_bytes = MATH_IMM_RM_OPCODES[instruction]["bytes"]
            modifier = MATH_IMM_RM_OPCODES[instruction]["modifier"]
            dest = modifier + " " + self.getRegisterNameMemory(rm)
            imm = self.getIntegerFromBytes(num_bytes, signed = True)    
            src = imm
        return mnemonic, src, dest, d

    @parser
    def parseMOVValue(self, instruction : int) -> Tuple[str, str, str, bool]:
        d = (instruction >> 1) & 0x1
        w = instruction & 0x1
        num_bytes = 2 if w else 1
        # if instruction starts with 0xA0 in the first six bits, then the instruction is MOV but we just pull direct memory address
        
        dest = "ax" if w else "al"
        src = "[" + str(self.getIntegerFromBytes(num_bytes)) + "]"
        # Print bytes of src
        return "mov", src, dest, d
    
    @parser
    def parseMOV(self, instruction : int) -> str:
        # the first 6 bytes are the opcode
        # the next bit is the d bit
        # the next bit is the w bit
        d = (instruction >> 1) & 0x1
        word = instruction & 0x1
        num_bytes = 2 if word else 1
        
        mod, reg, rm = self.getModRM()
        src, dest = self.parseDirectMemory(mod, reg, rm, word, lambda: self.getRegisterName(reg, word))
        
        # If the instruction is 0xc6, then the instruction is a MOV but with fixed inputs
        if instruction & 0xF6 == 0xc6:
            d = 0
            modifier = "word " if word else "byte "
            direct_memory_address = self.getIntegerFromBytes(num_bytes)
            # unset d bit
            src = modifier + str(direct_memory_address)

        return "mov", src, dest, d
    
    def parseMOVDirect(self) -> Tuple[str, str, str, bool]:
        # the first 4 bytes are the opcode
        mnenomic = "mov"
        instruction = self.getBytes(1)
        word = (instruction >> 3) & 0x1
        reg = instruction & 0x7
        immediate = self.getIntegerFromBytes(word + 1)
        return mnenomic, self.getRegisterName(reg, word), immediate, False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compile 8086 assembly')
    parser.add_argument('input', type=str, help='Input file')
    parser.add_argument('--output', type=str, required=False, default=None, help='Output file')
    parser.add_argument('--bits', type=int, default=16, help='Number of bits')

    args = parser.parse_args()
    if args.output is None:
        args.output = args.input + ".bin"
    return args

def readAssembly(filename) -> bytes:
    with open(filename, 'rb') as f:
        raw = f.read()
        return raw

def main() -> None:
    args = parse_args()
    # The input is a bit file with 8086 assembly code
    # The output is a disassembled file with 8086 assembly code in strings
    print("Input: ", args.input)
    print("Output: ", args.output)
    print("Bits: ", args.bits)
    asm_bytes = readAssembly(args.input)
    comparison=asm_bytes
    
    parser = ASMParser(asm_bytes, args.bits)
    parser.parse()
    parser.write(args.output)
    
    
    # Call nasm to assemble the output
    test_file = args.output.replace(".bin", ".o")
    subprocess.check_output(["nasm", "-f", "bin", "-o", test_file,  args.output])
    output = readAssembly(test_file)

    if output == comparison:
        print("Success")
    else:
        print("Failure")
        print("Expected: ", comparison)
        print("Received: ", output)

if __name__ == "__main__":
    main()
