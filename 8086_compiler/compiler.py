
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

from typing import List, Optional, Tuple

def debugBytesStr(line):
    bitStr = ''.join(f'{byte:08b}' for byte in line)
    print(bitStr)

def debugBytesNum(i):
    print(f'{i:08b}')

def getRegisterName(register : int, word: bool):
    reg_table_16 = ['ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di']
    reg_table_8  = ['al', 'cl', 'dl', 'bl', 'ah', 'ch', 'dh', 'bh']
    registers = reg_table_16 if word else reg_table_8
    return registers[register]

def getRegisterNameMemory(register : int, displacement : Optional[int] = None, force_zero : bool = False) -> str:
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
    0x80: { "commands": {0: 'add', 5: 'sub', 7: 'cmp'}, "word": "byte", "bytes": 1},
    0x81: { "commands": {0: 'add', 5: 'sub', 7: 'cmp'}, "word": "word", "bytes": 2},
    0x83: { "commands": {0: 'add', 5: 'sub', 7: 'cmp'}, "word": "word", "bytes": 1},
}

class ASMParser:
    def __init__(self, asm_bytes : List[bytes]):
        self.asm_bytes = asm_bytes
        self.pc = 0
    
    def parse(self):
        max_count = 100
        while self.pc < len(self.asm_bytes):
            line = self.parseInstruction()
            yield line + "\n"
            print(line)
            max_count -= 1
            if max_count == 0:
                break
        
    
    def parseInstruction(self) -> str:
        # opcode is the first byte sort of
        if self.asm_bytes[self.pc] >> 2 == 0b100010:
            return self.parseMOV()
        elif self.asm_bytes[self.pc] >> 1 == 0b1100011:
            return self.parseMOV()
        elif self.asm_bytes[self.pc] >> 2 == 0b101000:
            return self.parseMOVValue()
        elif self.asm_bytes[self.pc] >> 4 == 0b1011:
            return self.parseMOVDirect()
        elif self.asm_bytes[self.pc] in MATH_REG_OPCODES:
            return self.parseMath()
        elif self.asm_bytes[self.pc] in MATH_IMM_RM_OPCODES:
            return self.parseMath()
        elif self.asm_bytes[self.pc] in MATH_ACC_OPCODES:
            return self.parseMath()
        else:
            raise ValueError("Invalid opcode %s which is %s in hex and %s in binary at position %d" % 
                             (self.asm_bytes[self.pc], 
                              hex(self.asm_bytes[self.pc]), 
                              bin(self.asm_bytes[self.pc]), 
                              self.pc))
    
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
    
    def parseDirectMemory(self, mod, reg, rm, word, lam, mnemonic=None, force_zero=False):
        if mnemonic is None:
            mnemonic = ""
        elif not mnemonic.endswith(" "):
            mnemonic += " "
            
        if mod == 0b11:
            src, dest = getRegisterName(reg, word), getRegisterName(rm, word)
        else: 
            displacement = self.getIntegerFromBytes(mod, signed = True)
            if mod == 0b00 and rm == 0b110:
                # pull two bytes from the instruction
                direct_memory_value = self.getIntegerFromBytes(2)
                dest = [direct_memory_value]
            else:
                print(mnemonic, getRegisterNameMemory(rm, displacement, force_zero=force_zero))
                print(rm, displacement, force_zero)
                dest = mnemonic + getRegisterNameMemory(rm, displacement, force_zero=force_zero)
            src = lam()
        return src, dest
    
    def parseMath(self) -> str:
        instruction = self.getBytes(1)
        print("Instruction: ", instruction, "is", hex(instruction), "in hex")
        if instruction in MATH_ACC_OPCODES:
            mnemonic, reg = MATH_ACC_OPCODES[instruction]
            imm = self.getIntegerFromBytes(2 if reg == "ax" else 1)
            return "%s %s, %s" % (mnemonic, reg, imm)
        
        d = False
        mod, reg, rm = self.getModRM()
        if instruction in MATH_REG_OPCODES:
            mnemonic, word, d = MATH_REG_OPCODES[instruction]
            if mod == 0b11:
                src = getRegisterName(reg, word)
                dest = getRegisterName(rm, word)
            else:
                print(mod, reg, rm, instruction)
                src, dest = self.parseDirectMemory(mod, reg, rm, word, lambda: getRegisterName(reg, word), force_zero = True)
        elif instruction == 0x83:
            word= MATH_IMM_RM_OPCODES[instruction]["word"]
            mnemonic = OP83_TABLE.get(reg, f"OP{reg}")
            if mod == 3:
                dest=getRegisterName(rm, True)
                src = self.getIntegerFromBytes(1)
            else:
                
                displacement = self.getIntegerFromBytes(mod, signed = True)
                if mod == 0b00 and rm == 0b110:
                    # pull two bytes from the instruction
                    direct_memory_value = self.getIntegerFromBytes(2)
                    dest = [direct_memory_value]
                else:
                    dest = mnemonic + getRegisterNameMemory(rm, displacement, force_zero=True)
                src = self.getIntegerFromBytes(1, signed=True)
        elif instruction in MATH_IMM_RM_OPCODES:
            mnemonic = MATH_IMM_RM_OPCODES[instruction]["commands"][reg]
            num_bytes = MATH_IMM_RM_OPCODES[instruction]["bytes"]
            word = MATH_IMM_RM_OPCODES[instruction]["word"]

            dest = word + " " + getRegisterNameMemory(rm)
            imm = self.getIntegerFromBytes(num_bytes, signed = True)
            src = imm
        if d: 
            src, dest = dest, src
        return "%s %s, %s" % (mnemonic, dest, src)

    def parseMOVValue(self) -> str:
        instruction = self.getBytes(1)
        d = (instruction >> 1) & 0x1
        w = instruction & 0x1
        num_bytes = 2 if w else 1
        # if instruction starts with 0xA0 in the first six bits, then the instruction is MOV but we just pull direct memory address
        if instruction & 0b11111100 == 0xA0:
            dest = "ax" if w else "al"
            src = "[" + str(self.getIntegerFromBytes(num_bytes, signed = False)) + "]"
            # Print bytes of src
            if d: 
                src, dest = dest, src
            return "mov %s, %s" % (dest, src)

    def parseMOV(self) -> str:
        # the first 6 bytes are the opcode
        # the next bit is the d bit
        # the next bit is the w bit
        instruction = self.getBytes(1)
        d = (instruction >> 1) & 0x1
        w = instruction & 0x1
        bytes = 2 if w else 1

        mod, reg, rm = self.getModRM()

        src, dest = self.parseDirectMemory(mod, reg, rm, w, lambda: getRegisterName(reg, w))
        
        # If the instruction is 0xc6, then the instruction is a MOV but with fixed inputs
        if instruction & 0xF6 == 0xc6:
            d = 0
            mnemonic = "word " if w else "byte "
            direct_memory_address = self.getIntegerFromBytes(bytes)
            # unset d bit
            src = mnemonic + str(direct_memory_address)

        # Flip order if d is set
        if d == 0b1:
            src, dest = dest, src
        return "mov %s, %s" % (dest, src)
    
    def parseMOVDirect(self):
        # the first 4 bytes are the opcode
        instruction = self.getBytes(1)
        w = (instruction >> 3) & 0x1
        reg = instruction & 0x7
        immediate = self.getIntegerFromBytes(w+1)
        return "mov %s, %s" % (getRegisterName(reg, w), immediate)

       
def decode8086(asm_bytes) -> list[str]:
    # The first step is to parse the six bits of the opcode
    # if the first six bits of the opcode are 100010, then the opcode is MOV
    # if the first six bits of the opcode are 100010:
    parser = ASMParser(asm_bytes)
    return parser.parse()
    

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
    inputs = readAssembly(args.input)
    comparison=inputs

    with open(args.output, 'w') as f:
        f.write("bits %d\n" % args.bits)
        for line in decode8086(inputs):
            # print(line)
            f.write(line)
        
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
