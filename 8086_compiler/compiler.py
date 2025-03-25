
# Our goal is to write something that parses valid 8086 assembly code and outputs the corresponding machine code.
# We will use the following instructions:
# MOV, ADD, SUB, MUL, DIV, AND, OR, XOR, CMP, JMP, JE, JNE, JG, JGE, JL, JLE, CALL, RET, INT, NOP
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
import subprocess

def debugBytesStr(line):
    bitstr = ''.join(f'{byte:08b}' for byte in line)
    print(bitstr)

def debugBytesNum(i):
    print(f'{i:08b}')

def getRegisterName(register, w):
    reg_table_16 = ['ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di']
    reg_table_8  = ['al', 'cl', 'dl', 'bl', 'ah', 'ch', 'dh', 'bh']
    regs = reg_table_8 if w == 0 else reg_table_16
    return regs[register]

def getRegisterNameMemory(register, displacement = None):
    if displacement is None: 
        displacement = 0
    reg_table = ['bx + si', 'bx + di', 'bp + si', 'bp + di', 'si', 'di', 'bp', 'bx']
    reg = reg_table[register]
    if displacement > 0:
        reg += f" + {displacement}"
    elif displacement < 0:
        reg += f" - {-displacement}"
    return f"[{reg}]"

class ASMParser:
    def __init__(self, asm_bytes):
        self.asm_bytes = asm_bytes
        self.pc = 0
    
    def parse(self):
        ret = []
        while self.pc < len(self.asm_bytes):
            line = self.parseInstruction()
            ret += line
            # print(line)
        return [line + "\n" for line in ret]
    
    def parseInstruction(self):
        # opcode is the first byte sort of
        if self.asm_bytes[self.pc] >> 2 == 0b100010:
            return self.parseMOV()
        elif self.asm_bytes[self.pc] >> 1 == 0b1100011:
            return self.parseMOV()
        elif self.asm_bytes[self.pc] >> 2 == 0b101000:
            return self.parseMOV()
        elif self.asm_bytes[self.pc] >> 4 == 0b1011:
            print ("MOV Direct")
            return self.parseMOVDirect()
        else:
            raise ValueError("Invalid opcode %s which is %s in hex and %s in binary at position %d" % (self.asm_bytes[self.pc], hex(self.asm_bytes[self.pc]), bin(self.asm_bytes[self.pc]), self.pc))

    def pullBytes(self, num_bytes, signed = False):
        ret = 0
        for i in range(num_bytes-1, -1, -1):
            ret = (ret << 8) | self.asm_bytes[self.pc + i]
        self.pc += num_bytes
        return ret
    
    def getIntegerFromBytes(self, num_bytes, signed = True):
        if num_bytes == 0:
            return 0
        ret = self.pullBytes(num_bytes)
        if signed:
            sign_bit = 1 << (num_bytes * 8 - 1)
            ret = (ret ^ sign_bit) - sign_bit
        return ret

    def parseModRM(self):
        modrm = self.pullBytes(1)
        mod = modrm >> 6
        reg = (modrm >> 3) & 0x7
        rm = modrm & 0x7
        return mod, reg, rm
    

    def parseMOV(self):
        # the first 6 bytes are the opcode
        # the next bit is the d bit
        # the next bit is the w bit

        displacement = 0
        instruction = self.pullBytes(1)
        d = (instruction >> 1) & 0x1
        w = instruction & 0x1
        bytes = 2 if w else 1
        # if instruction starts with 0xA0 in the first six bits, then the instruction is MOV but we just pull direct memory address
        if instruction & 0b11111100 == 0xA0:
            dest = "ax" if w else "al"
            src = "[" + str(self.getIntegerFromBytes(bytes, signed = False)) + "]"
            # Print bytes of src
            if d: 
                src, dest = dest, src
            return ["mov %s, %s" % (dest, src)]

        mod, reg, rm = self.parseModRM()
        if mod == 0b11:
            src, dest = getRegisterName(rm, w), " " + getRegisterName(reg, w)
        else: 
            displacement = self.getIntegerFromBytes(mod, signed = True)

            if mod == 0b00 and rm == 0b110:
                # pull two bytes from the instruction
                direct_memory_address = self.getIntegerFromBytes(2)
                src = [direct_memory_address]
            else:
                src = getRegisterNameMemory(rm, displacement)
            dest = getRegisterName(reg, w)
        
        # If the instruction is 0xc6, then the instruction is a MOV but with fixed inputs
        if instruction & 0xc6 == 0xc6:
            text = "word " if w else "byte "
            direct_memory_address = self.getIntegerFromBytes(bytes)
            # unset d bit
            d = 0
            dest = text + str(direct_memory_address)

        # Flip order if d is set
        if d == 0b1:
            src, dest = dest, src
        return ["mov %s, %s" % (src, dest)]
    
    def parseMOVDirect(self):
        # the first 4 bytes are the opcode
        instruction = self.pullBytes(1)
        w = (instruction >> 3) & 0x1
        reg = instruction & 0x7
        if w:
            immediate = self.getIntegerFromBytes(2)
        else:
            immediate = self.getIntegerFromBytes(1)
        
        return ["mov %s, %s" % (getRegisterName(reg, w), immediate)]

       
def decode8086(asm_bytes) -> list[str]:
    # The first step is to parse the six bits of the opcode
    # if the first six bits of the opcode are 100010, then the opcode is MOV
    # if the first six bits of the opcode are 100010:
    parser = ASMParser(asm_bytes)
    return parser.parse()
    

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Compile 8086 assembly')
    parser.add_argument('input', type=str, help='Input file')
    parser.add_argument('--output', type=str, required=False, default=None, help='Output file')
    parser.add_argument('--bits', type=int, default=16, help='Number of bits')

    args = parser.parse_args()
    if args.output is None:
        args.output = args.input + ".bin"
    return args

def readAssembly(filename):
    with open(filename, 'rb') as f:
        raw = f.read()
        return raw

def main():
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
        lines = decode8086(inputs)
        for line in lines:
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
