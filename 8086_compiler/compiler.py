
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

class Store:
    def update(self, bits):
        self.chunk_size = bits // 8

    @property
    def chunk_size(self):
        return self._chunk_size
    
    @chunk_size.setter
    def chunk_size(self, value):
        self._chunk_size = value

STORE = Store()

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

def decodeMOV(line) -> str:
    d = line >> 9 & 0b1
    w = line >> 8 & 0b1
    mod = line >> 6 & 0b11
    reg = line >> 3 & 0b111
    rm = line & 0b111
    print("d: ", d)
    print("w: ", w)
    print("mod: ", mod)
    print("reg: ", reg)
    print("rm: ", rm)

    reg_name = getRegisterName(reg, w)
    rm_name = getRegisterName(rm, w)

    if d:
        return f"MOV {reg_name}, {rm_name}"
    else:
        return f"MOV {rm_name}, {reg_name}"


def decode8086(asm_bytes) -> str:
    # The first step is to parse the six bits of the opcode
    # if the first six bits of the opcode are 100010, then the opcode is MOV
    # if the first six bits of the opcode are 100010:

    # Print the bits of line
    debugBytesNum(asm_bytes)
    mask = 0b111111 << 10
    debugBytesNum(mask)
    opcode = asm_bytes & mask
    debugBytesNum(opcode)
    opcode = opcode >> 10
    debugBytesNum(opcode)
    if opcode == 0b100010:
        print("MOV")
        return decodeMOV(asm_bytes)
    
    raise ValueError("Invalid opcode %s" % opcode)
    

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

def parseAssembly(filename):
    with open(filename, 'rb') as f:
        raw = f.read()
        output = [int.from_bytes(raw[i:i+STORE.chunk_size], byteorder='big') 
                for i in range(0, len(raw), STORE.chunk_size)]
    return output

def main():
    args = parse_args()
    # The input is a bit file with 8086 assembly code
    # The output is a disassembled file with 8086 assembly code in strings
    print("Input: ", args.input)
    print("Output: ", args.output)
    print("Bits: ", args.bits)

    STORE.update(args.bits)
    inputs = parseAssembly(args.input)
    print("Inputs: ", inputs)
    
    comparison=inputs

    with open(args.output, 'w') as f:
        f.write("bits %d\n" % args.bits)
        for asm_line in inputs:
            f.write(decode8086(asm_line) + "\n")
        
    # Call nasm to assemble the output
    test_file = args.output.replace(".bin", ".o")
    subprocess.check_output(["nasm", "-f", "bin", "-o", test_file,  args.output]).decode("utf-8")
    output = parseAssembly(test_file)

    if output == comparison:
        print("Success")
    else:
        print("Failure")
        print("Expected: ", comparison)
        print("Received: ", output)

if __name__ == "__main__":
    main()
