import sys, struct
with open(sys.argv[1], "rb") as f:
    content = f.read()
while len(content) % 4 != 0:
    content += b'\x00'
for i in range(0, len(content), 4):
    instr = struct.unpack("<I", content[i:i+4])[0]
    print(f"0x{instr:08x}")
