#!/usr/bin/env python3
"""
RISC-V RV32I 指令解码器
将十六进制机器码解码为汇编指令
"""

def decode_instruction(hex_str):
    """解码单条 RISC-V RV32I 指令"""
    value = int(hex_str, 16)
    
    opcode = value & 0x7F
    rd = (value >> 7) & 0x1F
    funct3 = (value >> 12) & 0x7
    rs1 = (value >> 15) & 0x1F
    rs2 = (value >> 20) & 0x1F
    funct7 = (value >> 25) & 0x7F
    
    # 寄存器名称映射
    reg_names = {
        0: 'x0', 1: 'ra', 2: 'sp', 3: 'gp', 4: 'tp',
        5: 't0', 6: 't1', 7: 't2',
        8: 's0', 9: 's1',
        10: 'a0', 11: 'a1', 12: 'a2', 13: 'a3', 14: 'a4', 15: 'a5',
        16: 'a6', 17: 'a7',
        18: 's2', 19: 's3', 20: 's4', 21: 's5', 22: 's6', 23: 's7',
        24: 's8', 25: 's9', 26: 's10', 27: 's11',
        28: 't3', 29: 't4', 30: 't5', 31: 't6'
    }
    
    def get_reg(reg):
        return reg_names.get(reg, f'x{reg}')
    
    def sign_extend_12(imm):
        if imm & 0x800:
            return imm - 0x1000
        return imm
    
    def sign_extend_20(imm):
        if imm & 0x80000:
            return imm - 0x100000
        return imm
    
    # 根据操作码解码
    if opcode == 0x37:  # LUI (U-type)
        imm = (value >> 12) & 0xFFFFF
        return f"lui {get_reg(rd)}, 0x{imm:X}"
    
    elif opcode == 0x17:  # AUIPC (U-type)
        imm = (value >> 12) & 0xFFFFF
        return f"auipc {get_reg(rd)}, 0x{imm:X}"
    
    elif opcode == 0x6F:  # JAL (J-type)
        imm_20 = (value >> 31) & 0x1
        imm_10_1 = (value >> 21) & 0x3FF
        imm_11 = (value >> 20) & 0x1
        imm_19_12 = (value >> 12) & 0xFF
        imm = (imm_20 << 20) | (imm_19_12 << 12) | (imm_11 << 11) | (imm_10_1 << 1)
        if imm & 0x100000:
            imm = imm - 0x200000
        return f"jal {get_reg(rd)}, {imm}"
    
    elif opcode == 0x67:  # JALR (I-type)
        imm = (value >> 20) & 0xFFF
        imm = sign_extend_12(imm)
        if rd == 0 and rs1 == 1 and imm == 0:
            return "ret"
        return f"jalr {get_reg(rd)}, {imm}({get_reg(rs1)})"
    
    elif opcode == 0x63:  # 分支指令 (B-type)
        imm_12 = (value >> 31) & 0x1
        imm_10_5 = (value >> 25) & 0x3F
        imm_4_1 = (value >> 8) & 0xF
        imm_11 = (value >> 7) & 0x1
        imm = (imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1)
        if imm & 0x1000:
            imm = imm - 0x2000
        
        branch_types = {
            0: ('beq', '=='),
            1: ('bne', '!='),
            4: ('blt', '<'),
            5: ('bge', '>='),
            6: ('bltu', '<u'),
            7: ('bgeu', '>=u')
        }
        if funct3 in branch_types:
            mnemonic, _ = branch_types[funct3]
            return f"{mnemonic} {get_reg(rs1)}, {get_reg(rs2)}, {imm}"
        return f"unknown_branch 0x{hex_str}"
    
    elif opcode == 0x03:  # 加载指令 (I-type)
        imm = (value >> 20) & 0xFFF
        imm = sign_extend_12(imm)
        load_types = {
            0: ('lb', 1),
            1: ('lh', 2),
            2: ('lw', 4),
            4: ('lbu', 1),
            5: ('lhu', 2)
        }
        if funct3 in load_types:
            mnemonic, _ = load_types[funct3]
            return f"{mnemonic} {get_reg(rd)}, {imm}({get_reg(rs1)})"
        return f"unknown_load 0x{hex_str}"
    
    elif opcode == 0x23:  # 存储指令 (S-type)
        imm_11_5 = (value >> 25) & 0x7F
        imm_4_0 = (value >> 7) & 0x1F
        imm = (imm_11_5 << 5) | imm_4_0
        imm = sign_extend_12(imm)
        store_types = {
            0: ('sb', 1),
            1: ('sh', 2),
            2: ('sw', 4)
        }
        if funct3 in store_types:
            mnemonic, _ = store_types[funct3]
            return f"{mnemonic} {get_reg(rs2)}, {imm}({get_reg(rs1)})"
        return f"unknown_store 0x{hex_str}"
    
    elif opcode == 0x13:  # 立即数算术指令 (I-type)
        imm = (value >> 20) & 0xFFF
        imm = sign_extend_12(imm)
        
        if funct3 == 0:  # ADDI
            if rd == 0 and imm == 0:
                return f"nop"
            if rs1 == 0:
                return f"li {get_reg(rd)}, {imm}"
            if imm == 0:
                return f"mv {get_reg(rd)}, {get_reg(rs1)}"
            return f"addi {get_reg(rd)}, {get_reg(rs1)}, {imm}"
        elif funct3 == 1:  # SLLI
            shamt = imm & 0x1F
            return f"slli {get_reg(rd)}, {get_reg(rs1)}, {shamt}"
        elif funct3 == 2:  # SLTI
            return f"slti {get_reg(rd)}, {get_reg(rs1)}, {imm}"
        elif funct3 == 3:  # SLTIU
            return f"sltiu {get_reg(rd)}, {get_reg(rs1)}, {imm}"
        elif funct3 == 4:  # XORI
            return f"xori {get_reg(rd)}, {get_reg(rs1)}, {imm}"
        elif funct3 == 5:
            shamt = imm & 0x1F
            if funct7 == 0:  # SRLI
                return f"srli {get_reg(rd)}, {get_reg(rs1)}, {shamt}"
            elif funct7 == 0x20:  # SRAI
                return f"srai {get_reg(rd)}, {get_reg(rs1)}, {shamt}"
        elif funct3 == 6:  # ORI
            return f"ori {get_reg(rd)}, {get_reg(rs1)}, {imm}"
        elif funct3 == 7:  # ANDI
            return f"andi {get_reg(rd)}, {get_reg(rs1)}, {imm}"
        return f"unknown_imm_arith 0x{hex_str}"
    
    elif opcode == 0x33:  # 寄存器算术指令 (R-type)
        if funct3 == 0:
            if funct7 == 0:  # ADD
                return f"add {get_reg(rd)}, {get_reg(rs1)}, {get_reg(rs2)}"
            elif funct7 == 0x20:  # SUB
                return f"sub {get_reg(rd)}, {get_reg(rs1)}, {get_reg(rs2)}"
        elif funct3 == 1 and funct7 == 0:  # SLL
            return f"sll {get_reg(rd)}, {get_reg(rs1)}, {get_reg(rs2)}"
        elif funct3 == 2 and funct7 == 0:  # SLT
            return f"slt {get_reg(rd)}, {get_reg(rs1)}, {get_reg(rs2)}"
        elif funct3 == 3 and funct7 == 0:  # SLTU
            return f"sltu {get_reg(rd)}, {get_reg(rs1)}, {get_reg(rs2)}"
        elif funct3 == 4 and funct7 == 0:  # XOR
            return f"xor {get_reg(rd)}, {get_reg(rs1)}, {get_reg(rs2)}"
        elif funct3 == 5:
            if funct7 == 0:  # SRL
                return f"srl {get_reg(rd)}, {get_reg(rs1)}, {get_reg(rs2)}"
            elif funct7 == 0x20:  # SRA
                return f"sra {get_reg(rd)}, {get_reg(rs1)}, {get_reg(rs2)}"
        elif funct3 == 6 and funct7 == 0:  # OR
            return f"or {get_reg(rd)}, {get_reg(rs1)}, {get_reg(rs2)}"
        elif funct3 == 7 and funct7 == 0:  # AND
            return f"and {get_reg(rd)}, {get_reg(rs1)}, {get_reg(rs2)}"
        return f"unknown_reg_arith 0x{hex_str}"
    
    elif opcode == 0x0F:  # FENCE
        return f"fence"
    
    elif opcode == 0x73:  # 系统指令
        if funct3 == 0:
            if funct7 == 0:  # ECALL
                return "ecall"
            elif funct7 == 0x20:  # EBREAK
                return "ebreak"
        return f"unknown_system 0x{hex_str}"
    
    return f"unknown 0x{hex_str}"


def main():
    """主函数：读取 test.out 并解码所有指令"""
    with open('test.out', 'r') as f:
        lines = f.readlines()
    
    print("RISC-V RV32I 指令解码结果")
    print("=" * 70)
    print(f"{'行号':<6} {'地址':<10} {'机器码':<12} {'汇编指令'}")
    print("-" * 70)
    
    line_num = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        line_num += 1
        
        # 解析行格式: "0x机器码"
        hex_code = line.strip()
        
        # 移除 0x 前缀（如果有）
        if hex_code.startswith('0x'):
            hex_code = hex_code[2:]
        
        # 解码指令
        instruction = decode_instruction(hex_code)
        
        # 计算地址（假设每条指令4字节，行号从1开始）
        addr = (line_num - 1) * 4
        
        print(f"{line_num:<6} {addr:#06x}     0x{hex_code:<8} {instruction}")


if __name__ == '__main__':
    main()
