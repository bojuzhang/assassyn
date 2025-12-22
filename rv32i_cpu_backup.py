#!/usr/bin/env python3
"""
五级流水线RV32I CPU实现
使用Assassyn语言实现完整的RISC-V 32位基础指令集处理器
"""

from assassyn.frontend import *
from assassyn.backend import elaborate
from assassyn import utils
from assassyn.ir.memory.sram import SRAM
from assassyn.ir.module import downstream, Downstream

# ==================== 常量定义 ===================
XLEN = 32  # RISC-V XLEN
REG_COUNT = 32  # 通用寄存器数量


PIPELINE_REGS = Record(
    # IF/ID阶段寄存器
    if_id_pc=UInt(XLEN),
    if_id_instruction=UInt(XLEN),
    if_id_valid=UInt(1),
    
    # ID/EX阶段寄存器
    id_ex_pc=UInt(XLEN),
    id_ex_control=UInt(32),       # 控制信号
    id_ex_valid=UInt(1),
    id_ex_rs1_idx=UInt(5),        # rs1寄存器索引
    id_ex_rs2_idx=UInt(5),        # rs2寄存器索引
    id_ex_immediate=UInt(XLEN),
    
    # EX/MEM阶段寄存器
    ex_mem_pc=UInt(XLEN),
    ex_mem_control=UInt(32),        # 控制信号
    ex_mem_valid=UInt(1),
    ex_mem_result=UInt(XLEN),
    ex_mem_data=UInt(XLEN),
    
    # MEM/WB阶段寄存器
    mem_wb_control=UInt(32),        # 控制信号
    mem_wb_valid=UInt(1),
    mem_wb_mem_data=UInt(XLEN),     # 内存读取的数据
    mem_wb_ex_result=UInt(XLEN)     # EX阶段的结果
)

# ==================== 寄存器文件 ===================
# 32个通用寄存器，x0硬线为0
# 使用RegArray定义寄存器数组，而不是模块

# ==================== ALU ===================
class ALU(Downstream):
    """算术逻辑单元"""
    def __init__(self):
        super().__init__()
    
    @downstream.combinational
    def build(self, op: Value, a: Value, b: Value):
        
        # 默认结果
        result = UInt(XLEN)(0)
        zero = UInt(1)(0)
        
        # 根据操作码执行不同操作
        with Condition(op == UInt(5)(0b00000)):  # ADD
            result = a + b
        
        with Condition(op == UInt(5)(0b00001)):  # SUB
            result = a - b
        
        with Condition(op == UInt(5)(0b00010)):  # SLL
            result = a << (b & UInt(XLEN)(0x1F))
        
        with Condition(op == UInt(5)(0b00011)):  # SLT
            # 有符号比较：如果a < b（有符号），则结果为1
            a_signed = SInt(XLEN)(a)
            b_signed = SInt(XLEN)(b)
            result = UInt(XLEN)(0)
            with Condition(a_signed < b_signed):
                result = UInt(XLEN)(1)
        
        with Condition(op == UInt(5)(0b00100)):  # XOR
            result = a ^ b
        
        with Condition(op == UInt(5)(0b00101)):  # SRL
            # 逻辑右移：高位补0
            shift_amount = b & UInt(XLEN)(0x1F)
            result = a >> shift_amount
        
        with Condition(op == UInt(5)(0b00110)):  # SRA
            # 算术右移：保持符号位
            shift_amount = b & UInt(XLEN)(0x1F)
            a_signed = SInt(XLEN)(a)
            result = UInt(XLEN)(a_signed >> shift_amount)
        
        with Condition(op == UInt(5)(0b00111)):  # SLTU
            # 无符号比较：如果a < b（无符号），则结果为1
            result = UInt(XLEN)(0)
            with Condition(a < b):
                result = UInt(XLEN)(1)
        
        with Condition(op == UInt(5)(0b01000)):  # OR
            result = a | b
        
        with Condition(op == UInt(5)(0b01001)):  # AND
            result = a & b
        
        log("ALU: OP={:05b}, A={:08x}, B={:08x}, Result={:08x}",
            op, a, b, result)
        
        return result

# ==================== 分支单元 ===================
class BranchUnit(Downstream):
    """分支比较单元"""
    def __init__(self):
        super().__init__()
    
    @downstream.combinational
    def build(self, op: Value, a: Value, b: Value):
        
        taken = UInt(1)(0)
        
        # 执行分支比较
        with Condition(op == UInt(3)(0b001)):  # BEQ (修改后的值)
            with Condition(a == b):
                taken = UInt(1)(1)
        
        with Condition(op == UInt(3)(0b010)):  # BNE (修改后的值)
            with Condition(a != b):
                taken = UInt(1)(1)
        
        with Condition(op == UInt(3)(0b011)):  # BLT (修改后的值)
            # 有符号比较：如果a < b（有符号），则分支成功
            a_signed = SInt(XLEN)(a)
            b_signed = SInt(XLEN)(b)
            with Condition(a_signed < b_signed):
                taken = UInt(1)(1)
        
        with Condition(op == UInt(3)(0b100)):  # BGE (修改后的值)
            # 有符号比较：如果a >= b（有符号），则分支成功
            a_signed = SInt(XLEN)(a)
            b_signed = SInt(XLEN)(b)
            with Condition(a_signed >= b_signed):
                taken = UInt(1)(1)
        
        with Condition(op == UInt(3)(0b101)):  # BLTU (修改后的值)
            # 无符号比较：如果a < b（无符号），则分支成功
            with Condition(a < b):
                taken = UInt(1)(1)
        
        with Condition(op == UInt(3)(0b110)):  # BGEU (修改后的值)
            # 无符号比较：如果a >= b（无符号），则分支成功
            with Condition(a >= b):
                taken = UInt(1)(1)
        
        log("BRANCH: OP={:03b}, A={:08x}, B={:08x}, Taken={}",
            op, a, b, taken)
        
        return taken

# ==================== IF阶段：指令获取 ===================
class FetchStage(Module):
    """指令获取阶段(IF)"""
    def __init__(self):
        super().__init__(ports={
            'pc_in': Port(UInt(XLEN)),      # 输入PC
            'stall': Port(UInt(1)),        # 暂停信号
        })
    
    @module.combinational
    def build(self, pipeline_regs, imem, decode_stage):
        pc_in, stall = self.pop_all_ports(True)

        with Condition(~stall):
        
            # 计算下一个PC
            next_pc = pc_in + UInt(XLEN)(4)
            
            # 直接使用输入的pc_in
            current_pc = pc_in
            
            # 指令内存访问 - 直接从寄存器数组读取
            # 字对齐地址 - 右移2位（除以4）得到字地址
            word_addr = current_pc >> UInt(XLEN)(2)
            instruction = imem[word_addr]
            
            # 计算有效信号
            valid_signal = UInt(1)(1)
            
            pipeline_regs[0].if_id_pc = current_pc
            pipeline_regs[0].if_id_instruction = instruction
            pipeline_regs[0].if_id_valid = valid_signal
            
            log("IF: PC={:08x}, Instruction={:08x}, Stall={}", current_pc, instruction, stall)
            
        # ID阶段 - 使用valid_in控制执行
        # 当stall为真时，将valid_in设为0来阻断运行
        # 注意：ID阶段应该在stall时停止，以防止新指令进入EX阶段
        id_valid_in = pipeline_regs[0].if_id_valid & ~stall
        decode_stage.async_called(
            instruction_in=pipeline_regs[0].if_id_instruction,
            valid_in=id_valid_in,
            if_id_pc_in=pipeline_regs[0].if_id_pc
        )

# ==================== ID阶段：指令解码 ===================
class DecodeStage(Module):
    """指令解码阶段(ID)"""
    def __init__(self):
        super().__init__(ports={
            'instruction_in': Port(UInt(XLEN)),  # 输入指令
            'valid_in': Port(UInt(1)),       # 输入有效信号
            'if_id_pc_in': Port(UInt(XLEN)),  # IF/ID PC输入
        })
    
    @module.combinational
    def build(self, pipeline_regs, reg_file, execute_stage):
        instruction, valid_in, if_id_pc_in = self.pop_all_ports(True)
        
        # 如果指令无效，直接返回，不更新ID/EX寄存器
        with Condition(~valid_in):
            log("ID: Invalid instruction, skipping decode")
            return
        
        # 解析指令字段
        opcode = instruction[6:0]           # bits 6:0
        rd = instruction[11:7]             # bits 11:7
        func3 = instruction[14:12]          # bits 14:12
        rs1 = instruction[19:15]           # bits 19:15
        rs2 = instruction[24:20]           # bits 24:20
        funct7 = instruction[31:25]         # bits 31:25
        
        # 从寄存器数组读取操作数
        rs1_data = reg_file[rs1]
        rs2_data = reg_file[rs2]
        
        # 提取立即数
        immediate_i = sext(instruction[31:20])     # I型立即数
        immediate_s = sext(concat(instruction[31:25], instruction[11:7]))  # S型立即数
        immediate_b = sext(concat(instruction[31], instruction[7], instruction[30:25], instruction[11:8], UInt(1)(0)))  # B型立即数
        immediate_u = instruction[31:12] << UInt(XLEN)(12)  # U型立即数
        immediate_j = sext(concat(instruction[31], instruction[19:12], instruction[20], instruction[30:21], UInt(1)(0)))  # J型立即数
        
        # 控制信号解码
        alu_op = UInt(5)(0)
        mem_read = UInt(1)(0)
        mem_write = UInt(1)(0)
        reg_write = UInt(1)(0)
        mem_to_reg = UInt(1)(0)
        alu_src = UInt(2)(0)  # 00:寄存器, 01:立即数, 10:PC
        branch_op = UInt(3)(0)
        jump_op = UInt(1)(0)  # 跳转指令标志
        immediate = UInt(XLEN)(0)  # 初始化立即数
        
        # 根据opcode设置控制信号
        with Condition(opcode == UInt(7)(0b0110011)):  # R型指令
            with Condition(funct7[5] == UInt(1)(1)):  # SUB or SRA
                with Condition(func3 == UInt(3)(0b000)):  # SUB
                    alu_op = UInt(5)(0b00001)
                with Condition(func3 == UInt(3)(0b101)):  # SRA
                    alu_op = UInt(5)(0b00110)
            # ADD, AND, OR, XOR, SLT, SLTU, SLL, SRL
            with Condition(funct7[5] == UInt(1)(0)):  # funct7[5] == 0
                with Condition(func3 == UInt(3)(0b000)):  # ADD
                    alu_op = UInt(5)(0b00000)
                with Condition(func3 == UInt(3)(0b111)):  # AND
                    alu_op = UInt(5)(0b01001)
                with Condition(func3 == UInt(3)(0b110)):  # OR
                    alu_op = UInt(5)(0b01000)
                with Condition(func3 == UInt(3)(0b100)):  # XOR
                    alu_op = UInt(5)(0b00100)
                with Condition(func3 == UInt(3)(0b010)):  # SLT
                    alu_op = UInt(5)(0b00011)
                with Condition(func3 == UInt(3)(0b011)):  # SLTU
                    alu_op = UInt(5)(0b00111)
                with Condition(func3 == UInt(3)(0b001)):  # SLL
                    alu_op = UInt(5)(0b00010)
                with Condition(func3 == UInt(3)(0b101)):  # SRL
                    alu_op = UInt(5)(0b00101)
            # 其他操作...
            reg_write = UInt(1)(1) & (rd != UInt(5)(0))  # x0寄存器不会写入
            alu_src = UInt(2)(0)
            
        with Condition(opcode == UInt(7)(0b0010011)):  # I型指令
            with Condition(func3 == UInt(3)(0b000)):  # ADDI
                alu_op = UInt(5)(0b00000)
            with Condition(func3 == UInt(3)(0b111)):  # ANDI
                alu_op = UInt(5)(0b01001)
            with Condition(func3 == UInt(3)(0b110)):  # ORI
                alu_op = UInt(5)(0b01000)
            with Condition(func3 == UInt(3)(0b100)):  # XORI
                alu_op = UInt(5)(0b00100)
            with Condition(func3 == UInt(3)(0b010)):  # SLTI
                alu_op = UInt(5)(0b00011)
            with Condition(func3 == UInt(3)(0b011)):  # SLTIU
                alu_op = UInt(5)(0b00111)
            with Condition(func3 == UInt(3)(0b001)):  # SLLI
                alu_op = UInt(5)(0b00010)
            with Condition(func3 == UInt(3)(0b101)):  # SRLI or SRAI
                with Condition(funct7[5] == UInt(1)(1)):  # SRAI
                    alu_op = UInt(5)(0b00110)
                with Condition(funct7[5] == UInt(1)(0)):  # SRLI
                    alu_op = UInt(5)(0b00101)
            # 其他操作...
            reg_write = UInt(1)(1) & (rd != UInt(5)(0))  # x0寄存器不会写入
            alu_src = UInt(2)(1)
            immediate = immediate_i
            
        with Condition(opcode == UInt(7)(0b0000011)):
            # LW加载指令
            alu_op = UInt(5)(0b00000)  # ADD用于地址计算
            mem_read = UInt(1)(1)
            reg_write = UInt(1)(1) & (rd != UInt(5)(0))  # x0寄存器不会写入
            mem_to_reg = UInt(1)(1)
            alu_src = UInt(2)(1)
            immediate = immediate_i
            
        store_type_bits = UInt(2)(0)
        with Condition(opcode == UInt(7)(0b0100011)):  # 存储指令
            with Condition(func3 == UInt(3)(0b010)):  # SW (Store Word)
                alu_op = UInt(5)(0b00000)  # ADD用于地址计算
                mem_write = UInt(1)(1)     # 存储使能
                alu_src = UInt(2)(1)
                immediate = immediate_s
                store_type_bits = UInt(2)(0b10)
            with Condition(func3 == UInt(3)(0b000)):  # SB (Store Byte)
                alu_op = UInt(5)(0b00000)
                mem_write = UInt(1)(1)     # 存储使能
                alu_src = UInt(2)(1)
                immediate = immediate_s
                store_type_bits = UInt(2)(0b00)
            with Condition(func3 == UInt(3)(0b001)):  # SH (Store Halfword)
                alu_op = UInt(5)(0b00000)
                mem_write = UInt(1)(1)     # 存储使能
                alu_src = UInt(2)(1)
                immediate = immediate_s
                store_type_bits = UInt(2)(0b01)

        with Condition(opcode == UInt(7)(0b1100011)):  # 分支指令
            immediate = immediate_b
            with Condition(func3 == UInt(3)(0b000)):  # BEQ
                branch_op = UInt(3)(0b001)  # 修改为非0值，确保被识别为分支指令
            with Condition(func3 == UInt(3)(0b001)):  # BNE
                branch_op = UInt(3)(0b010)
            with Condition(func3 == UInt(3)(0b100)):  # BLT
                branch_op = UInt(3)(0b011)
            with Condition(func3 == UInt(3)(0b101)):  # BGE
                branch_op = UInt(3)(0b100)
            with Condition(func3 == UInt(3)(0b110)):  # BLTU
                branch_op = UInt(3)(0b101)
            with Condition(func3 == UInt(3)(0b111)):  # BGEU
                branch_op = UInt(3)(0b110)
            
        
        with Condition(opcode == UInt(7)(0b0110111)):  # LUI
            reg_write = UInt(1)(1) & (rd != UInt(5)(0))  # x0寄存器不会写入
            alu_src = UInt(2)(1)
            immediate = immediate_u
        
        with Condition(opcode == UInt(7)(0b0010111)):  # AUIPC
            reg_write = UInt(1)(1) & (rd != UInt(5)(0))  # x0寄存器不会写入
            alu_src = UInt(2)(2)
            immediate = immediate_u
        
        with Condition(opcode == UInt(7)(0b1101111)):  # JAL
            reg_write = UInt(1)(1) & (rd != UInt(5)(0))  # x0寄存器不会写入
            alu_src = UInt(2)(1)  # 使用立即数作为ALU输入
            immediate = immediate_j
            jump_op = UInt(1)(1)  # 设置跳转指令标志
            # JAL指令需要特殊处理，在EX阶段会计算返回地址(PC+4)
        
        control_signals = concat(
            alu_op,           # [4:0]   ALU操作码
            mem_read,         # [5]     内存读
            mem_write,        # [6]     内存写
            reg_write,        # [7]     寄存器写
            mem_to_reg,       # [8]     内存到寄存器
            UInt(1)(0),       # [9]     保留位
            alu_src,          # [10:9]  ALU输入选择
            UInt(6)(0),       # [16:11] 保留位
            branch_op,        # [19:17] 分支操作类型
            jump_op,          # [20]    跳转指令标志
            store_type_bits,  # [23:22] 存储类型: 00=SB, 01=SH, 10=SW
            UInt(1)(0),       # [21]    保留位
            rd,               # [29:25] rd地址
            immediate[11:0]   # [31:30] 立即数低12位
        )
        
        (pipeline_regs[0].id_ex_pc & self) <= if_id_pc_in
        (pipeline_regs[0].id_ex_control & self) <= control_signals   # 传递控制信号
        (pipeline_regs[0].id_ex_valid & self) <= valid_in
        (pipeline_regs[0].id_ex_rs1_idx & self) <= rs1                # 保存rs1索引
        (pipeline_regs[0].id_ex_rs2_idx & self) <= rs2                # 保存rs2索引
        (pipeline_regs[0].id_ex_immediate & self) <= immediate
        
        log("ID: PC={}, Opcode={:07x}, RD={}, RS1={}, RS2={}",
            if_id_pc_in, opcode, rd, rs1, rs2)

        # EX阶段 - 使用valid_in控制执行
        # 当stall为真时，将valid_in设为0来阻断运行
        # 注意：EX阶段应该在stall时停止，以防止新指令进入MEM阶段
        ex_valid_in = pipeline_regs[0].id_ex_valid & ~stall
        pc_change, target_pc, alu_result = execute_stage.async_called(
            pc_in=pipeline_regs[0].id_ex_pc,
            rs1_idx_in=pipeline_regs[0].id_ex_rs1_idx,
            rs2_idx_in=pipeline_regs[0].id_ex_rs2_idx,
            immediate_in=pipeline_regs[0].id_ex_immediate,
            control_in=pipeline_regs[0].id_ex_control,    # 控制信号
            valid_in=ex_valid_in
        )

# ==================== EX阶段：执行 ===================
class ExecuteStage(Module):
    """执行阶段(EX)"""
    def __init__(self):
        super().__init__(ports={
            'pc_in': Port(UInt(XLEN)),          # 输入PC
            'rs1_idx_in': Port(UInt(5)),       # 输入rs1索引
            'rs2_idx_in': Port(UInt(5)),       # 输入rs2索引
            'immediate_in': Port(UInt(XLEN)),   # 输入立即数
            'control_in': Port(UInt(32)),     # 输入控制信号
            'valid_in': Port(UInt(1)),        # 输入有效信号
        })
        
        self.alu = ALU()
        self.branch_unit = BranchUnit()
    
    @module.combinational
    def build(self, pipeline_regs, reg_file, memory_stage):
        pc_in, rs1_idx, rs2_idx, immediate_in, control_in, valid_in = self.pop_all_ports(True)
        
        # 直接从寄存器文件读取rs1和rs2的值
        rs1_data = reg_file[rs1_idx]
        rs2_data = reg_file[rs2_idx]
        
        # 如果指令无效，直接返回，不更新EX/MEM寄存器
        with Condition(~valid_in):
            log("EX: Invalid instruction, skipping execution")
            return UInt(1)(0), pc_in, pc_in  # 返回默认值：pc_change, target_pc, alu_result
        
        # 解析控制信号
        alu_op = control_in[4:0]
        mem_read = control_in[5]
        mem_write = control_in[6]
        reg_write = control_in[7]
        mem_to_reg = control_in[8]
        alu_src = control_in[10:9]
        branch_op = control_in[19:17]  # 修正：branch_op在[19:17]位
        jump_op = control_in[20]  # 跳转指令标志
        rd_addr = control_in[29:25]  # rd地址
        immediate = control_in[31:22]  # 立即数
        
        # ALU输入B选择
        alu_b = immediate_in
        with Condition(alu_src == UInt(2)(0)):  # 寄存器
            alu_b = rs2_data
        
        # 根据指令类型决定执行ALU操作还是分支操作
        alu_result = UInt(XLEN)(0)
        
        # 判断是否为分支指令 (branch_op != 0)
        is_branch = (branch_op != UInt(3)(0b000))
        
        # 对于AUIPC指令，ALU输入A应该是PC而不是rs1_data
        alu_a = rs1_data
        with Condition(alu_src == UInt(2)(2)):  # AUIPC指令
            alu_a = pc_in
        
        # 初始化PC变化控制信号
        pc_change = UInt(1)(0)
        target_pc = pc_in + UInt(XLEN)(4)  # 默认目标PC是PC+4
        
        with Condition(is_branch | jump_op):
            with Condition(is_branch):
                # 分支指令：执行分支比较
                branch_result = self.branch_unit.build(branch_op, rs1_data, rs2_data)
                # 如果分支成功，计算目标地址
                with Condition(branch_result):
                    target_pc = pc_in + immediate_in
                    pc_change = UInt(1)(1)  # 分支成功，PC需要改变
                alu_result = pc_in  # 使用pc_in作为默认值，不会被实际使用
                
            with Condition(jump_op):
                # JAL指令：计算返回地址(PC+4)和跳转目标
                alu_result = pc_in + UInt(XLEN)(4)  # 返回地址是PC+4
                target_pc = pc_in + immediate_in  # 跳转目标是PC+立即数
                pc_change = UInt(1)(1)  # 跳转指令总是成功，PC需要改变
        
        with Condition(~is_branch & ~jump_op):
            # 普通指令：执行ALU操作
            # 对于需要rs2的ALU操作，使用rs2_data
            alu_b_final = alu_b
            with Condition(alu_src == UInt(2)(0)):  # 寄存器操作
                alu_b_final = rs2_data
            with Condition(alu_src == UInt(2)(1)):  # 立即数操作
                alu_b_final = alu_b
            with Condition(alu_src == UInt(2)(2)):  # PC操作 (AUIPC)
                alu_b_final = alu_b
            alu_result = self.alu.build(alu_op, alu_a, alu_b_final)
        
        (pipeline_regs[0].ex_mem_pc & self) <= pc_in
        (pipeline_regs[0].ex_mem_control & self) <= control_in          # 传递控制信号
        (pipeline_regs[0].ex_mem_valid & self) <= valid_in
        (pipeline_regs[0].ex_mem_result & self) <= alu_result
        (pipeline_regs[0].ex_mem_data & self) <= rs2_data
        
        log("EX: PC={}, ALU_OP={:05b}, Result={:08x}, PC_Change={}, Target_PC={:08x}",
            pc_in, alu_op, alu_result, pc_change, target_pc)
        
        # MEM阶段 - 使用valid_in控制执行
        # 修改：MEM阶段应该继续执行，即使stall为真，否则无法完成已进入MEM阶段的指令
        mem_valid_in = pipeline_regs[0].ex_mem_valid  # 不再考虑stall信号
        self.memory.async_called(
            pc_in=pipeline_regs[0].ex_mem_pc,
            pipeline_regs=pipeline_regs,
            ex_mem_result=alu_result,
            addr_in=pipeline_regs[0].ex_mem_result,  # 直接使用ex_mem_result作为内存地址
            data_in=pipeline_regs[0].ex_mem_data,
            control_in=pipeline_regs[0].ex_mem_control,    # 控制信号
            valid_in=mem_valid_in
        )

        next_pc = pc_in + UInt(XLEN)(4)
        with Condition(pc_change):
            next_pc = target_pc
            # PC改变时需要将后续指令替换为NOP
            # IF/ID寄存器：替换为NOP指令
            pipeline_regs[0].if_id_instruction = UInt(XLEN)(0x00000013)  # NOP指令
            # ID/EX寄存器：替换为NOP指令的控制信号
            # NOP指令的控制信号：ADDI x0, x0, 0
            nop_control = concat(
                UInt(5)(0b00000),    # [4:0]   ALU操作码: ADD
                UInt(1)(0),          # [5]     内存读
                UInt(1)(0),          # [6]     内存写
                UInt(1)(0),          # [7]     寄存器写 (x0不需要写)
                UInt(1)(0),          # [8]     内存到寄存器
                UInt(1)(0),          # [9]     保留位
                UInt(2)(1),          # [10:9]  ALU输入选择: 立即数
                UInt(6)(0),          # [16:11] 保留位
                UInt(3)(0),          # [19:17] 分支操作类型
                UInt(1)(0),          # [20]    跳转指令标志
                UInt(2)(0),          # [23:22] 存储类型
                UInt(1)(0),          # [21]    保留位
                UInt(5)(0),          # [29:25] rd地址: x0
                UInt(12)(0)          # [31:20] 立即数低12位
            )
            pipeline_regs[0].id_ex_control = nop_control
            pipeline_regs[0].id_ex_immediate = UInt(XLEN)(0)  # 立即数为0
            pipeline_regs[0].id_ex_rs1_idx = UInt(5)(0)  # rs1为x0
            pipeline_regs[0].id_ex_rs2_idx = UInt(5)(0)  # rs2为x0
        
        # 注意：PC更新逻辑移到Driver模块中处理，因为ExecuteStage不应该直接访问PC
        # 这里只返回计算结果，由Driver模块决定是否更新PC
        
        log("EX: PC={:08x}, Next_PC={:08x}, PC_Change={}",
            pc_in, next_pc, pc_change)

# ==================== MEM阶段：内存访问 ===================
class MemoryStage(Module):
    """内存访问阶段(MEM)"""
    def __init__(self):
        super().__init__(ports={
            'pc_in': Port(UInt(XLEN)),          # 输入PC
            'addr_in': Port(UInt(XLEN)),        # 输入地址
            'data_in': Port(UInt(XLEN)),        # 输入数据
            'control_in': Port(UInt(32)),     # 输入控制信号
            'valid_in': Port(UInt(1))         # 输入有效信号
        })
        
        # 使用SRAM替代DataMemory，创建一个足够大的SRAM（1024个32位字）
        self.data_sram = SRAM(width=XLEN, depth=1024, init_file=None)
    
    @module.combinational
    def build(self, pipeline_regs, ex_mem_result, writeback_stage):
        pc_in, addr_in, data_in, control_in, valid_in = self.pop_all_ports(True)
        
        # 如果指令无效，直接返回，不更新MEM/WB寄存器
        with Condition(~valid_in):
            log("MEM: Invalid instruction, skipping memory access")
            return
        
        # 解析控制信号
        mem_read = control_in[5]
        mem_write = control_in[6]
        store_type = control_in[23:22]  # 存储类型: 00=SB, 01=SH, 10=SW
        
        # 默认输出
        mem_data = UInt(XLEN)(0)
        
        # 执行内存访问
        with Condition(mem_read | mem_write):
            # SRAM接口：we（写使能）, re（读使能）, addr（地址）, wdata（写数据）
            # 字对齐地址 - 右移2位（除以4）得到字地址
            word_addr = addr_in >> UInt(XLEN)(2)
            
            # 根据存储类型处理数据
            write_data = data_in
            with Condition(mem_write):
                with Condition(store_type == UInt(2)(0b00)):  # SB (Store Byte)
                    # 只保留低8位，其他位清零
                    write_data = data_in & UInt(XLEN)(0xFF)
                with Condition(store_type == UInt(2)(0b01)):  # SH (Store Halfword)
                    # 只保留低16位，其他位清零
                    write_data = data_in & UInt(XLEN)(0xFFFF)
                with Condition(store_type == UInt(2)(0b10)):  # SW (Store Word)
                    # 保留所有32位
                    write_data = data_in
            
            # 调用SRAM的build方法
            self.data_sram.build(we=mem_write, re=mem_read, addr=word_addr, wdata=write_data)
            
            # 读取数据从SRAM的dout寄存器
            mem_data = self.data_sram.dout[0]

        (pipeline_regs[0].mem_wb_control & self) <= control_in          # 传递控制信号
        (pipeline_regs[0].mem_wb_valid & self) <= valid_in
        (pipeline_regs[0].mem_wb_mem_data & self) <= mem_data          # 内存读取的数据
        (pipeline_regs[0].mem_wb_ex_result & self) <= ex_mem_result     # EX/MEM阶段的结果
        
        log("MEM: PC={}, Addr={:08x}, Read={}, Write={}",
            pc_in, addr_in, mem_read, mem_write)

        # WB阶段 - 使用valid_in控制执行
        # 修改：WB阶段应该继续执行，即使stall为真，否则无法完成已进入WB阶段的指令
        wb_valid_in = pipeline_regs[0].mem_wb_valid  # 不再考虑stall信号
        self.writeback.async_called(
            reg_file=reg_file,
            mem_data_in=pipeline_regs[0].mem_wb_mem_data,  # 内存读取的数据
            ex_result_in=pipeline_regs[0].mem_wb_ex_result, # EX阶段的结果
            control_in=pipeline_regs[0].mem_wb_control,    # 控制信号
            valid_in=wb_valid_in
        )

# ==================== WB阶段：写回 ===================
class WriteBackStage(Module):
    """写回阶段(WB)"""
    def __init__(self):
        super().__init__(ports={
            'mem_data_in': Port(UInt(XLEN)),    # 输入内存数据
            'ex_result_in': Port(UInt(XLEN)),   # 输入EX阶段结果
            'control_in': Port(UInt(32)),     # 输入控制信号
            'valid_in': Port(UInt(1)),        # 输入有效信号
        })
    
    @module.combinational
    def build(self, reg_file):
        mem_data_in, ex_result_in, control_in, valid_in = self.pop_all_ports(True)
        
        # 如果指令无效，直接返回
        with Condition(~valid_in):
            log("WB: Invalid instruction, skipping writeback")
            return
        
        # 解析控制信号
        reg_write = control_in[7] & (control_in[29:25] != UInt(5)(0))  # x0寄存器不会写入
        mem_to_reg = control_in[8]
        wb_rd = control_in[29:25]
        
        # 选择写回数据
        wb_data = UInt(XLEN)(0)
        
        with Condition(reg_write):
            wb_data = ex_result_in  # 默认从EX阶段的结果
            with Condition(mem_to_reg):
                wb_data = mem_data_in  # 从内存读取的数据
            # x0寄存器不会写入（已经在reg_write中处理）
            reg_file[wb_rd] <= wb_data
        
        log("WB: Write_Data={:08x}, RD={}, WE={}",
            wb_data, control_in[29:25], reg_write)

        
        # 冒险检测 - 连接所有必要的信号
        hazard_unit.async_called(
            id_ex_control=pipeline_regs[0].id_ex_control,
            id_ex_rs1_idx=pipeline_regs[0].id_ex_rs1_idx,
            id_ex_rs2_idx=pipeline_regs[0].id_ex_rs2_idx,
            ex_mem_control=pipeline_regs[0].ex_mem_control,
            mem_wb_control=pipeline_regs[0].mem_wb_control,
            id_ex_valid=pipeline_regs[0].id_ex_valid,
            ex_mem_valid=pipeline_regs[0].ex_mem_valid,
            mem_wb_valid=pipeline_regs[0].mem_wb_valid
        )

# ==================== 冒险检测单元 ===================
class HazardUnit(Module):
    """冒险检测单元"""
    def __init__(self):
        super().__init__(ports={
            'id_ex_control': Port(UInt(32)),          # ID/EX阶段控制信号
            'id_ex_rs1_idx': Port(UInt(5)),           # ID/EX阶段rs1索引
            'id_ex_rs2_idx': Port(UInt(5)),           # ID/EX阶段rs2索引
            'ex_mem_control': Port(UInt(32)),         # EX/MEM阶段控制信号
            'mem_wb_control': Port(UInt(32)),         # MEM/WB阶段控制信号
            'id_ex_valid': Port(UInt(1)),             # ID/EX阶段有效信号
            'ex_mem_valid': Port(UInt(1)),            # EX/MEM阶段有效信号
            'mem_wb_valid': Port(UInt(1))             # MEM/WB阶段有效信号
        })
    
    @module.combinational
    def build(self, pipeline_regs):
        # 获取所有需要的信号
        id_ex_control, id_ex_rs1_idx, id_ex_rs2_idx, ex_mem_control, mem_wb_control, id_ex_valid, ex_mem_valid, mem_wb_valid = self.pop_all_ports(True)
        
        # 解析各阶段指令的目标寄存器(rd)和写使能
        rd_mem = ex_mem_control[29:25]
        reg_write_mem = ex_mem_control[7:7] & (rd_mem != UInt(5)(0))  # x0寄存器不会写入
        
        rd_wb = mem_wb_control[29:25]
        reg_write_wb = mem_wb_control[7:7] & (rd_wb != UInt(5)(0))  # x0寄存器不会写入
        
        # 初始化数据冒险信号
        data_hazard_ex = UInt(1)(0)  # 与EX阶段指令的数据冒险
        data_hazard_wb = UInt(1)(0)   # 与WB阶段指令的数据冒险
        
        # 检测与EX阶段指令的数据冒险
        # 如果ID/EX阶段有效，且ID/EX阶段会写寄存器
        # 检查ID/EX阶段的rs1和rs2是否与后续阶段的rd冲突
        with Condition(id_ex_valid):
            # 检查ID/EX阶段指令是否需要读取rs1和rs2
            needs_rs1 = (id_ex_control[4:0] != UInt(5)(0)) | (id_ex_control[19:17] != UInt(3)(0)) | (id_ex_control[20:20] == UInt(1)(1))  # ALU操作、分支操作或跳转操作需要rs1
            needs_rs2 = (id_ex_control[10:9] == UInt(2)(0)) & ((id_ex_control[4:0] != UInt(5)(0)) | (id_ex_control[19:17] != UInt(3)(0)))  # 只有当ALU源是寄存器且是ALU或分支操作时才需要rs2
            
            with Condition(ex_mem_valid & reg_write_mem):
                with Condition((needs_rs1 & (id_ex_rs1_idx == rd_mem)) | (needs_rs2 & (id_ex_rs2_idx == rd_mem))):
                    data_hazard_ex = UInt(1)(1)
            
            # 检查与WB阶段的冲突
            with Condition(mem_wb_valid & reg_write_wb):
                with Condition((needs_rs1 & (id_ex_rs1_idx == rd_wb)) | (needs_rs2 & (id_ex_rs2_idx == rd_wb))):
                    data_hazard_wb = UInt(1)(1)
        
        # 综合数据冒险信号
        data_hazard = data_hazard_ex | data_hazard_wb
        
        # 最终输出信号
        stall_out = data_hazard  # 数据冒险需要暂停
        
        mem_rd = UInt(5)(0)
        with Condition(ex_mem_valid & reg_write_mem):
            mem_rd = rd_mem
        
        wb_rd_val = UInt(5)(0)
        with Condition(mem_wb_valid & reg_write_wb):
            wb_rd_val = rd_wb
            
        log("HAZARD: RS1={}, RS2={}, MEM_RD={}, WB_RD={}, Data_Hazard={}, Stall={}",
            id_ex_rs1_idx, id_ex_rs2_idx,
            mem_rd, wb_rd_val,
            data_hazard, stall_out)

# ==================== 顶层CPU模块 ===================
class Driver(Module):
    """五级流水线RV32I CPU"""
    def __init__(self, program_file="test_program.txt"):
        super().__init__(ports={})
        
        # 存储程序文件路径
        self.program_file = program_file
        
        # PC寄存器
        self.pc = RegArray(UInt(XLEN), 1, initializer=[0])
    
    def init_memory(self, program_file="test_program.txt"):
        """初始化内存内容 - 从指定文件加载程序到指令寄存器"""
        test_program = []
        
        try:
            # 尝试从文件读取指令
            with open(program_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释行
                    if not line or line.startswith('#'):
                        continue
                    # 支持十六进制格式（带或不带0x前缀）
                    if line.startswith('0x') or line.startswith('0X'):
                        instruction = int(line, 16)
                    else:
                        instruction = int(line, 0)  # 自动检测进制
                    test_program.append(instruction)
            
            print(f"Loaded {len(test_program)} instructions from {program_file}")
        
        except FileNotFoundError:
            print(f"Warning: Program file {program_file} not found. Using empty program.")
        except Exception as e:
            print(f"Error loading program from {program_file}: {e}")
        
        return test_program
    
    @module.combinational
    def build(self, pipeline_regs, instruction_memory, fetch_stage):
        # 初始化内存 - 在模块上下文中进行
        test_program = self.init_memory(self.program_file)
        
        # 直接加载测试程序到寄存器数组
        for i, instruction in enumerate(test_program):
            (instruction_memory & self)[i] <= UInt(XLEN)(instruction)
        
        print(f"Loaded {len(test_program)} instructions into instruction memory registers")
    
        current_pc = self.pc[0]

        # 调用fetch_stage并传递暂停信号
        fetch_stage.async_called(pc_in=current_pc, stall=stall_out)
        
        

def build_cpu(program_file="test_program.txt"):
    """构建RV32I CPU系统"""
    sys = SysBuilder('rv32i_cpu')
    with sys:

        pipeline_regs = RegArray(PIPELINE_REGS, 1, initializer=[PIPELINE_REGS.bundle(
            if_id_pc=UInt(XLEN)(0),
            if_id_instruction=UInt(XLEN)(0),
            if_id_valid=UInt(1)(0),
            id_ex_pc=UInt(XLEN)(0),
            id_ex_control=UInt(32)(0),
            id_ex_valid=UInt(1)(0),
            id_ex_rs1_idx=UInt(5)(0),
            id_ex_rs2_idx=UInt(5)(0),
            id_ex_immediate=UInt(XLEN)(0),
            ex_mem_pc=UInt(XLEN)(0),
            ex_mem_control=UInt(32)(0),
            ex_mem_valid=UInt(1)(0),
            ex_mem_result=UInt(XLEN)(0),
            ex_mem_data=UInt(XLEN)(0),
            mem_wb_control=UInt(32)(0),
            mem_wb_valid=UInt(1)(0),
            mem_wb_mem_data=UInt(XLEN)(0),
            mem_wb_ex_result=UInt(XLEN)(0)
        )])

        # 创建指令内存
        instruction_memory = RegArray(UInt(XLEN), 1024, initializer=[0x00000013]*1024)  # 初始化为NOP指令
        
        # 创建寄存器文件
        reg_file = RegArray(UInt(XLEN), REG_COUNT, initializer=[0]*REG_COUNT)
        
        hazard_unit = HazardUnit()
        fetch_stage = FetchStage()
        decode_stage = DecodeStage()
        execute_stage = ExecuteStage()
        memory_stage = MemoryStage()
        writeback_stage = WriteBackStage()
        driver = Driver(program_file=program_file)

        # 按照流水线顺序构建模块
        hazard_unit.build(pipeline_regs)
        writeback_stage.build(reg_file, hazard_unit)
        memory_stage.build(pipeline_regs, execute_stage, writeback_stage)
        pc_change, target_pc, alu_result = execute_stage.build(pipeline_regs, reg_file, memory_stage)
        decode_stage.build(pipeline_regs, reg_file, execute_stage)
        fetch_stage.build(pipeline_regs, instruction_memory, decode_stage)
        
        # 构建Driver模块，处理PC更新和stall信号
        driver.build(pipeline_regs, instruction_memory, fetch_stage)
    
    return sys

def test_rv32i_cpu(program_file="test_program.txt"):
    """测试RV32I CPU"""
    sys = build_cpu(program_file)
    
    # 生成模拟器
    simulator_path, _ = elaborate(sys, verilog=False)
    raw = utils.run_simulator(simulator_path)
    print(raw)

if __name__ == "__main__":
    test_rv32i_cpu(program_file="test_program.txt")
