#!/usr/bin/env python3
"""
五级流水线RV32I CPU实现 - 冯诺依曼架构
使用Assassyn语言实现完整的RISC-V 32位基础指令集处理器
统一内存架构：指令和数据共享同一个SRAM

关键设计：
1. 统一 SRAM 用于存储指令和数据
2. SRAM 只有一个读端口，需要仲裁 IF（取指）和 MEM（数据访问）
3. 仲裁策略：MEM 阶段优先，当 MEM 需要访问时暂停 IF
4. SRAM 读取有一周期延迟：本周期发请求，下周期在 dout 获取数据
"""

from assassyn.frontend import *
from assassyn.backend import elaborate
from assassyn import utils
from assassyn.ir.memory.sram import SRAM
from assassyn.ir.module import downstream, Downstream

# ==================== 常量定义 ===================
XLEN = 32  # RISC-V XLEN
REG_COUNT = 32  # 通用寄存器数量
CONTROL_LEN = 42 # 控制信号长度

# ==================== IF阶段：指令获取 ===================
class FetchStage(Module):
    """指令获取阶段(IF) - 冯诺依曼架构
    
    不直接访问内存，发送取指请求给仲裁器
    由于 SRAM 读取有一周期延迟，取指逻辑如下：
    - 周期 N: 发起 PC 对应地址的读请求
    - 周期 N+1: 从 SRAM.dout 获取指令
    """
    def __init__(self):
        super().__init__(ports={
        })
    
    @module.combinational
    def build(self, pc, stall, if_id_pc, if_id_instruction, if_id_valid, decode_stage):
        current_pc = pc[0]
        word_addr = current_pc >> UInt(XLEN)(2)

        log("IF_ID_VALID={}", if_id_valid[0])

        # 不再在这里更新 if_id_pc，改由 MemoryArbiter 统一更新
        # 这样可以使用最新的 stall 信息
        with Condition(~stall[0]):
            log("IF: PC={:08x}, Stall={}", current_pc, stall[0])

        decode_stage.async_called()

        # 返回取指请求信号：包含取指地址和使能信号
        fetch_enable = if_id_valid[0] & (~stall[0])
        fetch_signals = concat(
            word_addr,                              # [63:32] 取指地址（字地址）
            fetch_enable.bitcast(Bits(1)),          # [31] 取指使能
            if_id_instruction[0].bitcast(Bits(31))  # [30:0] 当前指令低31位（用于保持）
        )
        return fetch_signals

# ==================== ID阶段：指令解码 ===================
class DecodeStage(Module):
    """指令解码阶段(ID)
    
    职责：解码 IF/ID 寄存器中的指令，输出控制信号和寄存器索引
    不负责更新 ID/EX 寄存器，由 MemoryArbiter 统一管理
    """
    def __init__(self):
        super().__init__(ports={})
    
    @module.combinational
    def build(self, if_id_valid, if_id_pc, if_id_instruction, execute_stage):
        if_id_pc_in = if_id_pc[0]
        instruction = if_id_instruction[0]

        log("Instruction={:08x}", instruction)
        
        # 如果指令无效，直接返回，不更新ID/EX寄存器
        opcode = instruction[0:6]          # bits 6:0
        rd = instruction[7:11]             # bits 11:7
        func3 = instruction[12:14]          # bits 14:12
        rs1 = instruction[15:19]           # bits 19:15
        rs2 = instruction[20:24]           # bits 24:20
        funct7 = instruction[25:31]         # bits 31:25

        # 提取立即数 - 使用手动符号扩展
        # I型立即数 (12位有符号数)
        imm_i_bits = instruction[20:31]
        sign_bit_i = imm_i_bits[11:11]  # 获取符号位
        # 手动扩展符号位：如果符号位为1，则高位全为1；否则为0
        immediate_i = (sign_bit_i == UInt(1)(1)).select(
            concat(Bits(20)(0xFFFFF), imm_i_bits).bitcast(UInt(32)),  # 负数扩展
            concat(Bits(20)(0x00000), imm_i_bits).bitcast(UInt(32))   # 正数扩展
        )
        
        # S型立即数 (12位有符号数)
        imm_s_bits = concat(instruction[25:31], instruction[7:11])
        sign_bit_s = imm_s_bits[11:11]  # 获取符号位
        immediate_s = (sign_bit_s == UInt(1)(1)).select(
            concat(Bits(20)(0xFFFFF), imm_s_bits).bitcast(UInt(32)),  # 负数扩展
            concat(Bits(20)(0x00000), imm_s_bits).bitcast(UInt(32))   # 正数扩展
        )
        
        # B型立即数 (13位有符号数，左移1位)
        imm_b_bits = concat(instruction[31:31], instruction[7:7], instruction[25:30], instruction[8:11], UInt(1)(0))
        sign_bit_b = imm_b_bits[12:12]  # 获取符号位
        immediate_b = (sign_bit_b == UInt(1)(1)).select(
            concat(Bits(19)(0x7FFFF), imm_b_bits).bitcast(UInt(32)),  # 负数扩展
            concat(Bits(19)(0x00000), imm_b_bits).bitcast(UInt(32))   # 正数扩展
        )
        
        # U型立即数 (20位无符号数，左移12位)
        immediate_u = (instruction[12:31] << UInt(XLEN)(12)).bitcast(UInt(32))
        
        # J型立即数 (21位有符号数，左移1位)
        imm_j_bits = concat(instruction[31:31], instruction[12:19], instruction[20:20], instruction[21:30], UInt(1)(0))
        sign_bit_j = imm_j_bits[20:20]  # 获取符号位
        immediate_j = (sign_bit_j == UInt(1)(1)).select(
            concat(Bits(11)(0x7FF), imm_j_bits).bitcast(UInt(32)),  # 负数扩展
            concat(Bits(11)(0x000), imm_j_bits).bitcast(UInt(32))   # 正数扩展
        )
        
        # 控制信号解码
        alu_op = UInt(5)(0)
        mem_read = UInt(1)(0)
        mem_write = UInt(1)(0)
        reg_write = UInt(1)(0)
        mem_to_reg = UInt(1)(0)
        alu_src = UInt(2)(0)  # 00:寄存器, 01:立即数, 10:PC
        branch_op = UInt(3)(0)
        jump_op = UInt(1)(0)  # 跳转指令标志
        jumpr_op = UInt(1)(0)  # 寄存器跳转指令标志
        immediate = UInt(XLEN)(0)  # 初始化立即数
        
        is_r_type = (opcode == UInt(7)(0b0110011))
        is_i_type = (opcode == UInt(7)(0b0010011))
        is_l_type = (opcode == UInt(7)(0b0000011))
        is_s_type = (opcode == UInt(7)(0b0100011))
        is_b_type = (opcode == UInt(7)(0b1100011))
        is_j_type = (opcode == UInt(7)(0b1101111))
        is_jr_type = (opcode == UInt(7)(0b1100111))
        is_lui_type = (opcode == UInt(7)(0b0110111))
        is_auipc_type = (opcode == UInt(7)(0b0010111))
        alu_op_tmp = UInt(5)(0)
        alu_op_tmp = ((is_r_type & funct7[5:5] == UInt(1)(1)) & (func3 == UInt(3)(0b000))).select(UInt(5)(0b00001), alu_op_tmp)  # SUB
        alu_op_tmp = ((funct7[5:5] == UInt(1)(1)) & (func3 == UInt(3)(0b101))).select(UInt(5)(0b00110), alu_op_tmp)  # SRA
        alu_op_tmp = (~(is_r_type & funct7[5:5] == UInt(1)(1)) & (func3 == UInt(3)(0b000))).select(UInt(5)(0b00000), alu_op_tmp)  # ADD
        alu_op_tmp = (func3 == UInt(3)(0b111)).select(UInt(5)(0b01001), alu_op_tmp)  # AND
        alu_op_tmp = (func3 == UInt(3)(0b110)).select(UInt(5)(0b01000), alu_op_tmp)  # OR
        alu_op_tmp = (func3 == UInt(3)(0b100)).select(UInt(5)(0b00100), alu_op_tmp)  # XOR
        alu_op_tmp = (func3 == UInt(3)(0b010)).select(UInt(5)(0b00011), alu_op_tmp)  # SLT
        alu_op_tmp = (func3 == UInt(3)(0b011)).select(UInt(5)(0b00111), alu_op_tmp)  # SLTU
        alu_op_tmp = (func3 == UInt(3)(0b001)).select(UInt(5)(0b00010), alu_op_tmp)  # SLL
        alu_op_tmp = ((funct7[5:5] == UInt(1)(0)) & (func3 == UInt(3)(0b101))).select(UInt(5)(0b00101), alu_op_tmp)  # SRL
        alu_op = (is_r_type | is_i_type).select(alu_op_tmp, alu_op)
        reg_write = (is_r_type | is_i_type).select(UInt(1)(1), reg_write)
        alu_src = is_r_type.select(UInt(2)(0), alu_src)
        alu_src = is_i_type.select(UInt(2)(1), alu_src)
        immediate = is_i_type.select(immediate_i, immediate)
        
        mem_read = is_l_type.select(UInt(1)(1), mem_read)  # LW (Load Word)
        reg_write = is_l_type.select(UInt(1)(1), reg_write)  # x0寄存器不会写入
        mem_to_reg = is_l_type.select(UInt(1)(1), mem_to_reg)  # LW (Load Word)
        alu_src = is_l_type.select(UInt(2)(1), alu_src)
        immediate = is_l_type.select(immediate_i, immediate)
            
        store_type_bits = UInt(2)(0)
        mem_write = is_s_type.select(UInt(1)(1), mem_write)  # SW (Store Word)
        alu_src = is_s_type.select(UInt(2)(1), alu_src)
        immediate = is_s_type.select(immediate_s, immediate)
        store_type_bits = (is_s_type & (func3 == UInt(3)(0b010))).select(UInt(2)(0b10), store_type_bits)  # SW (Store Word)
        store_type_bits = (is_s_type & (func3 == UInt(3)(0b000))).select(UInt(2)(0b00), store_type_bits)  # SB (Store Byte)
        store_type_bits = (is_s_type & (func3 == UInt(3)(0b001))).select(UInt(2)(0b01), store_type_bits)  # SH (Store Halfword)

        branch_op_tmp = UInt(3)(0)
        branch_op_tmp = (func3 == UInt(3)(0b000)).select(UInt(3)(0b001), branch_op_tmp)  # BEQ
        branch_op_tmp = (func3 == UInt(3)(0b001)).select(UInt(3)(0b010), branch_op_tmp)  # BNE
        branch_op_tmp = (func3 == UInt(3)(0b100)).select(UInt(3)(0b011), branch_op_tmp)  # BLT
        branch_op_tmp = (func3 == UInt(3)(0b101)).select(UInt(3)(0b100), branch_op_tmp)  # BGE
        branch_op_tmp = (func3 == UInt(3)(0b110)).select(UInt(3)(0b101), branch_op_tmp)  # BLTU
        branch_op_tmp = (func3 == UInt(3)(0b111)).select(UInt(3)(0b110), branch_op_tmp)  # BGEU
        immediate = is_b_type.select(immediate_b, immediate)
        branch_op = is_b_type.select(branch_op_tmp, branch_op)
            
        reg_write = (is_lui_type | is_auipc_type).select(UInt(1)(1), reg_write)
        alu_src = is_lui_type.select(UInt(2)(1), alu_src)
        immediate = (is_lui_type | is_auipc_type).select(immediate_u, immediate)
        alu_src = is_auipc_type.select(UInt(2)(2), alu_src)
        
        reg_write = is_j_type.select(UInt(1)(1), reg_write)
        alu_src = is_j_type.select(UInt(2)(1), alu_src)
        immediate = is_j_type.select(immediate_j, immediate)
        jump_op = is_j_type.select(UInt(1)(1), jump_op)

        reg_write = is_jr_type.select(UInt(1)(1), reg_write)
        alu_src = is_jr_type.select(UInt(2)(1), alu_src)
        immediate = is_jr_type.select(immediate_i, immediate)
        jumpr_op = is_jr_type.select(UInt(1)(1), jumpr_op)

        reg_write = (rd == UInt(5)(0)).select(UInt(1)(0), reg_write)  # rd为x0时不写入
        
        control_signals = concat(
            immediate[0:11],   # [41:30] 立即数低12位
            rd,               # [29:25] rd地址
            UInt(1)(0),       # [24]    保留位
            store_type_bits,  # [23:22] 存储类型: 00=SB, 01=SH, 10=SW
            jumpr_op,       # [21]    保留位
            jump_op,          # [20]    跳转指令标志
            branch_op,        # [19:17] 分支操作类型
            UInt(6)(0),       # [16:11] 保留位
            alu_src,          # [10:9]  ALU输入选择
            mem_to_reg,       # [8]     内存到寄存器
            reg_write,        # [7]     寄存器写
            mem_write,        # [6]     内存写
            mem_read,         # [5]     内存读
            alu_op,           # [4:0]   ALU操作码
        )

        need_rs1 = (is_i_type | is_r_type | is_s_type | is_b_type | is_l_type | is_jr_type)
        need_rs2 = (is_r_type | is_s_type | is_b_type)
        
        # 日志输出
        with Condition(if_id_valid[0]):
            log("ID: PC={}, Opcode={:07b}, RD={}, RS1={}, RS2={}, Immediate={}, Alu_op={}, Branch_op={}, Jump_op={}, Alu_src={}, Mem_read={}, Mem_write={}, Reg_write={}, Mem_to_reg={}, Control={:042b}",
                if_id_pc_in, opcode, rd, rs1, rs2, immediate, alu_op, branch_op, jump_op, alu_src, mem_read, mem_write, reg_write, mem_to_reg, control_signals)

        execute_stage.async_called()

        # 简化：DecodeStage 只输出当前解码的信号，不管 id_ex_valid
        # 流水线控制完全由 MemoryArbiter 负责
        decode_signals = concat(
            need_rs2.bitcast(Bits(1)),           # [86] 需要 rs2
            need_rs1.bitcast(Bits(1)),           # [85] 需要 rs1
            immediate.bitcast(Bits(XLEN)),       # [84:53] 立即数
            rs2.bitcast(Bits(5)),                # [52:48] rs2
            rs1.bitcast(Bits(5)),                # [47:43] rs1
            control_signals.bitcast(Bits(CONTROL_LEN)),  # [42:1] 控制信号
            if_id_valid[0].bitcast(Bits(1)),     # [0] IF/ID 有效标志
        )
        return decode_signals

# ==================== EX阶段：执行 ===================
class ExecuteStage(Module):
    """执行阶段(EX)"""
    def __init__(self):
        super().__init__(ports={})
    
    def alu_unit(self, op: Value, a: Value, b: Value):
        
        # 默认结果
        result = UInt(XLEN)(0)
        zero = UInt(1)(0)
        a_signed = a.bitcast(Int(XLEN))
        b_signed = b.bitcast(Int(XLEN))
        
        # 根据操作码执行不同操作
        result = (op == UInt(5)(0b00000)).select(a + b, result)  # ADD
        result = (op == UInt(5)(0b00001)).select(a - b, result)  # SUB
        result = (op == UInt(5)(0b00010)).select((a << (b & UInt(XLEN)(0x1F))).bitcast(UInt(XLEN)), result)  # SLL
        result = (op == UInt(5)(0b00011)).select((a_signed < b_signed).select(UInt(XLEN)(1), UInt(XLEN)(0)), result)  # SLT
        result = (op == UInt(5)(0b00100)).select((a ^ b).bitcast(UInt(XLEN)), result)  # XOR
        result = (op == UInt(5)(0b00101)).select((a >> (b & UInt(XLEN)(0x1F))).bitcast(UInt(XLEN)), result)  # SRL
        result = (op == UInt(5)(0b00110)).select((a_signed >> (b & UInt(XLEN)(0x1F))).bitcast(UInt(XLEN)), result)  # SRA
        result = (op == UInt(5)(0b00111)).select((a < b).select(UInt(XLEN)(1), UInt(XLEN)(0)), result)  # SLTU
        result = (op == UInt(5)(0b01000)).select((a | b).bitcast(UInt(XLEN)), result)  # OR
        result = (op == UInt(5)(0b01001)).select((a & b).bitcast(UInt(XLEN)), result)  # AND
        
        log("ALU: OP={:05b}, A={:08x}, B={:08x}, Result={:08x}",
            op, a, b, result)
        
        return result

    def branch_unit(self, op: Value, a: Value, b: Value):
        
        taken = UInt(1)(0)
        a_signed = a.bitcast(Int(XLEN))
        b_signed = b.bitcast(Int(XLEN))
        taken = (op == UInt(3)(0b001)).select((a == b).select(UInt(1)(1), UInt(1)(0)), taken)  # BEQ
        taken = (op == UInt(3)(0b010)).select((a != b).select(UInt(1)(1), UInt(1)(0)), taken)  # BNE
        taken = (op == UInt(3)(0b011)).select((a_signed < b_signed).select(UInt(1)(1), UInt(1)(0)), taken)  # BLT
        taken = (op == UInt(3)(0b100)).select((a_signed >= b_signed).select(UInt(1)(1), UInt(1)(0)), taken)  # BGE
        taken = (op == UInt(3)(0b101)).select((a < b).select(UInt(1)(1), UInt(1)(0)), taken)  # BLTU
        taken = (op == UInt(3)(0b110)).select((a >= b).select(UInt(1)(1), UInt(1)(0)), taken)  # BGEU
        
        log("BRANCH: OP={:03b}, A={:08x}, B={:08x}, Taken={}",
            op, a, b, taken)
        
        return taken

    @module.combinational
    def build(self, id_ex_valid, id_ex_pc, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, id_ex_control, ex_mem_pc, ex_mem_control, ex_mem_valid, ex_mem_result, ex_mem_data, reg_file, memory_stage):
        pc_in = id_ex_pc[0]
        rs1_idx = id_ex_rs1_idx[0]
        rs2_idx = id_ex_rs2_idx[0]
        immediate_in = id_ex_immediate[0]
        control_in = id_ex_control[0]

        # 直接从寄存器文件读取rs1和rs2的值
        rs1_data = reg_file[rs1_idx]
        rs2_data = reg_file[rs2_idx]
        
        # 初始化PC变化控制信号
        pc_change = UInt(1)(0)
        target_pc = pc_in + UInt(XLEN)(4)  # 默认目标PC是PC+4

        # 解析控制信号
        alu_op = control_in[0:4]
        mem_read = control_in[5:5]
        mem_write = control_in[6:6]
        reg_write = control_in[7:7]
        mem_to_reg = control_in[8:8]
        alu_src = control_in[9:10]
        branch_op = control_in[17:19]  # 修正：branch_op在[19:17]位
        jump_op = control_in[20:20]  # 跳转指令标志
        jumpr_op = control_in[21:21]  # 寄存器跳转指令标志
        rd_addr = control_in[25:29]  # rd地址
        immediate = control_in[22:31]  # 立即数
        
        # ALU输入B选择
        alu_b = immediate_in
        alu_b = (alu_src == UInt(2)(0)).select(rs2_data, alu_b)
        
        # 根据指令类型决定执行ALU操作还是分支操作
        alu_result = UInt(XLEN)(0)
        
        # 判断是否为分支指令 (branch_op != 0)
        is_branch = (branch_op != UInt(3)(0b000))
        is_jump = (jump_op == UInt(1)(1))
        is_jumpr = (jumpr_op == UInt(1)(1))
        
        # 对于AUIPC指令，ALU输入A应该是PC而不是rs1_data
        alu_a = rs1_data
        alu_a = (alu_src == UInt(2)(2)).select(pc_in, alu_a)

        branch_result = is_branch.select(self.branch_unit(branch_op, rs1_data, rs2_data), UInt(1)(0))
        alu_result = is_branch.select(UInt(XLEN)(0), (is_jump | is_jumpr).select(pc_in + UInt(XLEN)(4), self.alu_unit(alu_op, alu_a, alu_b)))
        target_pc = (is_branch | is_jump).select(pc_in + immediate_in, target_pc)
        new_pc_temp = rs1_data + immediate_in
        new_pc = (new_pc_temp ^ (new_pc_temp & UInt(XLEN)(1)))
        target_pc = is_jumpr.select(new_pc.bitcast(UInt(32)), target_pc)
        pc_change = (branch_result.bitcast(Bits(1)) | is_jump | is_jumpr).select(UInt(1)(1), pc_change)

        with Condition(is_jump & (immediate_in == UInt(XLEN)(0))):
            log("Finish Execution. The result is {}", reg_file[10])
            finish()
        
        # EX 阶段始终执行，不管 id_ex_valid 的值
        # id_ex_valid 只决定下一周期的 ID/EX 寄存器是否更新
        # EX 阶段的当前指令应该始终传递到 MEM 阶段
        with Condition(ex_mem_valid[0]):
            ex_mem_pc[0] = pc_in
            ex_mem_control[0] = control_in
            ex_mem_result[0] = alu_result
            ex_mem_data[0] = rs2_data
            
            log("EX: PC={}, ALU_OP={:05b}, ALU_A={}, ALU_B={}, Result={:08x}, PC_Change={}, Target_PC={:08x}, Immediate={:08x}, ALU_SRC={}",
                pc_in, alu_op, alu_a, alu_b, alu_result, pc_change, target_pc, immediate_in, alu_src)
        
        memory_stage.async_called()

        # execute_signals 格式: [74:33]control(42), [32:1]target_pc(32), [0]pc_change(1)
        # 关键修正：pc_change 不能被 id_ex_valid 屏蔽！
        # 即使 id_ex_valid=0（数据冒险期间），跳转信号也必须传递出去
        # 否则会导致死循环：数据冒险 → id_ex_valid=0 → pc_change=0 → 继续数据冒险
        execute_signals = concat(
            id_ex_valid[0].select(control_in.bitcast(Bits(CONTROL_LEN)), Bits(CONTROL_LEN)(0)),
            target_pc.bitcast(Bits(XLEN)),   # [32:1] 目标PC（始终传递）
            pc_change.bitcast(Bits(1)),      # [0] PC变化标志（始终传递，不被 id_ex_valid 屏蔽）
        )

        return execute_signals

# ==================== MEM阶段：内存访问 ===================
class MemoryStage(Module):
    """内存访问阶段(MEM) - 冯诺依曼架构
    
    不直接访问 SRAM，发送内存访问请求给仲裁器
    实际的 SRAM 访问在 MemoryArbiter 中完成
    """
    def __init__(self):
        super().__init__(ports={})
    
    @module.combinational
    def build(self, ex_mem_valid, ex_mem_result, ex_mem_pc, ex_mem_data, ex_mem_control, mem_wb_control, mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, writeback_stage):
        pc_in = ex_mem_pc[0]
        addr_in = ex_mem_result[0]
        data_in = ex_mem_data[0]
        control_in = ex_mem_control[0]
        
        # 解析控制信号
        mem_read = control_in[5:5]
        mem_write = control_in[6:6]
        store_type = control_in[22:23]  # 存储类型: 00=SB, 01=SH, 10=SW
        
        word_addr = addr_in >> UInt(XLEN)(2)

        with Condition(mem_wb_valid[0]):
            mem_wb_control[0] = ex_mem_valid[0].select(control_in, UInt(CONTROL_LEN)(0))
            mem_wb_ex_result[0] = ex_mem_valid[0].select(ex_mem_result[0], UInt(XLEN)(0))
            
            log("MEM: PC={}, Addr={:08x}, WordAddr={:08x}, Read={}, Write={}, data_in={}",
                pc_in, addr_in, word_addr, mem_read, mem_write, data_in)

        writeback_stage.async_called()

        # 返回内存访问请求信号
        # 格式: [106:75]=wdata(32), [74:43]=addr(32), [42]=access(1), [41]=we(1), [40]=re(1), [39:0]=unused/padding
        # 总位宽 = 32 + 32 + 1 + 1 + 1 + 40 = 107 bits
        mem_access = ex_mem_valid[0] & (mem_read | mem_write)
        memory_signals = concat(
            data_in.bitcast(Bits(XLEN)),           # [106:75] 写入数据 (32位)
            word_addr.bitcast(Bits(XLEN)),         # [74:43]  字地址 (32位)
            mem_access.bitcast(Bits(1)),           # [42]     内存访问使能
            mem_write.bitcast(Bits(1)),            # [41]     写使能
            mem_read.bitcast(Bits(1)),             # [40]     读使能
            Bits(40)(0)                            # [39:0]   填充位
        )
        return memory_signals

# ==================== WB阶段：写回 ===================
class WriteBackStage(Module):
    """写回阶段(WB)"""
    def __init__(self):
        super().__init__(ports={})
    
    @module.combinational
    def build(self, mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, mem_wb_control, reg_file, unified_sram):
        # 从统一 SRAM 读取数据（上一周期 MEM 阶段发起的读请求）
        mem_data_in = unified_sram.dout[0]
        ex_result_in = mem_wb_ex_result[0]
        control_in = mem_wb_control[0]
        
            # 解析控制信号
        reg_write = control_in[7:7]
        mem_to_reg = control_in[8:8]
        wb_rd = control_in[25:29]
            
        # 选择写回数据
        wb_data = mem_to_reg.select(mem_data_in, ex_result_in)
            
        # 如果指令无效，直接返回
        with Condition(mem_wb_valid[0]):
            with Condition(reg_write):
                reg_file[wb_rd] = wb_data
            log("WB: Write_Data={}, RD={}, WE={}",
                wb_data, wb_rd, reg_write)

        writeback_signals = control_in.bitcast(Bits(CONTROL_LEN))
        return writeback_signals

class HazardUnit(Downstream):
    """数据冒险检测单元
    
    职责：检测 ID 阶段指令与 EX/MEM 阶段的数据冒险
    
    冒险检测逻辑：
    - 检测 ID 阶段的 rs1/rs2 是否与 EX/MEM 阶段的 rd 冲突
    - 检测 Store 指令的 RAW 冒险（后续指令读取 Store 的数据）
    - 只有当前周期 EX/MEM 有写寄存器的指令时才可能有冒险
    - 使用暂停计数器处理多周期暂停
    """
    def __init__(self):
        super().__init__()

    @downstream.combinational
    def build(self, stall_counter, id_ex_control, ex_mem_control, decode_signals, execute_signals):
        
        execute_signals = execute_signals.optional(Bits(XLEN + 1 + CONTROL_LEN)(0))
        # decode_signals 新格式: [86]need_rs2, [85]need_rs1, [84:53]imm, [52:48]rs2, [47:43]rs1, [42:1]control, [0]if_id_valid
        decode_signals = decode_signals.optional(Bits(87)(0))
        
        # 解析 execute_signals: [74:33]control, [32:1]target_pc, [0]pc_change
        pc_change = execute_signals[0:0].bitcast(UInt(1))
        
        # 解析 decode_signals
        if_id_valid = decode_signals[0:0].bitcast(UInt(1))
        control_in = decode_signals[1:CONTROL_LEN].bitcast(UInt(CONTROL_LEN))
        rs1 = decode_signals[CONTROL_LEN + 1:CONTROL_LEN + 5].bitcast(UInt(5))
        rs2 = decode_signals[CONTROL_LEN + 6:CONTROL_LEN + 10].bitcast(UInt(5))
        immediate = decode_signals[CONTROL_LEN + 11:CONTROL_LEN + 10 + XLEN].bitcast(UInt(XLEN))
        needs_rs1 = decode_signals[CONTROL_LEN + 11 + XLEN:CONTROL_LEN + 11 + XLEN].bitcast(UInt(1))
        needs_rs2 = decode_signals[CONTROL_LEN + 12 + XLEN:CONTROL_LEN + 12 + XLEN].bitcast(UInt(1))
        
        # ==================== 数据冒险检测 ===================
        # EX 阶段（id_ex_control）
        ex_stage_control = id_ex_control[0]
        rd_ex = ex_stage_control[25:29]
        reg_write_ex = ex_stage_control[7:7]
        
        # MEM 阶段（ex_mem_control）
        mem_stage_control = ex_mem_control[0]
        rd_mem = mem_stage_control[25:29]
        reg_write_mem = mem_stage_control[7:7]
        
        # 检测与 EX 阶段的数据冒险（需要等待 2 个周期）
        hazard_rs1_ex = if_id_valid & needs_rs1 & reg_write_ex & (rs1 == rd_ex) & (rd_ex != UInt(5)(0))
        hazard_rs2_ex = if_id_valid & needs_rs2 & reg_write_ex & (rs2 == rd_ex) & (rd_ex != UInt(5)(0))
        data_hazard_ex = hazard_rs1_ex | hazard_rs2_ex
        
        # 检测与 MEM 阶段的数据冒险（需要等待 1 个周期）
        hazard_rs1_mem = if_id_valid & needs_rs1 & reg_write_mem & (rs1 == rd_mem) & (rd_mem != UInt(5)(0))
        hazard_rs2_mem = if_id_valid & needs_rs2 & reg_write_mem & (rs2 == rd_mem) & (rd_mem != UInt(5)(0))
        # 排除已被 EX 覆盖的情况
        data_hazard_mem = (hazard_rs1_mem | hazard_rs2_mem) & (~data_hazard_ex)
        
        # 暂停计数器逻辑
        current_count = stall_counter[0]
        is_stalling = current_count > UInt(2)(0)
        
        # 新冒险检测（只在不暂停时检测）
        new_hazard_ex = data_hazard_ex & (~is_stalling)
        new_hazard_mem = data_hazard_mem & (~is_stalling)
        
        # 计算新的计数值
        # 优先级：pc_change > 新冒险 > 继续暂停
        new_count = UInt(2)(0)
        new_count = is_stalling.select(current_count - UInt(2)(1), new_count)
        new_count = new_hazard_mem.select(UInt(2)(1), new_count)
        new_count = new_hazard_ex.select(UInt(2)(2), new_count)
        new_count = pc_change.select(UInt(2)(0), new_count)  # 跳转清零
        
        stall_counter[0] = new_count
        
        # 最终数据冒险信号
        # 当 pc_change=1 时，强制为 0（跳转优先）
        raw_hazard = (new_hazard_ex | new_hazard_mem | is_stalling).select(UInt(1)(1), UInt(1)(0))
        data_hazard = pc_change.select(UInt(1)(0), raw_hazard)
        
        log("HazardUnit: if_id_valid={}, pc_change={}, rd_ex={}, rd_mem={}, rs1={}, rs2={}, needs_rs1={}, needs_rs2={}",
            if_id_valid, pc_change, rd_ex, rd_mem, rs1, rs2, needs_rs1, needs_rs2)
        log("HazardUnit: hazard_ex={}, hazard_mem={}, data_hazard={}, stall_count={}",
            data_hazard_ex, data_hazard_mem, data_hazard, new_count)
        
        # 返回冒险信号
        # 格式：[85]data_hazard, [84:53]immediate, [52:48]rs2, [47:43]rs1, [42:1]control, [0]if_id_valid
        hazard_signals = concat(
            data_hazard.bitcast(Bits(1)),           # [85] 数据冒险
            immediate.bitcast(Bits(XLEN)),          # [84:53] 立即数
            rs2.bitcast(Bits(5)),                   # [52:48] rs2
            rs1.bitcast(Bits(5)),                   # [47:43] rs1
            control_in.bitcast(Bits(CONTROL_LEN)),  # [42:1] 控制信号
            if_id_valid.bitcast(Bits(1)),           # [0] if_id_valid
        )
        # 总位宽: 1 + 32 + 5 + 5 + 42 + 1 = 86 bits
        return hazard_signals


class LoadDataHandler(Downstream):
    """Load 数据处理单元
    
    职责：保存 MEM 阶段从 SRAM 读取的数据到 MEM/WB 寄存器
    
    由于 SRAM 有一周期延迟，MEM 阶段发起读请求后，
    WB 阶段才能从 SRAM.dout 获取数据。
    需要使用 Downstream 来正确处理这个延迟。
    """
    def __init__(self):
        super().__init__()

    @downstream.combinational
    def build(self, mem_wb_mem_data, mem_wb_valid, mem_wb_control, unified_sram):
        # 解析控制信号
        control_in = mem_wb_control[0]
        mem_read = control_in[5:5]
        
        # 只有在 load 指令且有效时才保存数据
        should_save = mem_wb_valid[0] & mem_read
        with Condition(should_save):
            mem_wb_mem_data[0] = unified_sram.dout[0]
        
        return


class MemoryArbiter(Downstream):
    """内存仲裁器 - 冯诺依曼架构核心
    
    职责：
    1. SRAM 仲裁（MEM 优先于 IF）
    2. 处理三种停机情况：数据冒险、结构冲突、控制冒险
    3. 更新 PC 和流水线寄存器
    4. 处理跳转指令缓冲区
    
    停机情况处理：
    - 数据冒险：IF 保持，ID 保持，EX 插入气泡
    - 结构冲突（MEM 访问 SRAM）：IF 暂停一周期，ID 保持，EX 插入气泡
    - 控制冒险（跳转）：IF/ID 插入 NOP，等 SRAM 延迟
    """
    def __init__(self):
        super().__init__()

    @downstream.combinational
    def build(self, unified_sram, pc, prev_pc, stall, jump_pending, if_id_valid, if_id_instruction, if_id_pc, id_ex_pc, id_ex_control, id_ex_valid, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, ex_mem_valid, mem_wb_valid, fetch_signals, execute_signals, memory_signals, hazard_signals, jump_instruction_buffer, jump_buffer_valid):

        execute_signals = execute_signals.optional(Bits(XLEN + 1 + CONTROL_LEN)(0))
        fetch_signals = fetch_signals.optional(Bits(XLEN + 1 + 31)(0))
        memory_signals = memory_signals.optional(Bits(107)(0))
        # hazard_signals 新格式: [85]data_hazard, [84:53]imm, [52:48]rs2, [47:43]rs1, [42:1]control, [0]if_id_valid
        hazard_signals = hazard_signals.optional(Bits(86)(0))

        # ==================== 解析信号 ===================
        # execute_signals: [74:33]control, [32:1]target_pc, [0]pc_change
        pc_change = execute_signals[0:0].bitcast(UInt(1))
        target_pc = execute_signals[1:XLEN].bitcast(UInt(XLEN))
        
        # fetch_signals: [63:32]addr, [31]enable, [30:0]unused
        fetch_addr = fetch_signals[32:63].bitcast(UInt(XLEN))
        fetch_enable = fetch_signals[31:31].bitcast(UInt(1))
        
        # memory_signals: [106:75]wdata, [74:43]addr, [42]access, [41]we, [40]re
        mem_read = memory_signals[40:40].bitcast(UInt(1))
        mem_write = memory_signals[41:41].bitcast(UInt(1))
        mem_access = memory_signals[42:42].bitcast(UInt(1))
        mem_addr = memory_signals[43:74].bitcast(UInt(XLEN))
        mem_wdata = memory_signals[75:106].bitcast(UInt(XLEN))
        
        # hazard_signals 新格式: [85]data_hazard, [84:53]imm, [52:48]rs2, [47:43]rs1, [42:1]control, [0]if_id_valid_in
        if_id_valid_in = hazard_signals[0:0].bitcast(UInt(1))
        control_in = hazard_signals[1:CONTROL_LEN].bitcast(UInt(CONTROL_LEN))
        rs1 = hazard_signals[CONTROL_LEN + 1:CONTROL_LEN + 5].bitcast(UInt(5))
        rs2 = hazard_signals[CONTROL_LEN + 6:CONTROL_LEN + 10].bitcast(UInt(5))
        immediate = hazard_signals[CONTROL_LEN + 11:CONTROL_LEN + 10 + XLEN].bitcast(UInt(XLEN))
        data_hazard = hazard_signals[CONTROL_LEN + 11 + XLEN:CONTROL_LEN + 11 + XLEN].bitcast(UInt(1))

        # ==================== 状态读取 ===================
        is_jump_pending = jump_pending[0]
        is_buffer_valid = jump_buffer_valid[0]
        current_pc = pc[0]
        
        # ==================== SRAM 仲裁 ===================
        # 关键修复：跳转时强制取指
        # 当 pc_change=1 时，使用 target_pc 作为取指地址
        # 当 jump_pending=1 时，使用当前 PC（已更新为 target_pc）
        
        # 跳转取指：pc_change 时用 target_pc，jump_pending 时用当前 PC
        jump_fetch = (pc_change | is_jump_pending).bitcast(UInt(1))
        jump_fetch_addr = pc_change.select(
            (target_pc >> UInt(XLEN)(2)).bitcast(UInt(XLEN)),  # pc_change: 用 target_pc
            (current_pc >> UInt(XLEN)(2)).bitcast(UInt(XLEN))  # jump_pending: 用当前 PC（已是 target_pc）
        )
        
        # 最终取指地址和使能
        # 优先级：MEM > 跳转取指 > 正常取指
        actual_fetch_enable = (jump_fetch | fetch_enable).bitcast(UInt(1))
        actual_fetch_addr = jump_fetch.select(jump_fetch_addr, fetch_addr.bitcast(UInt(XLEN)))
        
        # MEM 优先于 IF
        sram_we = (mem_access & mem_write).bitcast(UInt(1))
        sram_re = mem_access.select(mem_read.bitcast(UInt(1)), actual_fetch_enable).bitcast(UInt(1))
        sram_addr = mem_access.select(mem_addr, actual_fetch_addr)
        
        unified_sram.build(we=sram_we, re=sram_re, addr=sram_addr, wdata=mem_wdata)
        
        # ==================== 停机信号计算 ===================
        # 结构冲突：MEM 访问 SRAM 时，IF 无法取指
        mem_stall = mem_access
        
        # 总停机信号（不包括 pc_change，因为 pc_change 不是"停机"，而是"重定向"）
        total_stall = data_hazard | mem_stall | is_jump_pending
        
        # ==================== 流水线有效信号 ===================
        # id_ex_valid: 当数据冒险、跳转、结构冲突时，EX 阶段执行 NOP
        id_ex_valid[0] = (~data_hazard) & (~is_jump_pending) & (~pc_change) & (~mem_stall)
        # if_id_valid: 当总停机或跳转时，ID 阶段执行 NOP
        # 修正：数据冒险时 IF 保持，ID 保持，EX 插入气泡
        # EX/MEM 和 MEM/WB 总是有效，它们的流水继续推进
        ex_mem_valid[0] = UInt(1)(1)
        mem_wb_valid[0] = UInt(1)(1)
        stall[0] = total_stall | pc_change

        # ==================== IF/ID 寄存器更新 ===================
        fetched_instruction = unified_sram.dout[0]
        
        # 跳转或跳转等待时插入 NOP
        should_nop = pc_change | is_jump_pending
        
        # 指令选择：
        # 1. 跳转等待且缓冲区有效 → 使用缓冲区指令
        # 2. 跳转/跳转等待 → NOP
        # 3. 数据冒险/结构冲突 → 保持当前指令
        # 4. 正常 → 使用取到的指令
        is_jump_and_buffer = is_jump_pending & is_buffer_valid
        instruction_from_buffer = is_jump_and_buffer.select(jump_instruction_buffer[0], UInt(XLEN)(0x00000013))
        instruction_from_normal = should_nop.select(UInt(XLEN)(0x00000013), (data_hazard | mem_stall).select(if_id_instruction[0], fetched_instruction))
        final_instruction = is_jump_and_buffer.select(instruction_from_buffer, instruction_from_normal)
        if_id_instruction[0] = final_instruction
        
        # PC 对应关系：由于 SRAM 延迟，fetched_instruction 对应 prev_pc
        # 当跳转等待且缓冲区有效时，使用 current_pc（已是 target_pc）
        pc_from_buffer = is_jump_and_buffer.select(current_pc, if_id_pc[0])
        final_pc = (total_stall | should_nop).select(pc_from_buffer, prev_pc[0])
        if_id_pc[0] = final_pc
        
        # ==================== 跳转指令缓冲区管理 ====================
        # 当 pc_change=1 时，保存取到的指令到缓冲区
        with Condition(pc_change):
            jump_instruction_buffer[0] = fetched_instruction
            jump_buffer_valid[0] = UInt(1)(1)

        # ==================== PC 更新 ===================
        # 跳转 → target_pc
        # 停机 → 保持
        # 正常 → PC + 4
        new_pc = pc_change.select(
            target_pc,
            total_stall.select(current_pc, current_pc + UInt(XLEN)(4))
        )
        
        prev_pc[0] = current_pc
        pc[0] = new_pc
        
        # 跳转等待标志
        jump_pending[0] = pc_change
        
        # 清除跳转缓冲区有效标志
        with Condition(~is_jump_pending):
            jump_buffer_valid[0] = UInt(1)(0)

        # ==================== ID/EX 寄存器更新 ===================
        nop_control = UInt(CONTROL_LEN)(0)
        
        # 关键逻辑：
        # - pc_change=1 或 jump_pending=1: 清空 ID/EX（控制冒险）
        # - data_hazard=1: 清空 ID/EX 并保持 IF/ID（数据冒险，插入气泡）
        # - mem_stall=1: 清空 ID/EX（结构冲突，IF 无法取新指令，IF/ID 被保持）
        # - 正常: 更新 ID/EX 为新解码的值
        
        # 需要清空 ID/EX 的情况
        # 注意：mem_stall 也需要清空 ID/EX，因为 IF/ID 被保持了，不能重复执行
        should_clear_ex = pc_change | is_jump_pending | data_hazard | mem_stall
        
        # 更新 ID/EX 寄存器
        # 注意：数据冒险和结构冲突时插入 NOP 而不是保持！
        # 保持的是 IF/ID 寄存器，不是 ID/EX
        id_ex_pc[0] = should_clear_ex.select(UInt(XLEN)(0), if_id_pc[0])
        id_ex_control[0] = should_clear_ex.select(nop_control, control_in)
        id_ex_immediate[0] = should_clear_ex.select(UInt(XLEN)(0), immediate)
        id_ex_rs1_idx[0] = should_clear_ex.select(UInt(5)(0), rs1)
        id_ex_rs2_idx[0] = should_clear_ex.select(UInt(5)(0), rs2)

        log("Arbiter: pc_change={}, target_pc={:08x}, data_hazard={}, mem_stall={}, jump_pending={}",
            pc_change, target_pc, data_hazard, mem_stall, is_jump_pending)
        log("Arbiter: PC={:08x}, new_PC={:08x}, if_id_valid={}, id_ex_valid={}, should_clear_ex={}",
            current_pc, new_pc, if_id_valid[0], id_ex_valid[0], should_clear_ex)

# ==================== 顶层CPU模块 ===================
class Driver(Module):
    """五级流水线RV32I CPU"""
    def __init__(self, program_file="test_program.txt"):
        super().__init__(ports={})

    @module.combinational
    def build(self, fetch_stage):
        fetch_stage.async_called()
        
def init_memory(program_file="test_program.txt", hex_output="unified_memory.hex"):
    """初始化内存内容 - 从指定文件加载程序并生成 SRAM 初始化文件
    
    冯诺依曼架构：程序和数据都存储在同一个 SRAM 中
    输入文件格式：每行一个十六进制数，支持 0x 前缀
    输出文件格式：每行8位十六进制数，无 0x 前缀（SRAM init_file 格式）
    """
    data = []
    
    try:
        # 从文件读取指令/数据
        with open(program_file, 'r') as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释行
                if not line or line.startswith('#'):
                    continue
                # 支持十六进制格式（带或不带0x前缀）
                if line.startswith('0x') or line.startswith('0X'):
                    value = int(line, 16)
                else:
                    try:
                        value = int(line, 16)  # 尝试作为十六进制解析
                    except ValueError:
                        value = int(line, 0)  # 自动检测进制
                data.append(value)
        
        print(f"Loaded {len(data)} words from {program_file}")
    
    except FileNotFoundError:
        print(f"Warning: Program file {program_file} not found. Using empty program.")
        data = [0x00000013]  # NOP
    except Exception as e:
        print(f"Error loading program from {program_file}: {e}")
        data = [0x00000013]  # NOP
    
    # 写入 hex 文件
    try:
        with open(hex_output, 'w') as f:
            for val in data:
                f.write(f"{val:08X}\n")
        print(f"Generated {hex_output} with {len(data)} words")
    except Exception as e:
        print(f"Error writing hex file: {e}")
    
    return hex_output     

def build_cpu(program_file="test_program.txt"):
    """构建RV32I CPU系统 - 冯诺依曼架构"""
    
    # 将程序文件转换为 hex 格式供 SRAM 初始化
    hex_file = init_memory(program_file, "unified_memory.hex")
    
    sys = SysBuilder('rv32i_cpu_von_neumann')
    with sys:
        # 创建单独的流水线寄存器，每个寄存器使用适合的宽度
        
        # IF/ID阶段寄存器
        if_id_pc = RegArray(UInt(XLEN), 1, initializer=[0])           # PC (32位)
        if_id_instruction = RegArray(UInt(XLEN), 1, initializer=[0x00000013])  # 指令 (32位)，初始化为 NOP
        if_id_valid = RegArray(UInt(1), 1, initializer=[1])            # 有效标志 (1位)

        # ID/EX阶段寄存器
        id_ex_pc = RegArray(UInt(XLEN), 1, initializer=[0])           # PC (32位)
        id_ex_control = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])  # 控制信号 (42位)
        id_ex_valid = RegArray(UInt(1), 1, initializer=[1])            # 有效标志 (1位)
        id_ex_rs1_idx = RegArray(UInt(5), 1, initializer=[0])         # rs1索引 (5位)
        id_ex_rs2_idx = RegArray(UInt(5), 1, initializer=[0])         # rs2索引 (5位)
        id_ex_immediate = RegArray(UInt(XLEN), 1, initializer=[0])    # 立即数 (32位)
        id_ex_need_rs1 = RegArray(UInt(1), 1, initializer=[0])        # 是否需要rs1 (1位)
        id_ex_need_rs2 = RegArray(UInt(1), 1, initializer=[0])        # 是否需要rs2 (1位)

        # EX/MEM阶段寄存器
        ex_mem_pc = RegArray(UInt(XLEN), 1, initializer=[0])           # PC (32位)
        ex_mem_control = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])  # 控制信号 (42位)
        ex_mem_valid = RegArray(UInt(1), 1, initializer=[1])            # 有效标志 (1位)
        ex_mem_result = RegArray(UInt(XLEN), 1, initializer=[0])       # ALU结果 (32位)
        ex_mem_data = RegArray(UInt(XLEN), 1, initializer=[0])          # 数据 (32位)

        # MEM/WB阶段寄存器
        mem_wb_control = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])  # 控制信号 (42位)
        mem_wb_valid = RegArray(UInt(1), 1, initializer=[1])            # 有效标志 (1位)
        mem_wb_mem_data = RegArray(UInt(XLEN), 1, initializer=[0])     # 内存数据 (32位)
        mem_wb_ex_result = RegArray(UInt(XLEN), 1, initializer=[0])     # EX阶段结果 (32位)

        # 创建寄存器文件
        reg_file = RegArray(UInt(XLEN), REG_COUNT, initializer=[0]*REG_COUNT)

        pc = RegArray(UInt(XLEN), 1, initializer=[0])
        # 上一周期的 PC，用于 IF/ID 寄存器（因为 SRAM 有一周期延迟）
        prev_pc = RegArray(UInt(XLEN), 1, initializer=[0])
        stall = RegArray(UInt(1), 1, initializer=[0])
        # 跳转等待标志：用于处理 SRAM 读取延迟
        # 跳转发生后需要等待一个周期让新 PC 地址的指令被取出
        jump_pending = RegArray(UInt(1), 1, initializer=[0])
        # 跳转指令缓冲区：用于保存跳转目标地址的指令
        jump_instruction_buffer = RegArray(UInt(XLEN), 1, initializer=[0x00000013])
        jump_buffer_valid = RegArray(UInt(1), 1, initializer=[0])
        
        # 数据冒险暂停计数器：用于处理多周期暂停
        # 与 EX 阶段的冒险需要暂停 2 个周期
        # 与 MEM 阶段的冒险需要暂停 1 个周期
        stall_counter = RegArray(UInt(2), 1, initializer=[0])
        
        # 创建统一的 SRAM（冯诺依曼架构核心）
        # 指令和数据都存储在这个 SRAM 中
        unified_sram = SRAM(width=XLEN, depth=65536, init_file=hex_file)
        
        # 创建各阶段模块
        hazard_unit = HazardUnit()
        memory_arbiter = MemoryArbiter()
        fetch_stage = FetchStage()
        decode_stage = DecodeStage()
        execute_stage = ExecuteStage()
        memory_stage = MemoryStage()
        writeback_stage = WriteBackStage()
        load_data_handler = LoadDataHandler()
        driver = Driver()

        # 按照流水线顺序构建模块
        writeback_signals = writeback_stage.build(mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, mem_wb_control, reg_file, unified_sram)
        memory_signals = memory_stage.build(ex_mem_valid, ex_mem_result, ex_mem_pc, ex_mem_data, ex_mem_control, mem_wb_control, mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, writeback_stage)
        execute_signals = execute_stage.build(id_ex_valid, id_ex_pc, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, id_ex_control, ex_mem_pc, ex_mem_control, ex_mem_valid, ex_mem_result, ex_mem_data, reg_file, memory_stage)
        decode_signals = decode_stage.build(if_id_valid, if_id_pc, if_id_instruction, execute_stage)
        fetch_signals = fetch_stage.build(pc, stall, if_id_pc, if_id_instruction, if_id_valid, decode_stage)
        
        # 构建 HazardUnit（数据冒险检测）- 简化参数列表
        hazard_signals = hazard_unit.build(stall_counter, id_ex_control, ex_mem_control, decode_signals, execute_signals)
        
        # 构建内存仲裁器（处理 IF 和 MEM 对统一 SRAM 的访问，以及流水线控制）
        memory_arbiter.build(unified_sram, pc, prev_pc, stall, jump_pending, if_id_valid, if_id_instruction, if_id_pc, id_ex_pc, id_ex_control, id_ex_valid, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, ex_mem_valid, mem_wb_valid, fetch_signals, execute_signals, memory_signals, hazard_signals, jump_instruction_buffer, jump_buffer_valid)
        
        # 构建 LoadDataHandler（处理 load 指令的数据通路）- 放在最后执行，确保在 MemoryArbiter 之后
        load_data_handler.build(mem_wb_mem_data, mem_wb_valid, mem_wb_control, unified_sram)
        
        # 构建Driver模块，处理PC更新
        driver.build(fetch_stage)
    
    return sys

def test_rv32i_cpu(program_file="test_program.txt"):
    """测试RV32I CPU"""
    sys = build_cpu(program_file)
    
    # 生成模拟器
    simulator_path, _ = elaborate(sys, verilog=False, sim_threshold=10000, resource_base='.')
    raw = utils.run_simulator(simulator_path)
    with open("result.out", 'w', encoding='utf-8') as f:
        print(raw, file=f)

if __name__ == "__main__":
    test_rv32i_cpu(program_file="test_program.txt")