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
CONTROL_LEN = 42 # 控制信号长度

# ==================== IF阶段：指令获取 ===================
class FetchStage(Module):
    """指令获取阶段(IF) - 冯诺依曼架构：从统一SRAM获取指令"""
    def __init__(self):
        super().__init__(ports={
        })
    
    @module.combinational
    def build(self, pc, if_fetch_ok, if_fetching_pc, decode_stage, unified_sram):
        """IF阶段：从SRAM获取指令
        
        Args:
            pc: 程序计数器（当前PC）
            if_fetch_ok: 上一周期是否成功获取指令
            if_fetching_pc: 上一周期发起取指请求的PC（SRAM返回的指令对应的PC）
            decode_stage: 解码阶段模块
            unified_sram: 统一内存SRAM
        """
        current_pc = pc[0]
        
        # 从SRAM的dout获取指令（上一周期发起的请求）
        # 这条指令对应的PC是 if_fetching_pc[0]
        instruction = unified_sram.dout[0]
        fetching_pc = if_fetching_pc[0]  # 这条指令实际对应的PC
        
        # 判断上一周期是否成功获取了指令
        fetch_success = if_fetch_ok[0]
        
        log("IF: PC={:08x}, Fetching_PC={:08x}, IF_FETCH_OK={}, Instruction={:08x}", 
            current_pc, fetching_pc, fetch_success, instruction)
        
        # FetchStage 只负责读取指令，不更新流水线寄存器
        # 所有流水线寄存器的更新由 UnifiedControlUnit 负责
        with Condition(fetch_success):
            log("IF: Got instruction from SRAM - Instruction_PC={:08x}, Instruction={:08x}", fetching_pc, instruction)

        decode_stage.async_called()
        
        # 返回信号：直接返回从SRAM读取的指令，让UnifiedControlUnit决定如何使用
        # concat语义：第一个参数在最高位，最后一个参数在最低位
        # 格式（从高到低）: 
        #   [96:65] instruction (32 bits)
        #   [64:64] fetch_success (1 bit)
        #   [63:32] current_pc (32 bits)
        #   [31:0]  fetching_pc (32 bits)
        return concat(
            instruction.bitcast(Bits(XLEN)),  # MSB: 从SRAM读取的指令
            fetch_success.bitcast(Bits(1)),   # 是否成功获取（上一周期发起的请求）
            current_pc.bitcast(Bits(XLEN)),   # 当前PC（用于发起新的取指请求）
            fetching_pc.bitcast(Bits(XLEN))   # LSB: 上一周期请求的PC（SRAM返回值对应的PC）
        )

# ==================== ID阶段：指令解码 ===================
class DecodeStage(Module):
    """指令解码阶段(ID)"""
    def __init__(self):
        super().__init__(ports={})
    
    @module.combinational
    def build(self, if_id_valid, if_id_pc, if_id_instruction, id_ex_pc, id_ex_control, id_ex_valid, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, id_ex_need_rs1, id_ex_need_rs2, reg_file, execute_stage):
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
        
        # 日志输出（只在 id_ex_valid=1 时输出，用于调试）
        with Condition(id_ex_valid[0]):
            log("ID: PC={}, Opcode={:07b}, RD={}, RS1={}, RS2={}, Immediate={}, Alu_op={}, Branch_op={}, Jump_op={}, Alu_src={}, Mem_read={}, Mem_write={}, Reg_write={}, Mem_to_reg={}, Control={:042b}",
                if_id_pc_in, opcode, rd, rs1, rs2, immediate, alu_op, branch_op, jump_op, alu_src, mem_read, mem_write, reg_write, mem_to_reg, control_signals)
        
        # 注意：id_ex_pc, id_ex_need_rs1, id_ex_need_rs2 的更新现在由 UnifiedControlUnit 负责
        # 这样可以确保在 data_hazard=1 时正确清空，在 data_hazard=0 时正确更新
        
        # rs1 = (~if_id_valid[0]).select(Bits(5)(0), rs1)
        # rs2 = (~if_id_valid[0]).select(Bits(5)(0), rs2)
        # immediate = (~if_id_valid[0]).select(UInt(XLEN)(0), immediate)
        # control_signals = (~if_id_valid[0]).select(Bits(CONTROL_LEN)(0), control_signals)

        execute_stage.async_called()

        # 返回 decode_signals：始终返回 IF/ID 中的当前值
        # UnifiedControlUnit 会根据当前周期的 data_hazard 来决定是否使用这些值更新 ID/EX
        # 不再使用 id_ex_valid[0] 来选择返回值，因为它是上一周期的值，有延迟问题
        decode_signals = concat(
            if_id_valid[0].select(need_rs2.bitcast(UInt(1)), UInt(1)(0)), 
            if_id_valid[0].select(need_rs1.bitcast(UInt(1)), UInt(1)(0)),
            if_id_valid[0].select(immediate, UInt(XLEN)(0)),
            if_id_valid[0].select(rs2.bitcast(UInt(5)), UInt(5)(0)),
            if_id_valid[0].select(rs1.bitcast(UInt(5)), UInt(5)(0)),
            if_id_valid[0].select(control_signals, Bits(CONTROL_LEN)(0)).bitcast(UInt(CONTROL_LEN)),
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
        

        with Condition(ex_mem_valid[0]):
            ex_mem_pc[0] = id_ex_valid[0].select(pc_in, UInt(XLEN)(0))
            ex_mem_control[0] = id_ex_valid[0].select(control_in, UInt(CONTROL_LEN)(0))
            # ex_mem_valid[0] = UInt(1)(1)
            ex_mem_result[0] = id_ex_valid[0].select(alu_result, UInt(XLEN)(0))
            ex_mem_data[0] = id_ex_valid[0].select(rs2_data, UInt(XLEN)(0))
            
            log("EX: PC={}, ALU_OP={:05b}, ALU_A={}, ALU_B={}, Result={:08x}, PC_Change={}, Target_PC={:08x}, Immediate={:08x}, ALU_SRC={}",
                pc_in, alu_op, alu_a, alu_b, alu_result, pc_change, target_pc, immediate_in, alu_src)
        
        memory_stage.async_called()

        # 注意：pc_change 和 target_pc 不能被 id_ex_valid 门控！
        # 因为当 stall 时，id_ex_valid 可能是 0，但 EX 阶段的指令仍然有效
        # 只有 control_in 需要被门控（用于数据冒险检测）
        execute_signals = concat(
            id_ex_valid[0].select(control_in.bitcast(Bits(CONTROL_LEN)), Bits(CONTROL_LEN)(0)),
            target_pc.bitcast(Bits(XLEN)),       # [31:1]  目标PC - 不门控
            pc_change.bitcast(Bits(1)),          # [0]     PC变化标志 - 不门控
        )

        return execute_signals

# ==================== MEM阶段：内存访问 ===================
class MemoryStage(Module):
    """内存访问阶段(MEM) - 冯诺依曼架构：不直接调用sram.build()，由仲裁器统一处理"""
    def __init__(self):
        super().__init__(ports={})
    
    @module.combinational
    def build(self, ex_mem_valid, ex_mem_result, ex_mem_pc, ex_mem_data, ex_mem_control, mem_wb_control, mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, writeback_stage, unified_sram):
        """MEM阶段：处理内存访问请求
        
        注意：在冯诺依曼架构中，MEM阶段不直接调用sram.build()
        而是由仲裁器根据MEM和IF的请求来决定SRAM访问
        """
        pc_in = ex_mem_pc[0]
        addr_in = ex_mem_result[0]
        data_in = ex_mem_data[0]
        control_in = ex_mem_control[0]
        
        # 解析控制信号
        mem_read = control_in[5:5]
        mem_write = control_in[6:6]
        store_type = control_in[22:23]  # 存储类型: 00=SB, 01=SH, 10=SW
        
        # 计算字地址和写数据
        word_addr = addr_in >> UInt(XLEN)(2)
        write_data = data_in

        with Condition(mem_wb_valid[0]):
            mem_wb_control[0] = ex_mem_valid[0].select(control_in, UInt(CONTROL_LEN)(0))
            mem_wb_ex_result[0] = ex_mem_valid[0].select(ex_mem_result[0], UInt(XLEN)(0))
            
            log("MEM: PC={}, Addr={:08x}, Read={}, Write={}, data_in={}",
                pc_in, addr_in, mem_read, mem_write, data_in)

        writeback_stage.async_called()

        # 返回MEM阶段的请求信号给仲裁器
        # 格式: [内存读使能, 内存写使能, 字地址, 写数据, 控制信号, 有效标志]
        mem_request = concat(
            control_in.bitcast(Bits(CONTROL_LEN)),  # 控制信号
            write_data.bitcast(Bits(XLEN)),         # 写数据
            word_addr.bitcast(Bits(XLEN)),          # 字地址
            mem_write.bitcast(Bits(1)),             # 写使能
            mem_read.bitcast(Bits(1)),              # 读使能
            ex_mem_valid[0].bitcast(Bits(1))        # MEM阶段有效标志
        )
        return mem_request

# ==================== WB阶段：写回 ===================
class WriteBackStage(Module):
    """写回阶段(WB) - 冯诺依曼架构：从统一SRAM的dout获取数据"""
    def __init__(self):
        super().__init__(ports={})
    
    @module.combinational
    def build(self, mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, mem_wb_control, reg_file, unified_sram):
        """WB阶段：写回寄存器
        
        在冯诺依曼架构中，内存读取的数据从统一SRAM的dout获取
        """
        # 从统一SRAM获取内存数据（上一周期MEM阶段发起的读请求）
        mem_data_in = unified_sram.dout[0]
        ex_result_in = mem_wb_ex_result[0]
        control_in = mem_wb_control[0]
        
        # 解析控制信号
        reg_write = control_in[7:7]
        mem_to_reg = control_in[8:8]
        wb_rd = control_in[25:29]
            
        # 选择写回数据：如果是load指令则从内存，否则从EX结果
        wb_data = mem_to_reg.select(mem_data_in, ex_result_in)
            
        # 如果指令无效，直接返回
        with Condition(mem_wb_valid[0]):
            with Condition(reg_write):
                reg_file[wb_rd] = wb_data
            log("WB: Write_Data={}, RD={}, WE={}, mem_to_reg={}",
                wb_data, wb_rd, reg_write, mem_to_reg)

        writeback_signals = control_in.bitcast(Bits(CONTROL_LEN))
        return writeback_signals


# ==================== 统一控制单元：冒险检测 + SRAM仲裁 ===================
class UnifiedControlUnit(Downstream):
    """统一控制单元 - 冯诺依曼架构的核心
    
    整合了冒险检测和SRAM仲裁功能：
    1. 检测数据冒险
    2. 处理PC跳转
    3. 仲裁IF和MEM对统一SRAM的访问
    4. 更新流水线寄存器
    """
    def __init__(self):
        super().__init__()

    @downstream.combinational
    def build(self, pc, stall, if_id_valid, if_id_instruction, if_id_pc, id_ex_control, id_ex_valid,
              id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, id_ex_pc, id_ex_need_rs1, id_ex_need_rs2,
              ex_mem_valid, mem_wb_valid,
              if_fetch_ok, if_fetching_pc, fetch_signals, decode_signals, execute_signals, mem_request, 
              writeback_signals, unified_sram,
              pending_valid, pending_pc, pending_control, pending_rs1_idx, pending_rs2_idx,
              pending_immediate, pending_need_rs1, pending_need_rs2,
              if_pending_valid, if_pending_instruction, if_pending_pc):
        """统一控制单元的核心逻辑"""
        
        # ==================== 解析各阶段信号 ====================
        
        # 解析EX阶段的信号
        execute_signals = execute_signals.optional(Bits(XLEN + 1 + CONTROL_LEN)(0))
        pc_change = execute_signals[0:0].bitcast(UInt(1))
        target_pc = execute_signals[1:XLEN].bitcast(UInt(XLEN))
        
        # 解析ID阶段的信号
        decode_signals = decode_signals.optional(Bits(2 + CONTROL_LEN + 5 + 5 + XLEN)(0))
        immediate = decode_signals[CONTROL_LEN + 5 + 5:CONTROL_LEN + 5 + 5 + XLEN - 1].bitcast(UInt(XLEN))
        rs1 = decode_signals[CONTROL_LEN:CONTROL_LEN + 5 - 1].bitcast(UInt(5))
        rs2 = decode_signals[CONTROL_LEN + 5:CONTROL_LEN + 5 + 5 - 1].bitcast(UInt(5))
        control_in = decode_signals[0:CONTROL_LEN - 1].bitcast(UInt(CONTROL_LEN))
        needs_rs1 = decode_signals[CONTROL_LEN + 5 + 5 + XLEN:CONTROL_LEN + 5 + 5 + XLEN].bitcast(UInt(1))
        needs_rs2 = decode_signals[CONTROL_LEN + 5 + 5 + XLEN + 1:CONTROL_LEN + 5 + 5 + XLEN + 1].bitcast(UInt(1))
        
        # 解析IF阶段的信号
        # fetch_signals = concat(instruction, fetch_success, current_pc, fetching_pc)
        # concat语义：第一个参数在最高位，最后一个参数在最低位
        # 所以实际布局（从高到低）：
        # [96:65] = instruction (32 bits)
        # [64:64] = fetch_success (1 bit)
        # [63:32] = current_pc (32 bits)
        # [31:0]  = fetching_pc (32 bits)
        fetch_signals = fetch_signals.optional(Bits(XLEN + 1 + XLEN + XLEN)(0))
        if_fetching_pc_val = fetch_signals[0:XLEN - 1].bitcast(UInt(XLEN))           # [31:0]
        if_current_pc = fetch_signals[XLEN:XLEN + XLEN - 1].bitcast(UInt(XLEN))       # [63:32]
        if_fetch_success = fetch_signals[XLEN + XLEN:XLEN + XLEN].bitcast(UInt(1))    # [64]
        if_instruction = fetch_signals[XLEN + XLEN + 1:XLEN + XLEN + 1 + XLEN - 1].bitcast(UInt(XLEN))  # [96:65]
        
        # 解析MEM阶段的请求信号
        # mem_request格式: [控制信号(42), 写数据(32), 字地址(32), 写使能(1), 读使能(1), 有效标志(1)]
        mem_request = mem_request.optional(Bits(CONTROL_LEN + XLEN + XLEN + 1 + 1 + 1)(0))
        mem_valid = mem_request[0:0].bitcast(UInt(1))
        mem_read = mem_request[1:1].bitcast(UInt(1))
        mem_write = mem_request[2:2].bitcast(UInt(1))
        mem_addr = mem_request[3:3 + XLEN - 1].bitcast(UInt(XLEN))
        mem_wdata = mem_request[3 + XLEN:3 + XLEN + XLEN - 1].bitcast(UInt(XLEN))
        mem_control = mem_request[3 + XLEN + XLEN:3 + XLEN + XLEN + CONTROL_LEN - 1].bitcast(UInt(CONTROL_LEN))
        
        writeback_signals = writeback_signals.optional(Bits(CONTROL_LEN)(0))
        
        # ==================== 数据冒险检测 ====================
        
        # 计算EX->MEM的控制信号（用于数据冒险检测）
        memory_control = execute_signals[XLEN + 1:XLEN + 1 + CONTROL_LEN - 1].bitcast(UInt(CONTROL_LEN))
        memory_control = id_ex_valid[0].select(memory_control, UInt(CONTROL_LEN)(0))
        rd_mem = memory_control[25:29]
        reg_write_mem = memory_control[7:7]
        
        # 计算WB阶段的控制信号
        wb_control = mem_control  # 使用MEM返回的控制信号
        wb_control = ex_mem_valid[0].select(wb_control, UInt(CONTROL_LEN)(0))
        rd_wb = wb_control[25:29]
        reg_write_wb = wb_control[7:7]
        
        # 数据冒险检测
        data_hazard_ex = UInt(1)(0)
        data_hazard_wb = UInt(1)(0)
        
        data_hazard_ex = (reg_write_mem & ((needs_rs1 & (rs1 == rd_mem)) | (needs_rs2 & (rs2 == rd_mem)))).select(UInt(1)(1), data_hazard_ex)
        data_hazard_wb = (reg_write_wb & ((needs_rs1 & (rs1 == rd_wb)) | (needs_rs2 & (rs2 == rd_wb)))).select(UInt(1)(1), data_hazard_wb)
        
        # 综合数据冒险信号
        data_hazard = ((data_hazard_ex | data_hazard_wb) & ~pc_change)
        
        # ==================== SRAM仲裁逻辑 ====================
        
        # MEM阶段是否有内存访问请求
        mem_has_request = mem_valid & (mem_read | mem_write)
        
        # 取指阻塞：当上一周期IF没有成功获取指令时（但这不应该阻止发起新的取指请求）
        fetch_stall = ~if_fetch_ok[0]
        
        # 结构冲突：当MEM阶段占用SRAM时，IF无法取指
        # 此时需要暂停PC增加，否则会跳过一条指令
        structural_hazard = mem_has_request
        
        # 总的阻塞信号：数据冒险 或 取指阻塞 或 结构冲突（用于控制流水线其他部分）
        total_stall = (data_hazard | fetch_stall | structural_hazard)
        
        # 计算下一周期的PC值
        # 如果PC跳转，使用跳转目标
        # 如果stall，保持当前PC
        # 否则增加4
        next_pc = pc_change.select(target_pc, total_stall.select(pc[0], pc[0] + UInt(XLEN)(4)))
        
        # 计算IF阶段的取指地址（字地址）
        # 关键修复：使用 next_pc 作为下一周期要取的指令地址
        # 因为 SRAM.build() 发起的请求在下一周期返回，而下一周期的 PC 就是 next_pc
        # 当有 stall 时，next_pc = pc[0]，所以会重新请求当前 PC
        if_word_addr = (next_pc >> UInt(XLEN)(2)).bitcast(UInt(XLEN))
        
        # 仲裁逻辑：MEM优先 - 确保类型一致
        mem_addr_uint = mem_addr.bitcast(UInt(XLEN))
        final_addr = mem_has_request.select(mem_addr_uint, if_word_addr)
        final_we = mem_has_request.select(mem_write, UInt(1)(0))  # IF不写
        # IF读取请求：
        # 1. 如果没有MEM请求，并且（没有total_stall 或者 有pc_change 或者 是fetch_stall导致的stall）
        # 关键：fetch_stall时也应该发起读取（为了获取指令），只是不推进流水线
        # 简化逻辑：如果没有MEM请求，并且（没有data_hazard导致的stall 或 有pc_change）
        if_should_read = ((~data_hazard) | pc_change).bitcast(UInt(1))
        final_re = mem_has_request.select(mem_read, if_should_read)
        mem_wdata_uint = mem_wdata.bitcast(UInt(XLEN))
        final_wdata = mem_has_request.select(mem_wdata_uint, UInt(XLEN)(0))
        
        # 调用SRAM.build() - 这是程序中唯一的sram.build()调用
        unified_sram.build(we=final_we, re=final_re, addr=final_addr, wdata=final_wdata)
        
        # 计算下一周期IF是否成功获取指令
        if_will_succeed = (~mem_has_request) & final_re
        
        # ==================== 更新寄存器 ====================
        
        # 更新冯诺依曼控制寄存器
        # if_fetch_ok: 这个周期是否成功发起了IF读取
        if_fetch_ok[0] = if_will_succeed
        # if_fetching_pc: 记录这次发起读取请求的PC（如果没有发起读取，保持原值）
        # 这个值会在下一周期被用来标记SRAM返回的指令对应的PC
        # 关键修复：使用 next_pc 而不是 pc[0]，因为我们请求的是 next_pc 处的指令
        if_fetching_pc[0] = if_will_succeed.select(next_pc, if_fetching_pc[0])
        
        # 更新流水线寄存器有效标志
        # id_ex_valid: 只有数据冒险才阻止 ID->EX 的传递
        # fetch_stall 只影响 IF->ID，不应该阻止 ID 阶段的指令进入 EX
        id_ex_valid[0] = (~data_hazard)
        # if_id_valid: 表示 IF/ID 中是否有有效指令
        # 当 IF/ID 被更新为新指令时为 1，当被清空为 NOP 时为 0
        # 在 stall 期间保持当前值（因为 IF/ID 保持不变）
        # 具体逻辑在下面的 IF/ID 更新部分处理
        ex_mem_valid[0] = UInt(1)(1)
        mem_wb_valid[0] = UInt(1)(1)
        stall[0] = total_stall
        
        nop_control = UInt(CONTROL_LEN)(0)

        # 更新PC（只有在成功发起取指请求后才增加）
        # 如果PC跳转，使用跳转目标；否则如果有stall保持不变；否则增加4
        pc[0] = next_pc
        
        # ==================== 更新IF/ID寄存器（使用 if_pending 缓冲区） ====================
        # 
        # 问题：当 data_hazard=1 时，IF 可能已经取到了新指令（if_fetch_success=1），
        #       但因为 stall 无法更新到 IF/ID。这个指令会在下一周期丢失。
        # 
        # 解决方案：使用 if_pending_* 缓冲区保存取到但未能进入 IF/ID 的指令
        # 
        # 逻辑：
        # 1. pc_change=1 -> 插入NOP，清空 if_pending
        # 2. data_hazard=1 且 if_fetch_success=1 且 if_pending_valid=0 -> 保存到 if_pending
        # 3. ~data_hazard 且 if_pending_valid=1 -> 从 if_pending 恢复到 IF/ID
        # 4. ~data_hazard 且 if_pending_valid=0 且 if_fetch_success=1 -> 正常更新 IF/ID
        # 5. ~data_hazard 且 fetch_stall=1 且 if_pending_valid=0 -> 插入NOP
        
        # 读取 if_pending 状态（上一周期的值）
        has_if_pending = if_pending_valid[0]
        
        # 计算更新条件
        should_save_to_if_pending = (~pc_change) & data_hazard & if_fetch_success & (~has_if_pending)
        should_restore_from_if_pending = (~pc_change) & (~data_hazard) & has_if_pending
        should_update_if_id_normal = (~pc_change) & (~data_hazard) & (~has_if_pending) & if_fetch_success
        # 只有在没有 pending 指令且没有新取到指令时才插入 NOP
        should_insert_nop_if_id = (~pc_change) & (~data_hazard) & (~has_if_pending) & (~if_fetch_success)
        
        log("IF_ID_UPDATE: pc_change={}, data_hazard={}, if_fetch_success={}, has_if_pending={}, if_instruction={:08x}, if_fetching_pc={:08x}",
            pc_change, data_hazard, if_fetch_success, has_if_pending, if_instruction, if_fetching_pc_val)
        log("IF_ID_UPDATE: should_save={}, should_restore={}, should_update={}, should_nop={}",
            should_save_to_if_pending, should_restore_from_if_pending, should_update_if_id_normal, should_insert_nop_if_id)
        
        # 更新 if_pending 缓冲区
        with Condition(pc_change):
            # 控制冒险：清空 if_pending
            if_pending_valid[0] = UInt(1)(0)
        with Condition(should_save_to_if_pending):
            # 保存取到的指令到 if_pending
            if_pending_valid[0] = UInt(1)(1)
            if_pending_instruction[0] = if_instruction
            if_pending_pc[0] = if_fetching_pc_val
            log("IF_PENDING: Saved instruction={:08x}, pc={:08x}", if_instruction, if_fetching_pc_val)
        with Condition(should_restore_from_if_pending):
            # 从 if_pending 恢复后清空
            if_pending_valid[0] = UInt(1)(0)
            log("IF_PENDING: Restored instruction={:08x}, pc={:08x}", if_pending_instruction[0], if_pending_pc[0])
        # 其他情况：保持不变
        
        # 更新 IF/ID 寄存器
        with Condition(pc_change):
            # 控制冒险：插入 NOP
            if_id_instruction[0] = UInt(XLEN)(0x00000013)  # NOP
            if_id_pc[0] = UInt(XLEN)(0)
            if_id_valid[0] = UInt(1)(0)
        with Condition(should_restore_from_if_pending):
            # 从 if_pending 恢复
            if_id_instruction[0] = if_pending_instruction[0]
            if_id_pc[0] = if_pending_pc[0]
            if_id_valid[0] = UInt(1)(1)
        with Condition(should_update_if_id_normal):
            # 正常更新
            if_id_instruction[0] = if_instruction
            if_id_pc[0] = if_fetching_pc_val
            if_id_valid[0] = UInt(1)(1)
        with Condition(should_insert_nop_if_id):
            # 没有指令可用：插入 NOP
            if_id_instruction[0] = UInt(XLEN)(0x00000013)  # NOP
            if_id_pc[0] = UInt(XLEN)(0)
            if_id_valid[0] = UInt(1)(0)
        # data_hazard=1 时，IF/ID 保持不变（不需要显式设置）
        
        # ==================== 更新ID/EX寄存器（使用等待缓冲区） ====================
        # 
        # 核心思路：使用 pending_* 缓冲区来保存因数据冒险而等待的指令
        # 
        # 时序问题：寄存器写入有一周期延迟
        #   - 周期N: 检测到 data_hazard=1，设置 pending_valid[0]=1
        #   - 周期N+1: pending_valid[0] 才能被读到为 1
        # 
        # 逻辑流程：
        # 1. pc_change=1 时：清空 ID/EX 和 pending 缓冲区
        # 2. data_hazard=1 时：
        #    - 如果 pending_valid[0]=0（读到的是上周期的值，意味着上周期没保存），
        #      将当前 IF/ID 的指令保存到 pending
        #    - 如果 pending_valid[0]=1（上周期已经保存了），保持 pending 不变
        #    - ID/EX 插入气泡
        # 3. ~data_hazard 且 ~pc_change 时：
        #    - 如果 pending_valid[0]=1（有等待的指令），从 pending 恢复到 ID/EX
        #    - 否则从 decode_signals 正常获取
        
        need_bubble = (pc_change | data_hazard)
        
        # 从 pending 缓冲区读取（上一周期写入的值）
        has_pending = pending_valid[0]
        
        # 决定写入 ID/EX 的数据源
        # 当冒险解除时，如果有 pending 指令，使用 pending；否则使用当前解码结果
        use_pending = (~need_bubble) & has_pending
        
        # 计算实际写入 ID/EX 的值
        actual_control = use_pending.select(pending_control[0], control_in)
        actual_immediate = use_pending.select(pending_immediate[0], immediate)
        actual_rs1 = use_pending.select(pending_rs1_idx[0], rs1)
        actual_rs2 = use_pending.select(pending_rs2_idx[0], rs2)
        actual_pc = use_pending.select(pending_pc[0], if_id_pc[0])
        actual_need_rs1 = use_pending.select(pending_need_rs1[0], needs_rs1)
        actual_need_rs2 = use_pending.select(pending_need_rs2[0], needs_rs2)
        
        log("PENDING: has_pending={}, use_pending={}, need_bubble={}", has_pending, use_pending, need_bubble)
        
        # 更新 pending 缓冲区
        # 情况1: pc_change=1 -> 清空 pending（控制冒险，丢弃所有等待的指令）
        # 情况2: data_hazard=1 且 pending_valid=0 -> 保存当前 IF/ID 到 pending
        # 情况3: data_hazard=1 且 pending_valid=1 -> 保持 pending 不变
        # 情况4: 使用了 pending (use_pending=1) -> 清空 pending
        # 情况5: 正常传递 -> 保持 pending 不变（或已经为空）
        
        with Condition(pc_change):
            # 控制冒险：清空 pending
            pending_valid[0] = UInt(1)(0)
        with Condition((~pc_change) & data_hazard & (~has_pending)):
            # 数据冒险，且 pending 为空：保存当前 IF/ID 的指令
            pending_valid[0] = UInt(1)(1)
            pending_control[0] = control_in
            pending_immediate[0] = immediate
            pending_rs1_idx[0] = rs1
            pending_rs2_idx[0] = rs2
            pending_pc[0] = if_id_pc[0]
            pending_need_rs1[0] = needs_rs1
            pending_need_rs2[0] = needs_rs2
            log("PENDING: Saved instruction to pending buffer, PC={:08x}, control={:042b}", if_id_pc[0], control_in)
        with Condition(use_pending):
            # 使用了 pending 中的指令：清空 pending
            pending_valid[0] = UInt(1)(0)
            log("PENDING: Restored instruction from pending buffer, PC={:08x}", pending_pc[0])
        # 其他情况: pending 保持不变
        
        # 更新 ID/EX 寄存器
        with Condition(need_bubble):
            # 插入气泡
            id_ex_control[0] = nop_control
            id_ex_immediate[0] = UInt(XLEN)(0)
            id_ex_rs1_idx[0] = UInt(5)(0)
            id_ex_rs2_idx[0] = UInt(5)(0)
            id_ex_pc[0] = UInt(XLEN)(0)
            id_ex_need_rs1[0] = UInt(1)(0)
            id_ex_need_rs2[0] = UInt(1)(0)
        with Condition(~need_bubble):
            # 正常传递指令（可能来自 pending 或当前解码）
            id_ex_control[0] = actual_control
            id_ex_immediate[0] = actual_immediate
            id_ex_rs1_idx[0] = actual_rs1
            id_ex_rs2_idx[0] = actual_rs2
            id_ex_pc[0] = actual_pc
            id_ex_need_rs1[0] = actual_need_rs1.bitcast(Bits(1))
            id_ex_need_rs2[0] = actual_need_rs2.bitcast(Bits(1))

        log("RD_MEM={}, REG_WRITE_MEM={}, RD_WB={}, REG_WRITE_WB={}",
            rd_mem, reg_write_mem, rd_wb, reg_write_wb)
        log("CONTROL: Data_Hazard={}, Fetch_Stall={}, Struct_Hazard={}, Total_Stall={}, PC_Change={}, Target_PC={:08x}, Next_PC={:08x}",
            data_hazard, fetch_stall, structural_hazard, total_stall, pc_change, target_pc, next_pc)
        log("ARBITER: MEM_has_req={}, MEM_read={}, MEM_write={}, MEM_addr={:08x}, IF_addr={:08x}, IF_will_succeed={}",
            mem_has_request, mem_read, mem_write, mem_addr, if_word_addr, if_will_succeed)


# ==================== 顶层CPU模块 ===================
class Driver(Module):
    """五级流水线RV32I CPU"""
    def __init__(self, program_file="test_program.txt"):
        super().__init__(ports={})

    @module.combinational
    def build(self, fetch_stage):
        fetch_stage.async_called()
        
def init_memory(program_file="test_program.txt"):
    """初始化统一内存内容 - 冯诺依曼架构
    
    从test_program.txt加载程序和数据到统一内存
    格式：每行一个十六进制数，支持带或不带0x前缀
    """
    memory_content = []
    
    try:
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
                    value = int(line, 16)  # 默认为16进制
                memory_content.append(value)
        
        print(f"Von Neumann: Loaded {len(memory_content)} words from {program_file}")
    
    except FileNotFoundError:
        print(f"Warning: Program file {program_file} not found. Using empty memory.")
    except Exception as e:
        print(f"Error loading from {program_file}: {e}")
    
    return memory_content


def create_hex_file_for_sram(memory_content, hex_file="unified_memory.hex"):
    """创建SRAM初始化用的hex文件（不带0x前缀）"""
    with open(hex_file, 'w') as f:
        for value in memory_content:
            f.write(f"{value:08x}\n")
    print(f"Created hex file: {hex_file} with {len(memory_content)} entries")
    return hex_file


def build_cpu(program_file="test_program.txt"):
    """构建冯诺依曼架构的RV32I CPU系统"""
    sys = SysBuilder('rv32i_cpu_von_neumann')
    with sys:
        # ==================== 流水线寄存器 ====================
        
        # IF/ID阶段寄存器
        if_id_pc = RegArray(UInt(XLEN), 1, initializer=[0])
        if_id_instruction = RegArray(UInt(XLEN), 1, initializer=[0])
        if_id_valid = RegArray(UInt(1), 1, initializer=[1])

        # ID/EX阶段寄存器
        id_ex_pc = RegArray(UInt(XLEN), 1, initializer=[0])
        id_ex_control = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])
        id_ex_valid = RegArray(UInt(1), 1, initializer=[1])
        id_ex_rs1_idx = RegArray(UInt(5), 1, initializer=[0])
        id_ex_rs2_idx = RegArray(UInt(5), 1, initializer=[0])
        id_ex_immediate = RegArray(UInt(XLEN), 1, initializer=[0])
        id_ex_need_rs1 = RegArray(UInt(1), 1, initializer=[0])
        id_ex_need_rs2 = RegArray(UInt(1), 1, initializer=[0])

        # 等待缓冲寄存器：当发生数据冒险时保存当前指令，等冒险解除后恢复
        pending_valid = RegArray(UInt(1), 1, initializer=[0])  # 缓冲区是否有等待的指令
        pending_pc = RegArray(UInt(XLEN), 1, initializer=[0])
        pending_control = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])
        pending_rs1_idx = RegArray(UInt(5), 1, initializer=[0])
        pending_rs2_idx = RegArray(UInt(5), 1, initializer=[0])
        pending_immediate = RegArray(UInt(XLEN), 1, initializer=[0])
        pending_need_rs1 = RegArray(UInt(1), 1, initializer=[0])
        pending_need_rs2 = RegArray(UInt(1), 1, initializer=[0])

        # EX/MEM阶段寄存器
        ex_mem_pc = RegArray(UInt(XLEN), 1, initializer=[0])
        ex_mem_control = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])
        ex_mem_valid = RegArray(UInt(1), 1, initializer=[1])
        ex_mem_result = RegArray(UInt(XLEN), 1, initializer=[0])
        ex_mem_data = RegArray(UInt(XLEN), 1, initializer=[0])

        # MEM/WB阶段寄存器
        mem_wb_control = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])
        mem_wb_valid = RegArray(UInt(1), 1, initializer=[1])
        mem_wb_mem_data = RegArray(UInt(XLEN), 1, initializer=[0])
        mem_wb_ex_result = RegArray(UInt(XLEN), 1, initializer=[0])

        # ==================== 冯诺依曼架构：统一内存 ====================
        # 从test_program.txt加载内容，创建临时hex文件供SRAM使用
        memory_content = init_memory(program_file)
        hex_file = create_hex_file_for_sram(memory_content, "unified_memory.hex")
        
        # 创建统一的SRAM（存储指令和数据）
        unified_sram = SRAM(width=XLEN, depth=65536, init_file=hex_file)
        unified_sram.name = 'unified_memory'
        
        # 创建寄存器文件
        reg_file = RegArray(UInt(XLEN), REG_COUNT, initializer=[0]*REG_COUNT)

        # 控制寄存器
        pc = RegArray(UInt(XLEN), 1, initializer=[0])
        stall = RegArray(UInt(1), 1, initializer=[0])
        
        # 冯诺依曼特有：IF阶段控制寄存器
        # if_fetch_ok: 上一周期是否成功获取指令
        # 初始化为0，因为第一个周期还没有发起过请求
        if_fetch_ok = RegArray(UInt(1), 1, initializer=[0])
        # if_fetching_pc: 上一周期发起取指请求的PC（SRAM返回值对应的PC）
        # 初始化为0
        if_fetching_pc = RegArray(UInt(XLEN), 1, initializer=[0])
        
        # IF指令缓冲区：当取到指令但因 stall 无法进入 IF/ID 时保存
        # 用于解决取指与 data_hazard 同时发生时指令丢失的问题
        if_pending_valid = RegArray(UInt(1), 1, initializer=[0])
        if_pending_instruction = RegArray(UInt(XLEN), 1, initializer=[0])
        if_pending_pc = RegArray(UInt(XLEN), 1, initializer=[0])

        # ==================== 创建模块 ====================
        fetch_stage = FetchStage()
        decode_stage = DecodeStage()
        execute_stage = ExecuteStage()
        memory_stage = MemoryStage()
        writeback_stage = WriteBackStage()
        unified_control = UnifiedControlUnit()
        driver = Driver()

        # ==================== 构建流水线 ====================
        # 按照流水线顺序构建模块
        writeback_signals = writeback_stage.build(
            mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, mem_wb_control, 
            reg_file, unified_sram
        )
        
        mem_request = memory_stage.build(
            ex_mem_valid, ex_mem_result, ex_mem_pc, ex_mem_data, ex_mem_control, 
            mem_wb_control, mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, 
            writeback_stage, unified_sram
        )
        
        execute_signals = execute_stage.build(
            id_ex_valid, id_ex_pc, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, 
            id_ex_control, ex_mem_pc, ex_mem_control, ex_mem_valid, ex_mem_result, 
            ex_mem_data, reg_file, memory_stage
        )
        
        decode_signals = decode_stage.build(
            if_id_valid, if_id_pc, if_id_instruction, id_ex_pc, id_ex_control, 
            id_ex_valid, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, 
            id_ex_need_rs1, id_ex_need_rs2, reg_file, execute_stage
        )
        
        fetch_signals = fetch_stage.build(
            pc, if_fetch_ok, if_fetching_pc, decode_stage, unified_sram
        )
        
        # 统一控制单元：整合冒险检测和SRAM仲裁（程序中唯一的sram.build()调用点）
        unified_control.build(
            pc, stall, if_id_valid, if_id_instruction, if_id_pc, id_ex_control, id_ex_valid,
            id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, id_ex_pc, id_ex_need_rs1, id_ex_need_rs2,
            ex_mem_valid, mem_wb_valid,
            if_fetch_ok, if_fetching_pc, fetch_signals, decode_signals, execute_signals, mem_request, 
            writeback_signals, unified_sram,
            pending_valid, pending_pc, pending_control, pending_rs1_idx, pending_rs2_idx,
            pending_immediate, pending_need_rs1, pending_need_rs2,
            if_pending_valid, if_pending_instruction, if_pending_pc
        )
        
        # 构建Driver模块
        driver.build(fetch_stage)
    
    return sys


def test_rv32i_cpu(program_file="test_program.txt"):
    """测试冯诺依曼架构的RV32I CPU"""
    sys = build_cpu(program_file)
    
    # 生成模拟器
    simulator_path, _ = elaborate(sys, verilog=False, sim_threshold=10000, resource_base='.')
    raw = utils.run_simulator(simulator_path)
    with open("result.out", 'w', encoding='utf-8') as f:
        print(raw, file=f)


if __name__ == "__main__":
    test_rv32i_cpu(program_file="test_program.txt")