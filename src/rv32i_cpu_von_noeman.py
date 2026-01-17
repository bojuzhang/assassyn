#!/usr/bin/env python3
"""
五级流水线RV32IM CPU实现
使用Assassyn语言实现完整的RISC-V 32位基础指令集处理器
支持BTB + 2-bit饱和计数器动态分支预测
支持RV32IM乘法除法扩展 (mul, mulh, mulhsu, mulhu, div, divu, rem, remu)
使用Wallace Tree 3周期乘法器
使用Radix-4 SRT 18周期除法器
"""

from assassyn.frontend import *
from assassyn.backend import elaborate
from assassyn import utils
from assassyn.ir.memory.sram import SRAM
from assassyn.ir.module import downstream, Downstream

# ==================== 常量定义 ===================
XLEN = 32  # RISC-V XLEN
REG_COUNT = 32  # 通用寄存器数量
CONTROL_LEN = 48 # 控制信号长度 (42 + 3位mul_op + 3位div_op)
BTB_SIZE = 64  # BTB表大小
BTB_INDEX_BITS = 6  # BTB索引位数 (log2(64)=6)
PREDICTION_INFO_LEN = 34  # 预测信息长度: [0]: btb_hit, [1]: predict_taken, [2:33]: predicted_pc
PREDICTION_RESULT_LEN = 68  # 预测结果长度

# ==================== M扩展乘法操作码 ===================
# mul_op 编码 (3位):
# 000 - 非乘法指令
# 001 - MUL    (signed × signed, low 32 bits)
# 010 - MULH   (signed × signed, high 32 bits)
# 011 - MULHSU (signed × unsigned, high 32 bits)
# 100 - MULHU  (unsigned × unsigned, high 32 bits)
MUL_OP_NONE   = 0b000
MUL_OP_MUL    = 0b001
MUL_OP_MULH   = 0b010
MUL_OP_MULHSU = 0b011
MUL_OP_MULHU  = 0b100

# ==================== M扩展除法操作码 ===================
# div_op 编码 (3位):
# 000 - 非除法指令
# 001 - DIV    (signed division)
# 010 - DIVU   (unsigned division)
# 011 - REM    (signed remainder)
# 100 - REMU   (unsigned remainder)
DIV_OP_NONE = 0b000
DIV_OP_DIV  = 0b001
DIV_OP_DIVU = 0b010
DIV_OP_REM  = 0b011
DIV_OP_REMU = 0b100

# ==================== Wallace Tree 乘法器说明 ====================
# Wallace Tree 乘法器集成在 ExecuteStage 中实现
# 
# 架构设计:
# - 输入: 32位 × 32位
# - 输出: 64位 (根据指令选择高32位或低32位)
# - 延迟: 3周期
#
# 支持的指令:
# - MUL:    signed × signed, 返回低32位
# - MULH:   signed × signed, 返回高32位  
# - MULHSU: signed × unsigned, 返回高32位
# - MULHU:  unsigned × unsigned, 返回高32位
#
# Wallace Tree压缩使用 Carry-Save Adder (CSA):
# - 3个操作数 → 2个操作数 (sum + carry)
# - sum = a ^ b ^ c
# - carry = ((a & b) | (b & c) | (a & c)) << 1
# - 只有最终阶段使用普通加法器
#
# 3周期流水线:
# - Cycle 1: 符号扩展 + 部分积生成 + CSA压缩 (32→22→15→10)
# - Cycle 2: 继续CSA压缩 (10→7→5→4→3→2)
# - Cycle 3: 最终加法 + 结果选择

# ==================== Radix-4 SRT 除法器说明 ====================
# Radix-4 SRT 除法器集成在 ExecuteStage 中实现
#
# 架构设计:
# - 输入: 32位被除数, 32位除数
# - 输出: 32位商或余数
# - 延迟: 18周期
#
# 支持的指令:
# - DIV:  有符号除法
# - DIVU: 无符号除法
# - REM:  有符号取余
# - REMU: 无符号取余
#
# Radix-4 SRT 算法:
# - 商数字集合: q ∈ {-2, -1, 0, +1, +2}
# - 递归关系: P_{i+1} = 4 * P_i - q_i * D
# - 每次迭代产生 2 位商
# - 16 次迭代产生 32 位商
#
# 18周期流水线:
# - Cycle 1:  初始化 (保存操作数, 处理符号, 检查除零)
# - Cycle 2-17: 16次迭代 (商数字选择, 余数更新, 商累积)
# - Cycle 18: 最终修正 (冗余商转换, 符号修正, 结果选择)
#
# 状态机:
# - IDLE: 空闲, 等待新的除法指令
# - INIT: 初始化
# - ITERATE: 迭代
# - FINAL_CORRECTION: 最终修正
# - DONE: 完成

class FetchStage(Module):
    """指令获取阶段(IF) - 冯诺依曼架构，使用统一SRAM，包含BTB预测逻辑"""
    def __init__(self):
        super().__init__(ports={
        })
    
    @module.combinational
    def build(self, pc, stall, if_id_pc, if_id_instruction, if_id_valid, if_id_prediction_info, if_fetch_pending, if_fetch_pc, if_fetch_prediction_info, if_last_can_fetch, if_stall_buffer_valid, if_stall_buffer_pc, if_stall_buffer_instruction, if_stall_buffer_pred_info, btb, bht, btb_valid, decode_stage, unified_sram, ex_mem_valid, ex_mem_control, mem_wb_valid, sb_sh_state, ex_mem_pc, mem_last_pc):
        """冯诺依曼架构取指阶段
        
        Args:
            unified_sram: 统一SRAM (用于读取dout)
            if_fetch_pending: 是否有待完成的取指请求
            if_fetch_pc: 待取指的PC地址
            if_fetch_prediction_info: 待取指的预测信息
            if_last_can_fetch: 上周期IF是否成功访问SRAM
            if_stall_buffer_*: 取指缓冲区（SRAM stall时保存指令）
            ex_mem_valid, ex_mem_control, mem_wb_valid, sb_sh_state: 用于计算mem_sram_busy
            ex_mem_pc, mem_last_pc: 用于计算 is_new_instruction
            
        冯诺依曼架构取指时序:
        - SRAM 有 1-cycle latency
        - Cycle N: 发起读请求 (if_fetch_pending=1, if_fetch_pc=pc)
        - Cycle N+1: 得到指令 (from unified_sram.dout)
        - MEM阶段优先访问SRAM，IF需要等待
        - 如果SRAM stall时有取指完成，保存到缓冲区
        
        Returns:
            fetch_signals: 包含指令和IF的SRAM请求信号
            布局: [31:0]=instruction, [32]=if_needs_sram, [64:33]=if_sram_addr
        """
        current_pc = pc[0]
        fetch_pending = if_fetch_pending[0]
        fetch_pc = if_fetch_pc[0]
        fetch_pred_info = if_fetch_prediction_info[0]
        last_can_fetch = if_last_can_fetch[0]
        stall_buffer_valid = if_stall_buffer_valid[0]
        stall_buffer_pc = if_stall_buffer_pc[0]
        stall_buffer_instruction = if_stall_buffer_instruction[0]
        stall_buffer_pred_info = if_stall_buffer_pred_info[0]
        
        # ==================== 计算MEM是否需要SRAM ====================
        # 直接从流水线寄存器计算，与MemoryStage中的计算逻辑相同
        mem_control = ex_mem_control[0]
        mem_read_sig = mem_control[5:5]
        mem_write_sig = mem_control[6:6]
        store_type_sig = mem_control[22:23]  # 存储类型: 00=SB, 01=SH, 10=SW
        
        is_sb_sig = (store_type_sig == UInt(2)(0b00))
        is_sh_sig = (store_type_sig == UInt(2)(0b01))
        is_sw_sig = (store_type_sig == UInt(2)(0b10))
        needs_rmw_sig = mem_write_sig & (is_sb_sig | is_sh_sig)
        
        # SB/SH状态机状态
        current_sb_sh_state = sb_sh_state[0]
        is_idle_sig = (current_sb_sh_state == UInt(2)(0))
        is_write_phase_sig = (current_sb_sh_state == UInt(2)(2))
        
        # **关键**: 检测是否是新指令，与 MemoryStage 中的计算一致
        # 用于防止 load/SW 重复执行
        mem_pc = ex_mem_pc[0]
        last_pc = mem_last_pc[0]
        is_new_instruction = mem_pc != last_pc
        
        # MEM阶段是否需要SRAM (条件与MemoryStage中计算的相同)
        ex_mem_valid_val = ex_mem_valid[0]
        mem_wb_valid_val = mem_wb_valid[0]
        do_rmw_write_sig = is_write_phase_sig & mem_wb_valid_val
        do_rmw_read_sig = is_idle_sig & ex_mem_valid_val & needs_rmw_sig & mem_wb_valid_val
        # **关键修复**: do_sw_write 和 do_load_read 需要检查 is_new_instruction
        do_sw_write_sig = is_idle_sig & is_new_instruction & mem_write_sig & is_sw_sig & mem_wb_valid_val
        do_load_read_sig = is_idle_sig & is_new_instruction & mem_read_sig & (~mem_write_sig) & mem_wb_valid_val
        
        mem_sram_we_sig = do_rmw_write_sig | do_sw_write_sig
        mem_sram_re_sig = do_rmw_read_sig | do_load_read_sig
        
        # mem_sram_busy = 正在发起请求（写或读）
        # 1-cycle Load：MEM 阶段发起请求时 SRAM 被占用，下周期 WB 读取数据
        mem_sram_busy = mem_sram_we_sig | mem_sram_re_sig
        
        log("IF MEM_BUSY_CALC: ex_mem_valid={}, mem_wb_valid={}, mem_read={}, mem_write={}, is_idle={}, is_new={}, do_load_read={}, mem_sram_busy={}",
            ex_mem_valid_val, mem_wb_valid_val, mem_read_sig, mem_write_sig, is_idle_sig, is_new_instruction, do_load_read_sig, mem_sram_busy)
        
        # BTB查询逻辑 - 使用PC[2:7]作为索引(6位)
        btb_index = current_pc[2:7].bitcast(UInt(BTB_INDEX_BITS))
        
        # 读取BTB、BHT和有效位
        btb_entry = btb[btb_index]  # 预测目标地址
        bht_entry = bht[btb_index]  # 2-bit饱和计数器
        btb_valid_bit = btb_valid[btb_index]  # 有效位
        
        # BTB命中判断
        btb_hit = btb_valid_bit
        
        # 根据BHT值判断预测方向: bht >= 2 预测跳转
        predict_taken = (bht_entry >= UInt(2)(2)).select(UInt(1)(1), UInt(1)(0))
        
        # 如果BTB命中且预测跳转,使用BTB中的目标地址
        predicted_pc = (btb_hit & predict_taken).select(btb_entry, current_pc + UInt(XLEN)(4))
        
        # 构建预测信息: [0]: btb_hit, [1]: predict_taken, [2:33]: predicted_pc
        prediction_info = concat(
            predicted_pc,           # [33:2] 预测的PC (32位)
            predict_taken,          # [1]    预测是否跳转
            btb_hit                 # [0]    BTB是否命中
        ).bitcast(UInt(PREDICTION_INFO_LEN))
        
        # ==================== 冯诺依曼架构取指逻辑 ====================
        # 从SRAM读取的指令（上周期发起的请求的结果）
        sram_instruction = unified_sram.dout[0]
        
        # 本周期是否可以发起新的取指请求
        # 条件: MEM不在使用SRAM
        can_fetch = (~mem_sram_busy)
        
        # 判断SRAM输出是否是有效的指令
        # 条件: 上周期IF发起了取指请求(pending=1) 且 上周期IF成功访问了SRAM(last_can_fetch=1)
        # 如果上周期MEM占用了SRAM，那么即使pending=1，SRAM输出的也是MEM的数据
        instruction_ready = fetch_pending & last_can_fetch
        instruction = sram_instruction
        
        # ==================== IF/ID 寄存器更新逻辑 ====================
        # 使用缓冲区处理 SRAM stall 期间的取指结果
        #
        # 情况1: 缓冲区有效 且 MEM不忙
        #   -> 使用缓冲区的指令更新IF/ID，清空缓冲区
        # 
        # 情况2: SRAM取指完成 且 MEM不忙 且 缓冲区无效
        #   -> 直接更新IF/ID
        #
        # 情况3: SRAM取指完成 但 MEM忙
        #   -> 保存到缓冲区（不更新IF/ID）
        #
        # 情况4: 其他
        #   -> IF/ID保持不变
        
        # 情况1: 使用缓冲区
        with Condition(if_id_valid[0] & stall_buffer_valid & (~mem_sram_busy)):
            if_id_pc[0] = stall_buffer_pc
            if_id_instruction[0] = stall_buffer_instruction
            if_id_prediction_info[0] = stall_buffer_pred_info
            if_stall_buffer_valid[0] = UInt(1)(0)  # 清空缓冲区
            log("IF COMPLETE (from buffer): PC={:08x}, Instruction={:08x}", stall_buffer_pc, stall_buffer_instruction)
        
        # 情况2: 直接更新IF/ID
        with Condition(if_id_valid[0] & instruction_ready & (~mem_sram_busy) & (~stall_buffer_valid)):
            if_id_pc[0] = fetch_pc
            if_id_instruction[0] = instruction
            if_id_prediction_info[0] = fetch_pred_info
            log("IF COMPLETE: PC={:08x}, Instruction={:08x}", fetch_pc, instruction)
        
        # 情况3: 保存到缓冲区
        with Condition(instruction_ready & mem_sram_busy & (~stall_buffer_valid)):
            if_stall_buffer_valid[0] = UInt(1)(1)
            if_stall_buffer_pc[0] = fetch_pc
            if_stall_buffer_instruction[0] = instruction
            if_stall_buffer_pred_info[0] = fetch_pred_info
            log("IF STALL: Buffering PC={:08x}, Instruction={:08x}", fetch_pc, instruction)
        
        # 更新取指状态寄存器
        # 记录本周期IF是否成功访问SRAM（用于下周期判断SRAM输出是否有效）
        if_last_can_fetch[0] = can_fetch
        
        # pending状态管理
        with Condition(can_fetch):
            # 本周期可以发起取指
            if_fetch_pending[0] = UInt(1)(1)
            if_fetch_pc[0] = current_pc
            if_fetch_prediction_info[0] = prediction_info
            log("IF REQUEST: PC={:08x}, can_fetch={}, mem_busy={}", current_pc, can_fetch, mem_sram_busy)
        with Condition(~can_fetch):
            # MEM busy，IF不能发起新请求
            # 如果有有效的取指结果(instruction_ready)，已经更新了IF/ID，清除pending
            # 如果上周期的取指请求被MEM抢占（pending=1但last_can_fetch=0），保持pending等待重试
            # 实际上，当MEM busy时，不管之前的状态如何，都需要在MEM idle后重新取指
            # 因为SRAM请求被抢占了
            if_fetch_pending[0] = fetch_pending  # 保持pending状态

        decode_stage.async_called()

        # fetch_signals 逻辑:
        # - 有pending且完成: 输出SRAM读取的指令
        # - stall: 输出 0 (NOP)
        # - 其他: 使用存储的指令
        instruction_out = (instruction_ready & (~stall[0])).select(
            instruction,
            stall[0].select(UInt(XLEN)(0), if_id_instruction[0])
        )
        
        # 计算IF的SRAM请求信号
        if_needs_sram = can_fetch
        if_sram_addr = current_pc >> UInt(XLEN)(2)  # 字地址
        
        # 返回 fetch_signals，包含指令和IF的SRAM请求信号
        # 布局: [31:0]=instruction, [32]=if_needs_sram, [64:33]=if_sram_addr
        fetch_signals = concat(
            if_sram_addr.bitcast(Bits(XLEN)),      # [64:33] IF SRAM地址
            if_needs_sram.bitcast(Bits(1)),       # [32] IF是否需要SRAM
            instruction_out.bitcast(Bits(XLEN))   # [31:0] 指令
        ).bitcast(Bits(XLEN + 1 + XLEN))
        
        return fetch_signals

# ==================== ID阶段：指令解码 ===================
class DecodeStage(Module):
    """指令解码阶段(ID) - 传递预测信息"""
    def __init__(self):
        super().__init__(ports={})
    
    @module.combinational
    def build(self, if_id_valid, if_id_pc, if_id_instruction, if_id_prediction_info, id_ex_pc, id_ex_control, id_ex_valid, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, id_ex_need_rs1, id_ex_need_rs2, id_ex_prediction_info, reg_file, execute_stage):
        if_id_pc_in = if_id_pc[0]
        instruction = if_id_instruction[0]
        prediction_info_in = if_id_prediction_info[0]

        log("ID: PC={:08x}, Instruction={:08x}", if_id_pc_in, instruction)
        
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
        alu_a_zero = UInt(1)(0)  # LUI指令需要alu_a=0
        
        is_r_type = (opcode == UInt(7)(0b0110011))
        is_i_type = (opcode == UInt(7)(0b0010011))
        is_l_type = (opcode == UInt(7)(0b0000011))
        is_s_type = (opcode == UInt(7)(0b0100011))
        is_b_type = (opcode == UInt(7)(0b1100011))
        is_j_type = (opcode == UInt(7)(0b1101111))
        is_jr_type = (opcode == UInt(7)(0b1100111))
        is_lui_type = (opcode == UInt(7)(0b0110111))
        is_auipc_type = (opcode == UInt(7)(0b0010111))
        
        # M扩展指令检测: opcode=0110011, funct7=0000001
        is_m_ext = (is_r_type & (funct7 == UInt(7)(0b0000001)))
        
        # M扩展乘法指令解码 (func3决定具体操作)
        # func3: 000=MUL, 001=MULH, 010=MULHSU, 011=MULHU
        mul_op = UInt(3)(MUL_OP_NONE)
        mul_op = (is_m_ext & (func3 == UInt(3)(0b000))).select(UInt(3)(MUL_OP_MUL), mul_op)     # MUL
        mul_op = (is_m_ext & (func3 == UInt(3)(0b001))).select(UInt(3)(MUL_OP_MULH), mul_op)    # MULH
        mul_op = (is_m_ext & (func3 == UInt(3)(0b010))).select(UInt(3)(MUL_OP_MULHSU), mul_op)  # MULHSU
        mul_op = (is_m_ext & (func3 == UInt(3)(0b011))).select(UInt(3)(MUL_OP_MULHU), mul_op)   # MULHU
        
        # 是否为乘法指令
        is_mul_inst = (mul_op != UInt(3)(MUL_OP_NONE))
        
        # M扩展除法指令解码 (func3决定具体操作)
        # func3: 100=DIV, 101=DIVU, 110=REM, 111=REMU
        div_op = UInt(3)(DIV_OP_NONE)
        div_op = (is_m_ext & (func3 == UInt(3)(0b100))).select(UInt(3)(DIV_OP_DIV), div_op)   # DIV
        div_op = (is_m_ext & (func3 == UInt(3)(0b101))).select(UInt(3)(DIV_OP_DIVU), div_op)  # DIVU
        div_op = (is_m_ext & (func3 == UInt(3)(0b110))).select(UInt(3)(DIV_OP_REM), div_op)   # REM
        div_op = (is_m_ext & (func3 == UInt(3)(0b111))).select(UInt(3)(DIV_OP_REMU), div_op)  # REMU
        
        # 是否为除法指令
        is_div_inst = (div_op != UInt(3)(DIV_OP_NONE))
        # log("ID DECODE: opcode={:07b}, funct7={:07b}, func3={:03b}, is_r_type={}, is_m_ext={}, mul_op={}, is_mul_inst={}", 
            # opcode, funct7, func3, is_r_type, is_m_ext, mul_op, is_mul_inst)
        
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
        
        mem_read = is_l_type.select(UInt(1)(1), mem_read)  # Load指令
        reg_write = is_l_type.select(UInt(1)(1), reg_write)  # x0寄存器不会写入
        mem_to_reg = is_l_type.select(UInt(1)(1), mem_to_reg)  # Load指令
        alu_src = is_l_type.select(UInt(2)(1), alu_src)
        immediate = is_l_type.select(immediate_i, immediate)
        
        # Load类型解码 (3位, 使用funct3)
        # 000 = LB (Load Byte, signed)
        # 001 = LH (Load Halfword, signed)
        # 010 = LW (Load Word)
        # 100 = LBU (Load Byte Unsigned)
        # 101 = LHU (Load Halfword Unsigned)
        load_type_bits = UInt(3)(0b010)  # 默认LW
        load_type_bits = (is_l_type & (func3 == UInt(3)(0b000))).select(UInt(3)(0b000), load_type_bits)  # LB
        load_type_bits = (is_l_type & (func3 == UInt(3)(0b001))).select(UInt(3)(0b001), load_type_bits)  # LH
        load_type_bits = (is_l_type & (func3 == UInt(3)(0b010))).select(UInt(3)(0b010), load_type_bits)  # LW
        load_type_bits = (is_l_type & (func3 == UInt(3)(0b100))).select(UInt(3)(0b100), load_type_bits)  # LBU
        load_type_bits = (is_l_type & (func3 == UInt(3)(0b101))).select(UInt(3)(0b101), load_type_bits)  # LHU
            
        store_type_bits = UInt(2)(0)






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
        alu_a_zero = is_lui_type.select(UInt(1)(1), alu_a_zero)  # LUI需要alu_a=0
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
        
        # M扩展乘法指令设置
        reg_write = is_mul_inst.select(UInt(1)(1), reg_write)  # 乘法指令写回寄存器
        alu_src = is_mul_inst.select(UInt(2)(0), alu_src)  # 乘法使用寄存器操作数
        
        # M扩展除法指令设置
        reg_write = is_div_inst.select(UInt(1)(1), reg_write)  # 除法指令写回寄存器
        alu_src = is_div_inst.select(UInt(2)(0), alu_src)  # 除法使用寄存器操作数

        reg_write = (rd == UInt(5)(0)).select(UInt(1)(0), reg_write)  # rd为x0时不写入
        
        # 新控制信号格式 (48位):
        # [47:45] - div_op (3位除法操作码)
        # [44:42] - mul_op (3位乘法操作码)
        # [41:30] - 立即数低12位
        # [29:25] - rd地址
        # [24]    - 保留位
        # [23:22] - 存储类型: 00=SB, 01=SH, 10=SW
        # [21]    - jumpr_op
        # [20]    - jump_op
        # [19:17] - branch_op
        # [16:14] - load_type: 000=LB, 001=LH, 010=LW, 100=LBU, 101=LHU
        # [13:11] - 保留位
        # [10:9]  - alu_src
        # [8]     - mem_to_reg
        # [7]     - reg_write
        # [6]     - mem_write
        # [5]     - mem_read
        # [4:0]   - alu_op
        control_signals = concat(
            div_op,           # [47:45] 除法操作码
            mul_op,           # [44:42] 乘法操作码
            immediate[0:11],   # [41:30] 立即数低12位
            rd,               # [29:25] rd地址
            alu_a_zero,       # [24]    alu_a_zero (LUI需要alu_a=0)
            store_type_bits,  # [23:22] 存储类型: 00=SB, 01=SH, 10=SW
            jumpr_op,       # [21]    jumpr_op
            jump_op,          # [20]    跳转指令标志
            branch_op,        # [19:17] 分支操作类型
            load_type_bits,   # [16:14] load类型: 000=LB, 001=LH, 010=LW, 100=LBU, 101=LHU
            UInt(3)(0),       # [13:11] 保留位
            alu_src,          # [10:9]  ALU输入选择
            mem_to_reg,       # [8]     内存到寄存器
            reg_write,        # [7]     寄存器写
            mem_write,        # [6]     内存写
            mem_read,         # [5]     内存读
            alu_op,           # [4:0]   ALU操作码
        )

        # 乘法指令和除法指令也需要rs1和rs2
        need_rs1 = (is_i_type | is_r_type | is_s_type | is_b_type | is_l_type | is_jr_type | is_mul_inst | is_div_inst)
        need_rs2 = (is_r_type | is_s_type | is_b_type | is_mul_inst | is_div_inst)
        
        
        with Condition(id_ex_valid[0]):
            id_ex_pc[0] = if_id_valid[0].select(if_id_pc_in, UInt(XLEN)(0))
            id_ex_need_rs1[0] = if_id_valid[0].select(need_rs1, Bits(1)(0))
            id_ex_need_rs2[0] = if_id_valid[0].select(need_rs2, Bits(1)(0))
            # 传递预测信息到EX阶段
            id_ex_prediction_info[0] = if_id_valid[0].select(prediction_info_in, UInt(PREDICTION_INFO_LEN)(0))
            
            # id_ex_control[0] = control_signals
            # id_ex_valid[0] = UInt(1)(1)
            # id_ex_rs1_idx[0] = rs1
            # id_ex_rs2_idx[0] = rs2
            # id_ex_immediate[0] = immediate
            
            # log("ID: PC={}, Opcode={:07b}, RD={}, RS1={}, RS2={}, Immediate={}, Alu_op={}, Branch_op={}, Jump_op={}, Alu_src={}, Mem_read={}, Mem_write={}, Reg_write={}, Mem_to_reg={}, Control={:042b}",
                # if_id_pc_in, opcode, rd, rs1, rs2, immediate, alu_op, branch_op, jump_op, alu_src, mem_read, mem_write, reg_write, mem_to_reg, control_signals)
        
        # rs1 = (~if_id_valid[0]).select(Bits(5)(0), rs1)
        # rs2 = (~if_id_valid[0]).select(Bits(5)(0), rs2)
        # immediate = (~if_id_valid[0]).select(UInt(XLEN)(0), immediate)
        # control_signals = (~if_id_valid[0]).select(Bits(CONTROL_LEN)(0), control_signals)

        execute_stage.async_called()

        # decode_signals 的生成逻辑:
        # - id_ex_valid=0: 输出旧值（保持EX阶段指令）
        # - id_ex_valid=1, if_id_valid=0: 输出 0（清空EX阶段）
        # - 正常情况 (if_id_valid=1, id_ex_valid=1): 输出新值
        # 
        # 逻辑: id_ex_valid.select(if_id_valid.select(new_value, zero), old_value)
        out_control = id_ex_valid[0].select(if_id_valid[0].select(control_signals.bitcast(UInt(CONTROL_LEN)), UInt(CONTROL_LEN)(0)), id_ex_control[0])
        out_mul_op = out_control[42:44]
        # log("DECODE OUT: if_id_valid={}, id_ex_valid={}, control_mul_op={}, id_ex_mul_op={}, out_mul_op={}",
        #     if_id_valid[0], id_ex_valid[0], mul_op, id_ex_control[0][42:44], out_mul_op)
        
        decode_signals = concat(
            id_ex_valid[0].select(if_id_valid[0].select(prediction_info_in, UInt(PREDICTION_INFO_LEN)(0)), id_ex_prediction_info[0]),  # 预测信息 (34位)
            id_ex_valid[0].select(if_id_valid[0].select(need_rs2.bitcast(UInt(1)), UInt(1)(0)), id_ex_need_rs2[0].bitcast(UInt(1))), 
            id_ex_valid[0].select(if_id_valid[0].select(need_rs1.bitcast(UInt(1)), UInt(1)(0)), id_ex_need_rs1[0].bitcast(UInt(1))),
            id_ex_valid[0].select(if_id_valid[0].select(immediate, UInt(XLEN)(0)), id_ex_immediate[0]),
            id_ex_valid[0].select(if_id_valid[0].select(rs2.bitcast(UInt(5)), UInt(5)(0)), id_ex_rs2_idx[0]),
            id_ex_valid[0].select(if_id_valid[0].select(rs1.bitcast(UInt(5)), UInt(5)(0)), id_ex_rs1_idx[0]),
            out_control,
        )
        return decode_signals

# ==================== EX阶段：执行 ===================
class ExecuteStage(Module):
    """执行阶段(EX) - 包含预测验证逻辑"""
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
        
        # log("BRANCH: OP={:03b}, A={:08x}, B={:08x}, Taken={}",
        #     op, a, b, taken)
        
        return taken

    @module.combinational
    def build(self, id_ex_valid, id_ex_pc, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, id_ex_control, id_ex_prediction_info, ex_mem_pc, ex_mem_control, ex_mem_valid, ex_mem_result, ex_mem_data, reg_file, memory_stage, mem_wb_control, mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, data_sram, mem_wb_addr, ex_mem_pc_change, ex_mem_target_pc, ex_mem_prediction_result, ex_sig_control, mul_a, mul_b, mul_op_reg, mul_start, mul_cycle_counter, mul_stage1_sum, mul_stage1_carry, mul_stage2_sum, mul_stage2_carry, mul_valid, mul_result_reg, mul_in_progress, mul_rd_reg, mul_control_reg, mul_pc_reg, div_dividend, div_divisor, div_op_reg, div_state, div_remainder, div_quotient_pos, div_quotient_neg, div_iter_count, div_sign, div_dividend_sign, div_valid, div_result_reg, div_rd_reg, div_control_reg, div_pc_reg, div_norm_shift, div_divisor_norm):
        pc_in = id_ex_pc[0]
        rs1_idx = id_ex_rs1_idx[0]
        rs2_idx = id_ex_rs2_idx[0]
        immediate_in = id_ex_immediate[0]
        control_in = id_ex_control[0]
        prediction_info_in = id_ex_prediction_info[0]

        # ==================== Bypass/Forwarding 逻辑 ====================
        # 从寄存器文件读取基础值
        rs1_reg = reg_file[rs1_idx]
        rs2_reg = reg_file[rs2_idx]
        
        # 解析 MEM 阶段控制信号（来自 EX/MEM 寄存器）用于前递
        mem_control = ex_sig_control[0]
        mem_reg_write = mem_control[7:7]  # reg_write 在第7位
        mem_rd = mem_control[25:29]       # rd 在第25-29位
        mem_result = ex_mem_result[0]     # MEM 阶段的 ALU 结果
        
        # 解析 WB 阶段控制信号用于前递
        wb_control = mem_wb_control[0]
        wb_reg_write = wb_control[7:7]    # reg_write 在第7位
        wb_mem_to_reg = wb_control[8:8]   # mem_to_reg 在第8位
        wb_rd = wb_control[25:29]         # rd 在第25-29位
        wb_ex_result = mem_wb_ex_result[0]
        # 从 SRAM 读取的数据，并根据load类型处理
        wb_raw_mem_data = data_sram.dout[0]
        wb_load_type = wb_control[14:16]  # load类型: 000=LB, 001=LH, 010=LW, 100=LBU, 101=LHU
        wb_byte_offset = mem_wb_addr[0][0:1]  # 地址低2位
        # 内联处理load数据 (ExecuteStage前递)
        wb_byte0 = wb_raw_mem_data[0:7]
        wb_byte1 = wb_raw_mem_data[8:15]
        wb_byte2 = wb_raw_mem_data[16:23]
        wb_byte3 = wb_raw_mem_data[24:31]
        wb_selected_byte = (wb_byte_offset == UInt(2)(0)).select(wb_byte0,
                           (wb_byte_offset == UInt(2)(1)).select(wb_byte1,
                           (wb_byte_offset == UInt(2)(2)).select(wb_byte2, wb_byte3)))
        wb_half0 = wb_raw_mem_data[0:15]
        wb_half1 = wb_raw_mem_data[16:31]
        wb_selected_half = (wb_byte_offset[1:1] == UInt(1)(0)).select(wb_half0, wb_half1)
        wb_byte_sign = wb_selected_byte[7:7]
        wb_lb_data = concat(wb_byte_sign.select(UInt(24)(0xFFFFFF), UInt(24)(0)), wb_selected_byte).bitcast(UInt(XLEN))
        wb_lbu_data = concat(UInt(24)(0), wb_selected_byte).bitcast(UInt(XLEN))
        wb_half_sign = wb_selected_half[15:15]
        wb_lh_data = concat(wb_half_sign.select(UInt(16)(0xFFFF), UInt(16)(0)), wb_selected_half).bitcast(UInt(XLEN))
        wb_lhu_data = concat(UInt(16)(0), wb_selected_half).bitcast(UInt(XLEN))
        wb_lw_data = wb_raw_mem_data
        wb_mem_data = (wb_load_type == UInt(3)(0b000)).select(wb_lb_data,
                      (wb_load_type == UInt(3)(0b001)).select(wb_lh_data,
                      (wb_load_type == UInt(3)(0b010)).select(wb_lw_data,
                      (wb_load_type == UInt(3)(0b100)).select(wb_lbu_data, wb_lhu_data))))
        
        # WB 阶段数据选择：若 mem_to_reg=1 使用内存数据，否则使用 ALU 结果
        wb_data = wb_mem_to_reg.select(wb_mem_data, wb_ex_result)
        
        # rs1 前递逻辑：优先级 MEM > WB > reg_file
        # 条件：reg_write=1 且 rs1_idx=rd 且 rd!=0（x0不能前递）
        # 注意：不使用 ex_mem_valid 作为条件，因为在 mem_sram_stall 期间 ex_mem_valid=0，
        # 但 MEM 阶段的 ALU 结果（非 load 指令）仍然是有效的，需要被转发。
        # mem_reg_write 来自 ex_sig_control，当 flush 时会被清零，所以不会产生错误转发。
        rs1_forward_mem = (mem_reg_write & (rs1_idx == mem_rd) & (mem_rd != UInt(5)(0)))
        rs1_forward_wb = (mem_wb_valid[0] & wb_reg_write & (rs1_idx == wb_rd) & (wb_rd != UInt(5)(0)))
        
        rs1_data = rs1_reg
        rs1_data = rs1_forward_wb.select(wb_data, rs1_data)
        rs1_data = rs1_forward_mem.select(mem_result, rs1_data)
        
        # rs2 前递逻辑：优先级 MEM > WB > reg_file
        rs2_forward_mem = (mem_reg_write & (rs2_idx == mem_rd) & (mem_rd != UInt(5)(0)))
        rs2_forward_wb = (mem_wb_valid[0] & wb_reg_write & (rs2_idx == wb_rd) & (wb_rd != UInt(5)(0)))
        
        rs2_data = rs2_reg
        rs2_data = rs2_forward_wb.select(wb_data, rs2_data)
        rs2_data = rs2_forward_mem.select(mem_result, rs2_data)
        
        log("EX FORWARD: PC={:08x}, rs1_idx={}, rs2_idx={}, rs1_fwd_mem={}, rs1_fwd_wb={}, rs2_fwd_mem={}, rs2_fwd_wb={}",
            pc_in, rs1_idx, rs2_idx, rs1_forward_mem, rs1_forward_wb, rs2_forward_mem, rs2_forward_wb)
        log("EX FORWARD DATA: rs1_reg={:08x}, rs2_reg={:08x}, mem_result={:08x}, wb_data={:08x}, rs1_data={:08x}, rs2_data={:08x}",
            rs1_reg, rs2_reg, mem_result, wb_data, rs1_data, rs2_data)
        log("EX FORWARD COND: ex_mem_valid={}, mem_wb_valid={}, mem_rd={}, wb_rd={}, mem_reg_write={}, wb_reg_write={}",
            ex_mem_valid[0], mem_wb_valid[0], mem_rd, wb_rd, mem_reg_write, wb_reg_write)
        
        # 初始化PC变化控制信号
        pc_change = UInt(1)(0)
        target_pc = pc_in + UInt(XLEN)(4)  # 默认目标PC是PC+4

        # 解析控制信号 (新格式48位)
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
        alu_a_zero = control_in[24:24]  # LUI需要alu_a=0
        immediate = control_in[22:31]  # 立即数
        mul_op = control_in[42:44]  # 乘法操作码 [44:42]
        div_op = control_in[45:47]  # 除法操作码 [47:45]
        
        # 判断是否为乘法指令
        is_mul_inst = (mul_op != UInt(3)(MUL_OP_NONE))
        
        # 判断是否为除法指令
        is_div_inst = (div_op != UInt(3)(DIV_OP_NONE))
        
        # 解析预测信息: [0]: btb_hit, [1]: predict_taken, [2:33]: predicted_pc
        btb_hit = prediction_info_in[0:0]
        predict_taken = prediction_info_in[1:1]
        predicted_pc = prediction_info_in[2:33].bitcast(UInt(XLEN))
        
        # ALU输入B选择
        alu_b = immediate_in
        alu_b = (alu_src == UInt(2)(0)).select(rs2_data, alu_b)  # alu_src=0: rs2_data
        alu_b = (alu_src == UInt(2)(1)).select(immediate_in, alu_b)  # alu_src=1: immediate
        alu_b = (alu_src == UInt(2)(2)).select(immediate_in, alu_b)  # alu_src=2: immediate
        
        # 根据指令类型决定执行ALU操作还是分支操作
        alu_result = UInt(XLEN)(0)
        
        # 判断是否为分支指令 (branch_op != 0)
        is_branch = (branch_op != UInt(3)(0b000))
        is_jump = (jump_op == UInt(1)(1))
        is_jumpr = (jumpr_op == UInt(1)(1))
        
        # 对于AUIPC指令，ALU输入A应该是PC而不是rs1_data
        alu_a = rs1_data
        alu_a = (alu_src == UInt(2)(0)).select(rs1_data, alu_a)  # alu_src=0: rs1_data
        alu_a = (alu_src == UInt(2)(1)).select(rs1_data, alu_a)  # alu_src=1: rs1_data
        alu_a = (alu_src == UInt(2)(2)).select(pc_in, alu_a)      # alu_src=2: pc
        # LUI需要alu_a=0
        alu_a = alu_a_zero.select(UInt(XLEN)(0), alu_a)
        # JAL/JALR需要alu_a=pc (计算返回地址 pc+4)
        alu_a = (is_jump | is_jumpr).select(pc_in, alu_a)
        # JAL/JALR需要alu_b=4 (计算返回地址 pc+4)
        alu_b = (is_jump | is_jumpr).select(UInt(XLEN)(4), alu_b)

        # 计算实际分支结果
        actual_taken = is_branch.select(self.branch_unit(branch_op, rs1_data, rs2_data), UInt(1)(0))
        
        # 计算实际目标地址
        actual_target_pc = pc_in + immediate_in
        new_pc_temp = rs1_data + immediate_in
        new_pc = (new_pc_temp ^ (new_pc_temp & UInt(XLEN)(1)))
        
        # 分支正确的下一个PC (taken则跳转到目标，否则PC+4)
        correct_pc = actual_taken.select(actual_target_pc, pc_in + UInt(XLEN)(4))
        
        # 预测验证逻辑 (根据branch_prediction_rules.md)
        # BTB命中时: prediction_correct = (predict_taken == actual_taken) && (predicted_pc == correct_pc)
        # BTB未命中时: prediction_correct = !actual_taken
        prediction_correct_hit = ((predict_taken == actual_taken) & (predicted_pc == correct_pc)).select(UInt(1)(1), UInt(1)(0))
        prediction_correct_miss = (~actual_taken).select(UInt(1)(1), UInt(1)(0))
        prediction_correct = btb_hit.select(prediction_correct_hit, prediction_correct_miss)
        
        # 仅对分支指令生成mispredict信号
        mispredict = (is_branch & ~prediction_correct).select(UInt(1)(1), UInt(1)(0))
        
        # DEBUG: 分支预测调试
        log("EX BRANCH DEBUG: PC={:08x}, is_branch={}, branch_op={:03b}, rs1_data={:08x}, rs2_data={:08x}, actual_taken={}, btb_hit={}, predict_taken={}, mispredict={}",
            pc_in, is_branch, branch_op, rs1_data, rs2_data, actual_taken, btb_hit, predict_taken, mispredict)
        
        # ==================== 乘法器逻辑 ====================
        # 乘法器状态检查
        mul_cycle = mul_cycle_counter[0]
        mul_busy = (mul_cycle != UInt(2)(0)).select(UInt(1)(1), UInt(1)(0))
        mul_done = (mul_cycle == UInt(2)(3)).select(UInt(1)(1), UInt(1)(0))
        
        # 当前是否需要启动新的乘法
        # 只有当乘法器空闲且当前指令是乘法指令时才启动
        start_new_mul = (is_mul_inst & id_ex_valid[0] & ~mul_busy).select(UInt(1)(1), UInt(1)(0))
        # log("MUL CHECK: is_mul_inst={}, id_ex_valid={}, mul_busy={}, mul_op={}, start_new_mul={}", 
        #     is_mul_inst, id_ex_valid[0], mul_busy, mul_op, start_new_mul)
        
        # 保存乘法操作数和控制信息
        with Condition(start_new_mul):
            mul_a[0] = rs1_data
            mul_b[0] = rs2_data
            mul_op_reg[0] = mul_op
            mul_rd_reg[0] = rd_addr
            mul_control_reg[0] = control_in
            mul_pc_reg[0] = pc_in
            mul_in_progress[0] = UInt(1)(1)
            mul_cycle_counter[0] = UInt(2)(1)  # 开始第1周期
            # log("MUL START: a={}, b={}, mul_op={}", rs1_data, rs2_data, mul_op)
        
        # ==================== Wallace Tree 乘法器计算 ====================
        # Cycle 1: 生成部分积并进行第一级CSA压缩
        with Condition(mul_cycle == UInt(2)(1)):
            a = mul_a[0]
            b = mul_b[0]
            saved_op = mul_op_reg[0]
            # log("MUL CYCLE 1 READ: a={}, b={}, mul_a[0]={}", a, b, mul_a[0])
            
            # 确定操作数符号属性
            a_signed = ((saved_op == UInt(3)(MUL_OP_MUL)) | (saved_op == UInt(3)(MUL_OP_MULH)) | (saved_op == UInt(3)(MUL_OP_MULHSU))).select(UInt(1)(1), UInt(1)(0))
            b_signed = ((saved_op == UInt(3)(MUL_OP_MUL)) | (saved_op == UInt(3)(MUL_OP_MULH))).select(UInt(1)(1), UInt(1)(0))
            
            # 符号扩展到64位
            a_sign = a[31:31]
            b_sign = b[31:31]
            a_high = (a_signed & a_sign).select(UInt(32)(0xFFFFFFFF), UInt(32)(0))
            b_high = (b_signed & b_sign).select(UInt(32)(0xFFFFFFFF), UInt(32)(0))
            
            # 直接将32位数转为64位进行计算，不使用concat
            a_64 = a.bitcast(UInt(64))
            b_64 = b.bitcast(UInt(64))
            # log("MUL DEBUG3: a={}, a_64={}, b={}, b_64={}", a, a_64, b, b_64)
            
            # 生成32个部分积 (移位后需要bitcast回UInt(64))
            # 使用显式比较确保条件正确
            pp0 = (b[0:0] == UInt(1)(1)).select(a_64, UInt(64)(0))
            pp1 = (b[1:1] == UInt(1)(1)).select((a_64 << UInt(64)(1)).bitcast(UInt(64)), UInt(64)(0))
            pp2 = (b[2:2] == UInt(1)(1)).select((a_64 << UInt(64)(2)).bitcast(UInt(64)), UInt(64)(0))
            pp3 = (b[3:3] == UInt(1)(1)).select((a_64 << UInt(64)(3)).bitcast(UInt(64)), UInt(64)(0))
            pp4 = (b[4:4] == UInt(1)(1)).select((a_64 << UInt(64)(4)).bitcast(UInt(64)), UInt(64)(0))
            pp5 = (b[5:5] == UInt(1)(1)).select((a_64 << UInt(64)(5)).bitcast(UInt(64)), UInt(64)(0))
            pp6 = (b[6:6] == UInt(1)(1)).select((a_64 << UInt(64)(6)).bitcast(UInt(64)), UInt(64)(0))
            pp7 = (b[7:7] == UInt(1)(1)).select((a_64 << UInt(64)(7)).bitcast(UInt(64)), UInt(64)(0))
            pp8 = (b[8:8] == UInt(1)(1)).select((a_64 << UInt(64)(8)).bitcast(UInt(64)), UInt(64)(0))
            pp9 = (b[9:9] == UInt(1)(1)).select((a_64 << UInt(64)(9)).bitcast(UInt(64)), UInt(64)(0))
            pp10 = (b[10:10] == UInt(1)(1)).select((a_64 << UInt(64)(10)).bitcast(UInt(64)), UInt(64)(0))
            pp11 = (b[11:11] == UInt(1)(1)).select((a_64 << UInt(64)(11)).bitcast(UInt(64)), UInt(64)(0))
            pp12 = (b[12:12] == UInt(1)(1)).select((a_64 << UInt(64)(12)).bitcast(UInt(64)), UInt(64)(0))
            pp13 = (b[13:13] == UInt(1)(1)).select((a_64 << UInt(64)(13)).bitcast(UInt(64)), UInt(64)(0))
            pp14 = (b[14:14] == UInt(1)(1)).select((a_64 << UInt(64)(14)).bitcast(UInt(64)), UInt(64)(0))
            pp15 = (b[15:15] == UInt(1)(1)).select((a_64 << UInt(64)(15)).bitcast(UInt(64)), UInt(64)(0))
            pp16 = (b[16:16] == UInt(1)(1)).select((a_64 << UInt(64)(16)).bitcast(UInt(64)), UInt(64)(0))
            pp17 = (b[17:17] == UInt(1)(1)).select((a_64 << UInt(64)(17)).bitcast(UInt(64)), UInt(64)(0))
            pp18 = (b[18:18] == UInt(1)(1)).select((a_64 << UInt(64)(18)).bitcast(UInt(64)), UInt(64)(0))
            pp19 = (b[19:19] == UInt(1)(1)).select((a_64 << UInt(64)(19)).bitcast(UInt(64)), UInt(64)(0))
            pp20 = (b[20:20] == UInt(1)(1)).select((a_64 << UInt(64)(20)).bitcast(UInt(64)), UInt(64)(0))
            pp21 = (b[21:21] == UInt(1)(1)).select((a_64 << UInt(64)(21)).bitcast(UInt(64)), UInt(64)(0))
            pp22 = (b[22:22] == UInt(1)(1)).select((a_64 << UInt(64)(22)).bitcast(UInt(64)), UInt(64)(0))
            pp23 = (b[23:23] == UInt(1)(1)).select((a_64 << UInt(64)(23)).bitcast(UInt(64)), UInt(64)(0))
            pp24 = (b[24:24] == UInt(1)(1)).select((a_64 << UInt(64)(24)).bitcast(UInt(64)), UInt(64)(0))
            pp25 = (b[25:25] == UInt(1)(1)).select((a_64 << UInt(64)(25)).bitcast(UInt(64)), UInt(64)(0))
            pp26 = (b[26:26] == UInt(1)(1)).select((a_64 << UInt(64)(26)).bitcast(UInt(64)), UInt(64)(0))
            pp27 = (b[27:27] == UInt(1)(1)).select((a_64 << UInt(64)(27)).bitcast(UInt(64)), UInt(64)(0))
            pp28 = (b[28:28] == UInt(1)(1)).select((a_64 << UInt(64)(28)).bitcast(UInt(64)), UInt(64)(0))
            pp29 = (b[29:29] == UInt(1)(1)).select((a_64 << UInt(64)(29)).bitcast(UInt(64)), UInt(64)(0))
            pp30 = (b[30:30] == UInt(1)(1)).select((a_64 << UInt(64)(30)).bitcast(UInt(64)), UInt(64)(0))
            pp31 = (b[31:31] == UInt(1)(1)).select((a_64 << UInt(64)(31)).bitcast(UInt(64)), UInt(64)(0))
            
            # CSA函数: sum = a ^ b ^ c, carry = ((a&b)|(b&c)|(a&c)) << 1
            def csa(x, y, z):
                s = (x ^ y ^ z).bitcast(UInt(64))
                c = (((x & y) | (y & z) | (x & z)) << UInt(64)(1)).bitcast(UInt(64))
                return s, c
            
            # 第一级CSA: 32->22 (10组CSA压缩)
            s0, c0 = csa(pp0, pp1, pp2)
            s1, c1 = csa(pp3, pp4, pp5)
            s2, c2 = csa(pp6, pp7, pp8)
            s3, c3 = csa(pp9, pp10, pp11)
            s4, c4 = csa(pp12, pp13, pp14)
            s5, c5 = csa(pp15, pp16, pp17)
            s6, c6 = csa(pp18, pp19, pp20)
            s7, c7 = csa(pp21, pp22, pp23)
            s8, c8 = csa(pp24, pp25, pp26)
            s9, c9 = csa(pp27, pp28, pp29)
            # pp30, pp31 保留
            
            # 第二级CSA: 22->15
            t0, u0 = csa(s0, c0, s1)
            t1, u1 = csa(c1, s2, c2)
            t2, u2 = csa(s3, c3, s4)
            t3, u3 = csa(c4, s5, c5)
            t4, u4 = csa(s6, c6, s7)
            t5, u5 = csa(c7, s8, c8)
            t6, u6 = csa(s9, c9, pp30)
            # pp31 保留
            
            # 第三级CSA: 15->10
            v0, w0 = csa(t0, u0, t1)
            v1, w1 = csa(u1, t2, u2)
            v2, w2 = csa(t3, u3, t4)
            v3, w3 = csa(u4, t5, u5)
            v4, w4 = csa(t6, u6, pp31)
            
            # 第四级CSA: 10->7
            # 输入: v0, w0, v1, w1, v2, w2, v3, w3, v4, w4
            x0, y0 = csa(v0, w0, v1)
            x1, y1 = csa(w1, v2, w2)
            x2, y2 = csa(v3, w3, v4)
            # 保留: w4
            # 输出: x0, y0, x1, y1, x2, y2, w4 (7个)
            
            # 第五级CSA: 7->5
            z0, z1 = csa(x0, y0, x1)
            z2, z3 = csa(y1, x2, y2)
            # 保留: w4
            # 输出: z0, z1, z2, z3, w4 (5个)
            
            # 第六级CSA: 5->4
            q0, q1 = csa(z0, z1, z2)
            # 保留: z3, w4
            # 输出: q0, q1, z3, w4 (4个)
            
            # log("MUL CYCLE 1: a_64={}, b_64={}, pp0={}, pp1={}, pp2={}", a_64, b_64, pp0, pp1, pp2)
            # log("MUL CYCLE 1 END: q0={}, q1={}, z3={}, w4={}", q0, q1, z3, w4)
            
            # 保存4个完整的64位中间值
            mul_stage1_sum[0] = q0
            mul_stage1_carry[0] = q1
            mul_stage2_sum[0] = z3
            mul_stage2_carry[0] = w4
            
            mul_cycle_counter[0] = UInt(2)(2)
        
        # Cycle 2: 继续CSA压缩 (4->3->2)
        with Condition(mul_cycle == UInt(2)(2)):
            # 从寄存器恢复4个64位中间值
            q0_r = mul_stage1_sum[0]
            q1_r = mul_stage1_carry[0]
            z3_r = mul_stage2_sum[0]
            w4_r = mul_stage2_carry[0]
            
            def csa2(x, y, z):
                s = (x ^ y ^ z).bitcast(UInt(64))
                c = (((x & y) | (y & z) | (x & z)) << UInt(64)(1)).bitcast(UInt(64))
                return s, c
            
            # 第七级CSA: 4->3
            r0, r1 = csa2(q0_r, q1_r, z3_r)
            # 保留: w4_r
            # 输出: r0, r1, w4_r (3个)
            
            # 第八级CSA: 3->2
            final_sum, final_carry = csa2(r0, r1, w4_r)
            # 输出: final_sum, final_carry (2个)
            
            # 保存最终的sum和carry
            mul_stage1_sum[0] = final_sum
            mul_stage1_carry[0] = final_carry
            mul_cycle_counter[0] = UInt(2)(3)
        
        # Cycle 3: 最终加法并选择结果
        with Condition(mul_cycle == UInt(2)(3)):
            final_result = mul_stage1_sum[0] + mul_stage1_carry[0]
            saved_op = mul_op_reg[0]
            
            # 根据mul_op选择高32位或低32位
            result_low = final_result[0:31].bitcast(UInt(32))
            result_high = final_result[32:63].bitcast(UInt(32))
            
            # MUL: 低32位; MULH/MULHSU/MULHU: 高32位
            mul_result_val = (saved_op == UInt(3)(MUL_OP_MUL)).select(result_low, result_high)
            # log("MUL CYCLE 3: sum={}, carry={}, final_result={}, result_low={}, saved_op={}", 
                # mul_stage1_sum[0], mul_stage1_carry[0], final_result, result_low, saved_op)
            mul_result_reg[0] = mul_result_val
            mul_valid[0] = UInt(1)(1)
            mul_cycle_counter[0] = UInt(2)(0)
            mul_in_progress[0] = UInt(1)(0)
        
        # 在外部也计算当前周期的乘法结果（供 mul_done 时使用）
        # 这个计算在每个周期都会执行，但只有在 mul_cycle == 3 时结果才有意义
        current_final_result = mul_stage1_sum[0] + mul_stage1_carry[0]
        current_result_low = current_final_result[0:31].bitcast(UInt(32))
        current_result_high = current_final_result[32:63].bitcast(UInt(32))
        current_saved_op = mul_op_reg[0]
        current_mul_result = (current_saved_op == UInt(3)(MUL_OP_MUL)).select(current_result_low, current_result_high)
        
        # 非乘法周期重置valid
        with Condition(mul_cycle == UInt(2)(0)):
            mul_valid[0] = UInt(1)(0)
        
        # ==================== Radix-4 SRT 不恢复除法器（带查表）====================
        # 状态机: 0=IDLE, 1=INIT, 2-17=ITERATE (16 iterations), 18=FINAL_CORRECTION, 19=DONE
        # 
        # Radix-4 SRT 算法关键：
        # - 商数字集合: q ∈ {-2, -1, 0, +1, +2}
        # - 递归关系: P_{i+1} = 4 * P_i - q_i * D
        # - 归一化: 将除数归一化使 D ∈ [0.5, 1)，即最高位为1
        # - 查表: 根据 P_i 的高位选择商数字
        # - On-the-fly 转换: 使用 Q+ 和 Q- 累积器避免最终修正
        
        # 除法器状态检查
        div_state_val = div_state[0]
        div_busy = (div_state_val != UInt(6)(0)).select(UInt(1)(1), UInt(1)(0))
        div_done = (div_state_val == UInt(6)(19)).select(UInt(1)(1), UInt(1)(0))
        
        # 当前是否需要启动新的除法
        start_new_div = (is_div_inst & id_ex_valid[0] & ~div_busy).select(UInt(1)(1), UInt(1)(0))
        
        # 保存除法操作数和控制信息
        with Condition(start_new_div):
            div_dividend[0] = rs1_data
            div_divisor[0] = rs2_data
            div_op_reg[0] = div_op
            div_rd_reg[0] = rd_addr
            div_control_reg[0] = control_in
            div_pc_reg[0] = pc_in
            div_state[0] = UInt(6)(1)  # INIT state
            div_iter_count[0] = UInt(5)(0)
            div_sign[0] = UInt(1)(0)
            div_dividend_sign[0] = UInt(1)(0)
            div_valid[0] = UInt(1)(0)
        
        # State 1: INIT - 初始化
        with Condition((div_state_val == UInt(6)(1)) & ~start_new_div):
            saved_op = div_op_reg[0]
            dividend = div_dividend[0]
            divisor = div_divisor[0]
            
            # 检查除零
            div_zero = (divisor == UInt(32)(0))
            
            # 处理符号（有符号除法）
            is_signed = ((saved_op == UInt(3)(DIV_OP_DIV)) | (saved_op == UInt(3)(DIV_OP_REM))).select(UInt(1)(1), UInt(1)(0))
            
            # 被除数和除数的符号
            dividend_sign_bit = dividend[31:31]
            divisor_sign_bit = divisor[31:31]
            
            # 商的符号（仅有符号除法需要）
            quotient_sign = (is_signed & (dividend_sign_bit ^ divisor_sign_bit)).select(UInt(1)(1), UInt(1)(0))
            # 被除数的符号（用于余数符号）
            remainder_sign = (is_signed & dividend_sign_bit).select(UInt(1)(1), UInt(1)(0))
            
            # 取绝对值（有符号时）
            dividend_abs = ((is_signed == UInt(1)(1)) & (dividend_sign_bit == UInt(1)(1))).select(
                (~dividend).bitcast(UInt(32)) + UInt(32)(1), dividend)
            divisor_abs = ((is_signed == UInt(1)(1)) & (divisor_sign_bit == UInt(1)(1))).select(
                (~divisor).bitcast(UInt(32)) + UInt(32)(1), divisor)
            
            # 除零处理
            with Condition(div_zero):
                div_result_val = UInt(32)(0xFFFFFFFF)  # DIV/DIVU: -1/0xFFFFFFFF
                rem_result_val = dividend              # REM/REMU: 原被除数
                final_div_result = ((saved_op == UInt(3)(DIV_OP_DIV)) | (saved_op == UInt(3)(DIV_OP_DIVU))).select(
                    div_result_val, rem_result_val)
                div_result_reg[0] = final_div_result
                div_valid[0] = UInt(1)(1)
                div_state[0] = UInt(6)(19)  # DONE
            
            with Condition(~div_zero):
                # ========== Radix-4 SRT（正确的定点格式）==========
                # 在 SRT 除法中，我们使用定点格式：
                # - P 是 64 位有符号数，实际值 = P / 2^32
                # - D 是 32 位无符号数，实际值 = D / 2^32
                # - 商 Q 的实际值 = P_initial / D
                # 
                # 初始设置：
                # - P_0 = dividend_abs（放在 64 位的低 32 位）
                # - D = divisor_abs（32 位）
                # 
                # 对于 radix-4，每次迭代：
                # 1. P_shifted = P << 2
                # 2. 选择 q ∈ {-2, -1, 0, 1, 2}
                # 3. P_new = P_shifted - q * D（D 扩展到 64 位）
                # 
                # 16 次迭代产生 32 位商
                
                # 初始余数 = 被除数（放在 64 位的低 32 位）
                initial_remainder_unsigned = concat(Bits(32)(0), dividend_abs.bitcast(Bits(32))).bitcast(UInt(64))
                initial_remainder = initial_remainder_unsigned.bitcast(Int(64))
                
                # DEBUG: 打印初始化状态
                # log("DIV INIT: dividend={}, divisor={}, dividend_abs={}, divisor_abs={}", 
                #     dividend, divisor, dividend_abs, divisor_abs)
                # log("DIV INIT: initial_P={:016x}, divisor={:08x}", 
                #     initial_remainder, divisor_abs)
                
                # 保存状态
                div_divisor[0] = divisor_abs
                div_divisor_norm[0] = divisor_abs  # 不使用归一化
                div_norm_shift[0] = UInt(6)(0)
                div_remainder[0] = initial_remainder
                div_quotient_pos[0] = UInt(32)(0)
                div_quotient_neg[0] = UInt(32)(0)
                div_sign[0] = quotient_sign
                div_dividend_sign[0] = remainder_sign
                div_iter_count[0] = UInt(5)(0)
                div_state[0] = UInt(6)(2)  # 进入迭代
        
        # States 2-17: ITERATE - 16次迭代（每次产生2位商）
        # Radix-4 SRT 迭代：
        # 1. 先左移 2 位（乘以 4）
        # 2. 根据部分余数的高位选择商数字 q_i ∈ {-2, -1, 0, +1, +2}
        # 3. 更新部分余数: P_{i+1} = P_shifted - q_i * D
        # 4. On-the-fly 更新商: Q+, Q-
        # 
        # 定点格式：
        # - P 是 64 位有符号数
        # - D 是 32 位无符号数
        # - 每次迭代前 P 左移 2 位，使高位进入比较范围
        # - 比较 P 的高 32 位与 D 来选择商
        state_ge_2 = (div_state_val >= UInt(6)(2))
        state_lt_18 = (div_state_val < UInt(6)(18))
        div_state_in_iterate = state_ge_2 & state_lt_18 & ~start_new_div
        with Condition(div_state_in_iterate):
            iter_num = div_iter_count[0]
            current_P = div_remainder[0]  # 有符号64位部分余数
            D = div_divisor_norm[0]       # 除数（32位无符号）
            Q_pos = div_quotient_pos[0]
            Q_neg = div_quotient_neg[0]
            
            # 将部分余数左移2位（乘以4），radix-4
            P_shifted = (current_P << Int(64)(2)).bitcast(Int(64))
            
            # ========== 商数字选择（基于 P 和 D 的比较）==========
            # P_shifted 是 64 位，取高 32 位进行比较
            # P_high = P_shifted[32:63]（有符号 32 位）
            P_high_32 = P_shifted[32:63].bitcast(Int(32))
            D_signed = D.bitcast(Int(32))
            
            # 计算比较边界
            # 对于 radix-4 SRT，商数字 q ∈ {-2, -1, 0, +1, +2}
            # 选择规则：
            # q = +2 if P_high >= 2*D
            # q = +1 if P_high >= D and P_high < 2*D
            # q = 0  if P_high >= 0 and P_high < D (or P_high >= -D and P_high < 0)
            # q = -1 if P_high >= -2*D and P_high < -D
            # q = -2 if P_high < -2*D
            # 
            # 但是对于非归一化的除法，我们需要更宽松的边界
            # 使用标准 SRT 选择：基于 P 和 D 的比值
            
            two_D = (D_signed << Int(32)(1)).bitcast(Int(32))
            neg_D = (~D_signed + Int(32)(1)).bitcast(Int(32))
            neg_two_D = (~two_D + Int(32)(1)).bitcast(Int(32))
            
            # 选择商数字 q_i
            q_sel = Int(3)(0)
            q_sel = (P_high_32 >= two_D).select(Int(3)(2), q_sel)         # q = +2
            q_sel = ((P_high_32 >= D_signed) & (P_high_32 < two_D)).select(Int(3)(1), q_sel)    # q = +1
            q_sel = ((P_high_32 >= Int(32)(0)) & (P_high_32 < D_signed)).select(Int(3)(0), q_sel)     # q = 0 (P >= 0)
            q_sel = ((P_high_32 >= neg_D) & (P_high_32 < Int(32)(0))).select(Int(3)(0), q_sel)        # q = 0 (P < 0)
            q_sel = ((P_high_32 >= neg_two_D) & (P_high_32 < neg_D)).select(Int(3)(-1), q_sel)   # q = -1
            q_sel = (P_high_32 < neg_two_D).select(Int(3)(-2), q_sel)     # q = -2
            
            # ========== 计算 q * D ==========
            # D 作为 64 位数的高 32 位（乘以 2^32）
            D_64 = concat(D.bitcast(Bits(32)), Bits(32)(0)).bitcast(Int(64))
            
            # 根据 q 计算 q * D
            qD = Int(64)(0)
            qD = (q_sel == Int(3)(2)).select((D_64 << Int(64)(1)).bitcast(Int(64)), qD)   # +2 * D
            qD = (q_sel == Int(3)(1)).select(D_64, qD)                                    # +1 * D
            qD = (q_sel == Int(3)(0)).select(Int(64)(0), qD)                              # 0 * D
            neg_D_64 = (~D_64 + Int(64)(1)).bitcast(Int(64))  # -D
            neg_2D_64 = (~(D_64 << Int(64)(1)).bitcast(Int(64)) + Int(64)(1)).bitcast(Int(64))  # -2D
            qD = (q_sel == Int(3)(-1)).select(neg_D_64, qD)   # -1 * D
            qD = (q_sel == Int(3)(-2)).select(neg_2D_64, qD)  # -2 * D
            
            # ========== 更新部分余数 ==========
            # P_{i+1} = 4 * P_i - q_i * D = P_shifted - qD
            new_P = (P_shifted - qD).bitcast(Int(64))
            
            # ========== On-the-fly 商转换 ==========
            # 更新 Q+ 和 Q-
            # 如果 q >= 0: Q+ = Q+ * 4 + q, Q- = Q- * 4
            # 如果 q < 0:  Q+ = Q+ * 4, Q- = Q- * 4 + |q|
            Q_pos_shifted = (Q_pos << UInt(32)(2)).bitcast(UInt(32))
            Q_neg_shifted = (Q_neg << UInt(32)(2)).bitcast(UInt(32))
            
            # q 的绝对值（2位无符号）
            q_abs = UInt(2)(0)
            q_abs = (q_sel == Int(3)(2)).select(UInt(2)(2), q_abs)
            q_abs = (q_sel == Int(3)(1)).select(UInt(2)(1), q_abs)
            q_abs = (q_sel == Int(3)(0)).select(UInt(2)(0), q_abs)
            q_abs = (q_sel == Int(3)(-1)).select(UInt(2)(1), q_abs)
            q_abs = (q_sel == Int(3)(-2)).select(UInt(2)(2), q_abs)
            
            q_is_negative = (q_sel < Int(3)(0))
            
            new_Q_pos = q_is_negative.select(Q_pos_shifted, (Q_pos_shifted + q_abs.bitcast(UInt(32))).bitcast(UInt(32)))
            new_Q_neg = q_is_negative.select((Q_neg_shifted + q_abs.bitcast(UInt(32))).bitcast(UInt(32)), Q_neg_shifted)
            
            # DEBUG: 打印迭代状态
            # log("DIV ITER {}: P={:016x}, P_high={}, D={}, q={}, new_P={:016x}, Q+={}, Q-={}", 
            #     iter_num, current_P, P_high_32, D, q_sel, new_P, new_Q_pos, new_Q_neg)
            
            # 检查是否完成16次迭代（在写入之前计算）
            iter_done = (iter_num >= UInt(5)(15))
            new_iter_count = iter_num + UInt(5)(1)
            new_state = iter_done.select(UInt(6)(18), div_state_val + UInt(6)(1))
            
            # 更新寄存器
            div_remainder[0] = new_P
            div_quotient_pos[0] = new_Q_pos
            div_quotient_neg[0] = new_Q_neg
            div_iter_count[0] = new_iter_count
            div_state[0] = new_state
        
        # State 18: FINAL_CORRECTION - 最终修正
        with Condition((div_state_val == UInt(6)(18)) & ~start_new_div):
            saved_op = div_op_reg[0]
            final_P = div_remainder[0]
            Q_pos = div_quotient_pos[0]
            Q_neg = div_quotient_neg[0]
            quotient_sign = div_sign[0]
            remainder_sign = div_dividend_sign[0]
            divisor_orig = div_divisor[0]
            
            # 计算最终商: Q = Q+ - Q-
            quotient_raw = (Q_pos - Q_neg).bitcast(UInt(32))
            
            # 余数 = P 的高 32 位
            remainder_raw_unsigned = final_P[32:63].bitcast(UInt(32))
            remainder_raw_signed = remainder_raw_unsigned.bitcast(Int(32))
            
            # 如果最终余数为负，需要修正商和余数
            # 当 P < 0 时，商需要减1，余数需要加上除数
            P_negative = (final_P < Int(64)(0))
            
            # 如果 P 为负，余数需要加上除数
            divisor_signed = divisor_orig.bitcast(Int(32))
            remainder_if_negative = (remainder_raw_signed + divisor_signed).bitcast(UInt(32))
            remainder_after_neg_fix = P_negative.select(remainder_if_negative, remainder_raw_unsigned)
            quotient_after_neg_fix = P_negative.select(quotient_raw - UInt(32)(1), quotient_raw)
            
            # 额外修正：如果余数 >= 除数，需要增加商并减去除数
            # 这可能发生在最后一次迭代选择的 q 不够大时
            remainder_too_large = (remainder_after_neg_fix >= divisor_orig)
            quotient_corrected = remainder_too_large.select(quotient_after_neg_fix + UInt(32)(1), quotient_after_neg_fix)
            remainder_corrected = remainder_too_large.select(remainder_after_neg_fix - divisor_orig, remainder_after_neg_fix)
            
            # 应用符号修正
            # 商的符号
            quotient_signed_val = quotient_corrected.bitcast(Int(32))
            quotient_negated = (~quotient_signed_val + Int(32)(1)).bitcast(UInt(32))
            quotient_final = quotient_sign.select(quotient_negated, quotient_corrected)
            
            # 余数的符号（与被除数相同）
            remainder_signed_val = remainder_corrected.bitcast(Int(32))
            remainder_negated = (~remainder_signed_val + Int(32)(1)).bitcast(UInt(32))
            remainder_final = remainder_sign.select(remainder_negated, remainder_corrected)
            
            # 结果选择
            is_div_op = ((saved_op == UInt(3)(DIV_OP_DIV)) | (saved_op == UInt(3)(DIV_OP_DIVU))).select(UInt(1)(1), UInt(1)(0))
            final_div_result = is_div_op.select(quotient_final, remainder_final)
            
            # DEBUG: 打印最终结果
            # log("DIV FINAL: Q+={}, Q-={}, quotient_raw={}, P_negative={}, rem_too_large={}", 
            #     Q_pos, Q_neg, quotient_raw, P_negative, remainder_too_large)
            # log("DIV FINAL: P={:016x}, remainder_raw={}, quotient_final={}, remainder_final={}, result={}", 
            #     final_P, remainder_raw_unsigned, quotient_final, remainder_final, final_div_result)
            
            div_result_reg[0] = final_div_result
            div_valid[0] = UInt(1)(1)
            div_state[0] = UInt(6)(19)  # DONE
        
        # State 19: DONE - 完成
        with Condition((div_state_val == UInt(6)(19)) & ~start_new_div):
            div_state[0] = UInt(6)(0)  # IDLE
        
        # 非除法周期重置valid
        with Condition((div_state_val == UInt(6)(0)) & ~start_new_div):
            div_valid[0] = UInt(1)(0)
        
        # ==================== ALU结果选择 ====================
        # 普通ALU结果
        normal_alu_result = is_branch.select(UInt(XLEN)(0), (is_jump | is_jumpr).select(pc_in + UInt(XLEN)(4), self.alu_unit(alu_op, alu_a, alu_b)))
        
        # 乘法或除法完成时使用对应的结果
        # 优先级：div_done > mul_done > normal_alu_result
        div_result_val = div_result_reg[0]
        alu_result = div_done.select(div_result_val, mul_done.select(current_mul_result, normal_alu_result))
        log("EX RESULT: PC={:08x}, alu_op={:05b}, alu_a={:08x}, alu_b={:08x}, normal_alu_result={:08x}, final_alu_result={:08x}",
            pc_in, alu_op, alu_a, alu_b, normal_alu_result, alu_result)
        
        target_pc = (is_branch | is_jump).select(actual_target_pc, target_pc)
        target_pc = is_jumpr.select(new_pc.bitcast(UInt(32)), target_pc)
        
        # 需要刷新的情况: 预测错误、JAL、JALR
        need_flush = (mispredict | is_jump | is_jumpr).select(UInt(1)(1), UInt(1)(0))
        pc_change = need_flush

        # DEBUG: 检查跳转指令
        # with Condition(is_jumpr):
        #     log("EX JALR: PC={:08x}, rs1_data={:08x}, imm={:08x}, target={:08x}, rs1_idx={}", 
        #         pc_in, rs1_data, immediate_in, new_pc, rs1_idx)
        # with Condition(is_jump):
        #     log("EX JAL: PC={:08x}, imm={:08x}, target={:08x}, rd={}", 
        #         pc_in, immediate_in, actual_target_pc, rd_addr)
        # with Condition(is_branch):
        #     log("EX BRANCH: PC={:08x}, taken={}, target={:08x}, rs1={:08x}, rs2={:08x}",
        #         pc_in, actual_taken, actual_target_pc, rs1_data, rs2_data)

        # 旧停止指令检测 (JAL x0, 0)
        with Condition(is_jump & (immediate_in == UInt(XLEN)(0))):
            log("Finish Execution. The result is {}", reg_file[10])
            finish()
        
        # 新停止指令检测: sb x0, -1(x0) = 0xFE000FA3
        # 特征: mem_write=1, store_type=00(SB), rs1=0, rs2=0, immediate=-1
        # store_type_ex = control_in[22:23]
        # is_finish_inst = (mem_write & (store_type_ex == UInt(2)(0)) & 
        #                  (rs1_idx == UInt(5)(0)) & (rs2_idx == UInt(5)(0)) & 
        #                  (immediate_in == UInt(XLEN)(0xFFFFFFFF)))
        # with Condition(is_finish_inst):
        #     log("Finish Execution. The result is {}", reg_file[10])
        #     finish()
        
        # 乘法指令需要等待乘法完成才能传递到MEM阶段
        # 当乘法器正在执行(cycle 1或2)时，向MEM阶段传递NOP
        # 当乘法完成(cycle 3, mul_done=1)时，传递乘法结果
        mul_in_ex_stage = is_mul_inst & id_ex_valid[0]
        mul_wait = mul_in_ex_stage & ~mul_done  # 乘法未完成，需要等待
        
        # 除法指令需要等待除法完成才能传递到MEM阶段
        # 当除法器正在执行(state != 0 and state != 19)时，向MEM阶段传递NOP
        # 当除法完成(state = 19, div_done=1)时，传递除法结果
        div_in_ex_stage = is_div_inst & id_ex_valid[0]
        div_wait = div_in_ex_stage & ~div_done  # 除法未完成，需要等待
        
        # 当乘法完成时，使用保存的控制信息而不是当前的 control_in（因为当前可能是 NOP）
        mul_control = mul_control_reg[0]
        mul_pc = mul_pc_reg[0]
        
        # 当除法完成时，使用保存的控制信息而不是当前的 control_in（因为当前可能是 NOP）
        div_control = div_control_reg[0]
        div_pc = div_pc_reg[0]

        # 如果是乘法或除法指令且未完成，传递NOP；否则正常传递
        # 乘法或除法完成时，使用保存的控制信息
        should_pass = id_ex_valid[0] & ~mul_wait & ~div_wait
        pass_or_done = should_pass | mul_done | div_done  # 要么正常传递，要么完成
        log("EX PASS LOGIC: PC={:08x}, id_ex_valid={}, should_pass={}, mul_wait={}, div_wait={}, mul_done={}, div_done={}, pass_or_done={}",
            pc_in,
            id_ex_valid[0], should_pass, mul_wait, div_wait, mul_done, div_done, pass_or_done)

        # PC: 完成时用保存的 PC，否则用当前 PC
        final_pc = mul_done.select(mul_pc, div_done.select(div_pc, pc_in))
        # 控制信号: 完成时用保存的控制信号，否则用当前控制信号
        final_control = mul_done.select(mul_control, div_done.select(div_control, control_in))
        
        # **关键修复**: 不使用 Condition(ex_mem_valid[0]) 来控制流水线寄存器的更新
        # 原因：ex_mem_valid 是由 HazardUnit 在同一周期设置的，用于控制下一周期 MEM 是否执行。
        # 但 EX 阶段的流水线寄存器更新应该基于当前指令是否有效 (pass_or_done)，
        # 而不是基于下一周期 MEM 是否应该执行。
        # 
        # 之前的 bug：当 mem_sram_stall=1 时，HazardUnit 设置 ex_mem_valid[0]=0。
        # 这导致下一周期 EX 阶段的 Condition(ex_mem_valid[0]) 为 false，
        # 不会更新 ex_mem_control，导致 MEM 阶段看到的控制信号是旧的（rd=0, reg_write=0）。
        # 
        # 修复：总是更新流水线寄存器，根据 pass_or_done 决定写入有效值还是 0。
        ex_mem_pc[0] = pass_or_done.select(final_pc, UInt(XLEN)(0))
        ex_mem_control[0] = pass_or_done.select(final_control, UInt(CONTROL_LEN)(0))
        ex_mem_result[0] = pass_or_done.select(alu_result, UInt(XLEN)(0))
        ex_mem_data[0] = pass_or_done.select(rs2_data, UInt(XLEN)(0))
        log("EX DATA PATH: PC={:08x}, rs2_data={:08x}, pass_or_done={}, ex_mem_data={:08x}", pc_in, rs2_data, pass_or_done, ex_mem_data[0])
            
            # log("EX: PC={}, ALU_OP={:05b}, ALU_A={}, ALU_B={}, Result={:08x}, PC_Change={}, Target_PC={:08x}, Immediate={:08x}, ALU_SRC={}",
            #     pc_in, alu_op, alu_a, alu_b, alu_result, pc_change, target_pc, immediate_in, alu_src)
        
        memory_stage.async_called()

        # 构建预测结果:
        # [0]: mispredict (预测错误标志)
        # [1:32]: correct_pc (正确的PC)
        # [33]: actual_taken (实际跳转标志)
        # [34:65]: actual_target_pc (实际目标地址)
        # [66]: btb_hit
        # [67]: predict_taken
        # [68:99]: pc_in (分支指令PC，用于计算BTB索引)
        # [100]: is_branch
        # [101]: is_jump
        # [102]: is_jumpr
        prediction_result = concat(
            is_jumpr.bitcast(Bits(1)),           # [102] is_jumpr
            is_jump.bitcast(Bits(1)),            # [101] is_jump
            is_branch.bitcast(Bits(1)),          # [100] is_branch
            pc_in.bitcast(Bits(XLEN)),           # [99:68] 分支指令的PC
            predict_taken.bitcast(Bits(1)),      # [67] 预测是否跳转
            btb_hit.bitcast(Bits(1)),            # [66] BTB是否命中
            actual_target_pc.bitcast(Bits(XLEN)), # [65:34] 实际目标地址
            actual_taken.bitcast(Bits(1)),       # [33] 实际跳转标志
            correct_pc.bitcast(Bits(XLEN)),      # [32:1] 正确的PC
            mispredict.bitcast(Bits(1))          # [0] 预测错误标志
        )
        
        # 乘法器信号
        # mul_busy: 乘法器正在执行中 (cycle 1, 2)
        # mul_done: 乘法器完成 (cycle 3)
        # mul_stall: 当前有乘法指令但乘法器正在执行中，需要暂停
        mul_executing = ((mul_cycle == UInt(2)(1)) | (mul_cycle == UInt(2)(2))).select(UInt(1)(1), UInt(1)(0))
        mul_stall_needed = (is_mul_inst & id_ex_valid[0] & mul_executing).select(UInt(1)(1), UInt(1)(0))
        
        # 除法器信号
        # div_busy: 除法器正在执行中 (state != 0 and state != 19)
        # div_done: 除法器完成 (state = 19)
        # div_stall: 当前有除法指令但除法器正在执行中，需要暂停
        div_executing = ((div_state_val != UInt(6)(0)) & (div_state_val != UInt(6)(35))).select(UInt(1)(1), UInt(1)(0))
        div_stall_needed = (is_div_inst & id_ex_valid[0] & div_executing).select(UInt(1)(1), UInt(1)(0))

        # execute_signals 的生成逻辑:
        # 
        # **关键修正**: ex_mem_valid 的含义是 "MEM 阶段是否空闲可以接收新数据"
        # - ex_mem_valid=0 (下游暂停): MEM 阶段繁忙，EX 需要保持输出
        #   此时应该输出寄存器中保存的值（上一周期传递给 MEM 的值）
        # - ex_mem_valid=1: MEM 阶段空闲，可以接收新数据
        #   - pass_or_done=0 (EX无效或stall): 输出0（无有效操作）
        #   - pass_or_done=1 (正常): 输出新值
        # 
        # 寄存器更新逻辑:
        # - ex_mem_valid=0: 不更新寄存器（保持上一周期的值）
        # - ex_mem_valid=1, pass_or_done=1: 更新为新值
        # - ex_mem_valid=1, pass_or_done=0: 更新为0（清空）
        
        out_pc_change = ex_mem_valid[0].select(
            pass_or_done.select(pc_change, UInt(1)(0)),
            ex_mem_pc_change[0]
        )
        out_target_pc = ex_mem_valid[0].select(
            pass_or_done.select(target_pc, UInt(XLEN)(0)),
            ex_mem_target_pc[0]
        )
        out_control = ex_mem_valid[0].select(
            pass_or_done.select(control_in, UInt(CONTROL_LEN)(0)),
            ex_sig_control[0]
        )
        out_prediction_result = ex_mem_valid[0].select(
            pass_or_done.select(prediction_result.bitcast(UInt(103)), UInt(103)(0)),
            ex_mem_prediction_result[0]
        )
        
        log("EX SIGNALS: pass_or_done={}, ex_mem_valid={}, pc_change={}, out_pc_change={}, control_in={:012x}, ex_sig_control={:012x}, out_control={:012x}",
            pass_or_done, ex_mem_valid[0], pc_change, out_pc_change, control_in, ex_sig_control[0], out_control)
        
        # 更新寄存器
        # **关键修复**: 只在 ex_mem_valid=1 时更新 ex_sig_control
        # 当 ex_mem_valid=0 (stall期间)，保持 ex_sig_control 不变，以支持前递
        # MEM 阶段的操作通过 ex_mem_valid 来保护，不会重复执行
        with Condition(ex_mem_valid[0]):
            ex_mem_pc_change[0] = pass_or_done.select(pc_change, UInt(1)(0))
            ex_mem_target_pc[0] = pass_or_done.select(target_pc, UInt(XLEN)(0))
            ex_mem_prediction_result[0] = pass_or_done.select(prediction_result.bitcast(UInt(103)), UInt(103)(0))
            ex_sig_control[0] = pass_or_done.select(control_in, UInt(CONTROL_LEN)(0))
        # 注意：不再在 ex_mem_valid=0 时清除 ex_sig_control
        # 这样 MEM 阶段的前递信息（rd, reg_write）在 stall 期间保持有效
        
        execute_signals = concat(
            div_stall_needed.bitcast(Bits(1)),   # [182] 除法暂停信号
            div_done.bitcast(Bits(1)),           # [181] 除法完成
            div_busy.bitcast(Bits(1)),           # [180] 除法忙
            mul_stall_needed.bitcast(Bits(1)),   # [179] 乘法暂停信号
            mul_done.bitcast(Bits(1)),           # [178] 乘法完成
            mul_busy.bitcast(Bits(1)),           # [177] 乘法忙
            out_prediction_result.bitcast(Bits(103)),  # 预测结果
            out_control.bitcast(Bits(CONTROL_LEN)),
            out_target_pc.bitcast(Bits(XLEN)),       # [31:1]  目标PC
            out_pc_change.bitcast(Bits(1)),      # [0]     PC变化标志
        )

        return execute_signals
# ==================== MEM阶段：内存访问 ===================
class MemoryStage(Module):
    """内存访问阶段(MEM) - 冯诺依曼架构，返回SRAM请求信号"""
    def __init__(self):
        super().__init__(ports={})
    
    @module.combinational
    def build(self, ex_mem_valid, ex_mem_result, ex_mem_pc, ex_mem_data, ex_mem_control, mem_wb_control, mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, mem_wb_addr, sb_sh_state, sb_sh_addr, sb_sh_data, sb_sh_type, writeback_stage, unified_sram, mem_last_pc):
        """MEM阶段处理 - 冯诺依曼架构 (1-cycle Load)
        
        返回包含：
        - MEM阶段的SRAM访问请求（we, re, addr, wdata）
        - MEM是否正在使用SRAM（用于仲裁）
        - sb_sh_active（用于stall信号）
        
        注意：不在此处调用SRAM.build()，由build_cpu统一处理仲裁
        """
        pc_in = ex_mem_pc[0]
        addr_in = ex_mem_result[0]
        data_in = ex_mem_data[0]
        control_in = ex_mem_control[0]
        
        # 解析控制信号
        mem_read = control_in[5:5]
        mem_write = control_in[6:6]
        store_type = control_in[22:23]  # 存储类型: 00=SB, 01=SH, 10=SW
        
        word_addr = addr_in >> UInt(XLEN)(2)
        byte_offset = addr_in[0:1]  # 地址低2位
        
        # **关键**: 检测是否是新指令（PC 变化）来防止 load/SW 重复执行
        # 当 EX stall 时，同一条指令会在 MEM 阶段重复出现，PC 不变
        is_new_instruction = pc_in != mem_last_pc[0]
        log("MEM PC CHECK: pc_in={:08x}, last_pc={:08x}, is_new={}", pc_in, mem_last_pc[0], is_new_instruction)
        
        # DEBUG: Log memory operations
        with Condition(mem_read & ex_mem_valid[0]):
            log("MEM READ: addr={:08x}, word_addr={:08x}, byte_offset={}", addr_in, word_addr, byte_offset)
        with Condition(mem_write & ex_mem_valid[0]):
            log("MEM WRITE: addr={:08x}, word_addr={:08x}, byte_offset={}, data={:08x}", addr_in, word_addr, byte_offset, data_in)
        
        # store_type: 00=SB, 01=SH, 10=SW
        is_sb = (store_type == UInt(2)(0b00))
        is_sh = (store_type == UInt(2)(0b01))
        is_sw = (store_type == UInt(2)(0b10))
        needs_rmw = mem_write & (is_sb | is_sh)  # 需要读-修改-写
        
        # 准备写入的数据
        data_byte = data_in[0:7]
        data_half = data_in[0:15]
        
        # SB/SH状态机: 0=IDLE, 2=WRITE (状态1未使用)
        current_state = sb_sh_state[0]
        is_idle = (current_state == UInt(2)(0))
        is_write_phase = (current_state == UInt(2)(2))
        
        # 使用保存的地址和数据（用于WRITE阶段）
        saved_addr = sb_sh_addr[0]
        saved_data = sb_sh_data[0]
        saved_type = sb_sh_type[0]
        saved_word_addr = saved_addr >> UInt(XLEN)(2)
        saved_byte_offset = saved_addr[0:1]
        saved_data_byte = saved_data[0:7]
        saved_data_half = saved_data[0:15]
        
        # 构建SB的写入数据和掩码（使用保存的值）
        sb_data_saved = (saved_byte_offset == UInt(2)(0)).select(
            concat(UInt(24)(0), saved_data_byte),
            (saved_byte_offset == UInt(2)(1)).select(
                concat(UInt(16)(0), saved_data_byte, UInt(8)(0)),
                (saved_byte_offset == UInt(2)(2)).select(
                    concat(UInt(8)(0), saved_data_byte, UInt(16)(0)),
                    concat(saved_data_byte, UInt(24)(0))
                )
            )
        ).bitcast(UInt(XLEN))
        
        sb_mask_saved = (saved_byte_offset == UInt(2)(0)).select(
            UInt(XLEN)(0xFFFFFF00),
            (saved_byte_offset == UInt(2)(1)).select(
                UInt(XLEN)(0xFFFF00FF),
                (saved_byte_offset == UInt(2)(2)).select(
                    UInt(XLEN)(0xFF00FFFF),
                    UInt(XLEN)(0x00FFFFFF)
                )
            )
        )
        
        sh_data_saved = (saved_byte_offset[1:1] == UInt(1)(0)).select(
            concat(UInt(16)(0), saved_data_half),
            concat(saved_data_half, UInt(16)(0))
        ).bitcast(UInt(XLEN))
        
        sh_mask_saved = (saved_byte_offset[1:1] == UInt(1)(0)).select(
            UInt(XLEN)(0xFFFF0000),
            UInt(XLEN)(0x0000FFFF)
        )
        
        # 输出SB/SH是否正在进行中（用于stall信号）
        sb_sh_active = is_idle & ex_mem_valid[0] & needs_rmw
        
        # ===== 计算 SRAM 控制信号（但不调用build，由仲裁单元统一调用）=====
        # 获取上周期读取的原始数据（用于RMW WRITE阶段）
        original_data = unified_sram.dout[0]
        
        # 计算RMW写入数据
        is_sb_saved = (saved_type == UInt(2)(0b00))
        merged_sb = (original_data & sb_mask_saved) | sb_data_saved
        merged_sh = (original_data & sh_mask_saved) | sh_data_saved
        rmw_write_data = is_sb_saved.select(merged_sb, merged_sh)
        
        # ==================== Load 指令处理 (1-cycle latency) ====================
        # 冯诺依曼架构下，Load 仍然是 1-cycle latency：
        # - Cycle N: MEM 阶段检测到 Load，通过 Downstream (SramArbiter) 发送 SRAM 请求
        # - Cycle N+1: WB 阶段从 sram.dout 读取数据（通过 mem_wb_mem_data 传递）
        #
        # 关键点：SramArbiter 是 Downstream，同周期内完成 MEM → SramArbiter → SRAM
        # 所以 SRAM 在 Cycle N 收到请求，Cycle N+1 输出数据，WB 阶段可以读取
        
        # 检测是否有 Load 指令
        # **关键修复**: 使用 is_new_instruction 防止重复执行
        # 当 EX stall 时，同一条 load 会在 MEM 重复出现，但 PC 不变，不应再次发送请求
        is_load = is_idle & is_new_instruction & mem_read & ~mem_write & mem_wb_valid[0]
        
        # 确定操作类型
        do_rmw_write = is_write_phase & mem_wb_valid[0]  # RMW WRITE阶段
        # SB/SH RMW 开始：需要检查 ex_mem_valid，防止 stall 期间开始新的 RMW
        do_rmw_read = is_idle & ex_mem_valid[0] & needs_rmw & mem_wb_valid[0]  # SB/SH开始读
        # SW: 使用 is_new_instruction 防止重复执行
        do_sw_write = is_idle & is_new_instruction & mem_write & is_sw & mem_wb_valid[0]  # SW写入
        do_load_read = is_load  # Load 读取（1-cycle，直接发送请求）
        
        log("MEM LOAD STATE: is_load={}, do_load_read={}, is_new_inst={}", is_load, do_load_read, is_new_instruction)
        
        # 计算最终的 SRAM 控制信号
        # we: RMW WRITE 或 SW 写入时为1
        mem_sram_we = do_rmw_write | do_sw_write
        # re: RMW READ 或 Load 读取时为1
        mem_sram_re = do_rmw_read | do_load_read
        # addr: RMW WRITE用saved_word_addr，其他用word_addr
        word_addr_uint = word_addr.bitcast(UInt(XLEN))
        saved_word_addr_uint = saved_word_addr.bitcast(UInt(XLEN))
        mem_sram_addr = do_rmw_write.select(saved_word_addr_uint, word_addr_uint)
        # wdata: RMW WRITE用rmw_write_data，SW用data_in
        rmw_write_data_uint = rmw_write_data.bitcast(UInt(XLEN))
        mem_sram_wdata = do_rmw_write.select(rmw_write_data_uint, data_in)
        
        # MEM阶段是否正在发起SRAM请求（用于仲裁和stall）
        mem_sram_busy = mem_sram_we | mem_sram_re
        
        log("MEM SRAM SIGNALS: we={}, re={}, addr={:08x}, busy={}",
            mem_sram_we, mem_sram_re, mem_sram_addr, mem_sram_busy)
        
        # DEBUG logs
        with Condition(do_rmw_write):
            log("MEM RMW WRITE PHASE: original_data={:08x}, saved_data={:08x}", original_data, saved_data)
            log("MEM RMW WRITE: addr={:08x}, wdata={:08x}", saved_word_addr, rmw_write_data)
        with Condition(do_rmw_read):
            log("MEM RMW READ: word_addr={:08x}", word_addr)
        with Condition(do_sw_write):
            log("MEM WRITE SW: word_addr={:08x}, wdata={:08x}", word_addr, data_in)
        with Condition(do_load_read):
            log("MEM LOAD READ: word_addr={:08x}", word_addr)
        
        with Condition(mem_wb_valid[0]):
            # ===== WRITE阶段：更新状态机 =====
            with Condition(is_write_phase):
                # 返回IDLE状态
                sb_sh_state[0] = UInt(2)(0)
            
            # ===== 正常操作（IDLE状态）=====
            with Condition(is_idle & ex_mem_valid[0]):
                with Condition(needs_rmw):
                    # 保存当前指令信息
                    sb_sh_addr[0] = addr_in
                    sb_sh_data[0] = data_in
                    sb_sh_type[0] = store_type
                    # 进入WRITE阶段（下周期写入）
                    sb_sh_state[0] = UInt(2)(2)
            
            # **关键**: 更新 mem_last_pc 来记录本周期 MEM 执行的指令
            # 用于下一周期检测是否是新指令（防止 load/SW 重复执行）
            with Condition(mem_sram_busy):
                mem_last_pc[0] = pc_in
                log("MEM UPDATE LAST PC: {:08x}", pc_in)
            
            # 更新MEM/WB流水线寄存器
            # Load 指令：正常传递控制信号，WB阶段会直接从 sram.dout 读取数据
            # 非 Load 指令：也正常传递
            # **关键修复**: 不再根据 ex_mem_valid 来决定是否传递控制信号
            # ex_mem_valid 控制的是 MEM 是否执行新的 load/store，不是是否传递到 WB
            # MEM 阶段的当前指令（control_in）总是应该传递到 WB
            with Condition((is_idle & ~needs_rmw) | is_write_phase):
                mem_wb_control[0] = control_in
                mem_wb_ex_result[0] = ex_mem_result[0]
                mem_wb_addr[0] = addr_in
            
            # SB/SH 开始时清零 MEM/WB 寄存器（RMW 第一周期不能写回）
            with Condition(is_idle & needs_rmw):
                mem_wb_control[0] = UInt(CONTROL_LEN)(0)  # 不写回寄存器
                mem_wb_ex_result[0] = UInt(XLEN)(0)
                mem_wb_addr[0] = UInt(XLEN)(0)

        writeback_stage.async_called()

        # 返回memory_signals，包含MEM的SRAM访问请求和sb_sh_active
        # 布局 (简化版，移除 load_completing):
        # [0:CONTROL_LEN-1] control_in (48位)
        # [CONTROL_LEN] sb_sh_active (1位)
        # [CONTROL_LEN+1] mem_sram_busy (1位)
        # [CONTROL_LEN+2] mem_sram_we (1位)
        # [CONTROL_LEN+3] mem_sram_re (1位)
        # [CONTROL_LEN+4:CONTROL_LEN+35] mem_sram_addr (32位)
        # [CONTROL_LEN+36:CONTROL_LEN+67] mem_sram_wdata (32位)
        MEMORY_SIGNALS_LEN = CONTROL_LEN + 2 + 2 + XLEN + XLEN  # 48 + 4 + 64 = 116
        memory_signals = concat(
            mem_sram_wdata.bitcast(Bits(XLEN)),  # [115:84] wdata
            mem_sram_addr.bitcast(Bits(XLEN)),   # [83:52] addr
            mem_sram_re.bitcast(Bits(1)),        # [51] re
            mem_sram_we.bitcast(Bits(1)),        # [50] we
            mem_sram_busy.bitcast(Bits(1)),      # [49] mem_sram_busy
            sb_sh_active.bitcast(Bits(1)),       # [48] sb_sh_active
            control_in.bitcast(Bits(CONTROL_LEN))  # [47:0] control
        ).bitcast(Bits(MEMORY_SIGNALS_LEN))
        return memory_signals

# ==================== WB阶段：写回 ===================
class WriteBackStage(Module):
    """写回阶段(WB) - 1-cycle Load: 直接从 unified_sram.dout 读取数据"""
    def __init__(self):
        super().__init__(ports={})
    
    @module.combinational
    def build(self, mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, mem_wb_control, mem_wb_addr, reg_file, unified_sram):
        # 冯诺依曼架构 1-cycle Load:
        # - MEM 阶段（Cycle N）发送 SRAM 读取请求
        # - WB 阶段（Cycle N+1）直接从 unified_sram.dout 读取数据
        # 
        # 这是因为 SRAM 有 1-cycle latency：
        # - Cycle N: SRAM 收到 re=1, addr=xxx
        # - Cycle N+1: SRAM 输出 dout = data[xxx]
        
        # 直接从 SRAM 的 dout 读取 Load 数据
        mem_data_in = unified_sram.dout[0]
        ex_result_in = mem_wb_ex_result[0]
        control_in = mem_wb_control[0]
        addr_in = mem_wb_addr[0]
        
        # 解析控制信号
        reg_write = control_in[7:7]
        mem_to_reg = control_in[8:8]
        wb_rd = control_in[25:29]
        load_type = control_in[14:16]  # load类型: 000=LB, 001=LH, 010=LW, 100=LBU, 101=LHU
        
        # 计算字节偏移 (地址的低2位)
        byte_offset = addr_in[0:1]  # 2位偏移
        
        # 使用公共函数处理load数据
        # 内联处理load数据 (WriteBackStage)
        wbs_byte0 = mem_data_in[0:7]
        wbs_byte1 = mem_data_in[8:15]
        wbs_byte2 = mem_data_in[16:23]
        wbs_byte3 = mem_data_in[24:31]
        wbs_selected_byte = (byte_offset == UInt(2)(0)).select(wbs_byte0,
                            (byte_offset == UInt(2)(1)).select(wbs_byte1,
                            (byte_offset == UInt(2)(2)).select(wbs_byte2, wbs_byte3)))
        wbs_half0 = mem_data_in[0:15]
        wbs_half1 = mem_data_in[16:31]
        wbs_selected_half = (byte_offset[1:1] == UInt(1)(0)).select(wbs_half0, wbs_half1)
        wbs_byte_sign = wbs_selected_byte[7:7]
        wbs_lb_data = concat(wbs_byte_sign.select(UInt(24)(0xFFFFFF), UInt(24)(0)), wbs_selected_byte).bitcast(UInt(XLEN))
        wbs_lbu_data = concat(UInt(24)(0), wbs_selected_byte).bitcast(UInt(XLEN))
        wbs_half_sign = wbs_selected_half[15:15]
        wbs_lh_data = concat(wbs_half_sign.select(UInt(16)(0xFFFF), UInt(16)(0)), wbs_selected_half).bitcast(UInt(XLEN))
        wbs_lhu_data = concat(UInt(16)(0), wbs_selected_half).bitcast(UInt(XLEN))
        wbs_lw_data = mem_data_in
        processed_mem_data = (load_type == UInt(3)(0b000)).select(wbs_lb_data,
                             (load_type == UInt(3)(0b001)).select(wbs_lh_data,
                             (load_type == UInt(3)(0b010)).select(wbs_lw_data,
                             (load_type == UInt(3)(0b100)).select(wbs_lbu_data, wbs_lhu_data))))
        
        log("WB LOAD PROCESS: mem_data_in={:08x}, load_type={:03b}, processed_mem_data={:08x}",
            mem_data_in, load_type, processed_mem_data)
            
        # 选择写回数据
        wb_data = mem_to_reg.select(processed_mem_data, ex_result_in)
        
        log("WB STAGE: ex_result_in={:08x}, mem_to_reg={}, wb_data={:08x}, wb_rd={}, reg_write={}, load_type={:03b}",
            ex_result_in, mem_to_reg, wb_data, wb_rd, reg_write, load_type)
            
        # 如果指令无效，直接返回
        with Condition(mem_wb_valid[0]):
            with Condition(reg_write):
                log("WB WRITE: reg[{}] = {:08x}", wb_rd, wb_data)
                # 特别跟踪 sp (x2) 的写入
                with Condition(wb_rd == UInt(5)(2)):
                    log("WB WRITE SP: sp = {:08x}", wb_data)
                reg_file[wb_rd] = wb_data

        writeback_signals = control_in.bitcast(Bits(CONTROL_LEN))
        return writeback_signals

class SramArbiter(Downstream):
    """SRAM仲裁单元 - 冯诺依曼架构中协调IF和MEM对SRAM的访问
    
    1-cycle Load 时序：
    - Cycle N: MEM 发送 Load 请求 → SramArbiter (Downstream, 无延迟) → SRAM.build(re=1, addr=...)
    - Cycle N+1: WB 从 SRAM.dout 读取数据
    
    仲裁规则: MEM 优先，IF 在 MEM 不用时才能访问
    """
    def __init__(self):
        super().__init__()
    
    @downstream.combinational
    def build(self, fetch_signals, memory_signals, unified_sram):
        """SRAM仲裁逻辑
        
        Args:
            fetch_signals: IF阶段返回的信号
                布局: [31:0]=instruction, [32]=if_needs_sram, [64:33]=if_sram_addr
            memory_signals: MEM阶段返回的信号 (包含SRAM请求)
                布局: [47:0]=control, [48]=sb_sh_active, [49]=mem_sram_busy, 
                      [50]=we, [51]=re, [83:52]=addr, [115:84]=wdata
            unified_sram: 统一SRAM实例
        """
        # 定义信号长度 (移除了 load_completing)
        FETCH_SIGNALS_LEN = XLEN + 1 + XLEN  # 65
        MEMORY_SIGNALS_LEN = CONTROL_LEN + 2 + 2 + XLEN + XLEN  # 116
        
        fetch_signals = fetch_signals.optional(Bits(FETCH_SIGNALS_LEN)(0))
        memory_signals = memory_signals.optional(Bits(MEMORY_SIGNALS_LEN)(0))
        
        # 从fetch_signals提取IF阶段的SRAM请求
        if_needs_sram = fetch_signals[XLEN:XLEN].bitcast(UInt(1))
        if_sram_addr = fetch_signals[XLEN + 1:XLEN * 2].bitcast(UInt(XLEN))
        
        # 从memory_signals提取MEM阶段的SRAM控制信号
        mem_sram_busy = memory_signals[CONTROL_LEN + 1:CONTROL_LEN + 1].bitcast(UInt(1))
        mem_sram_we = memory_signals[CONTROL_LEN + 2:CONTROL_LEN + 2].bitcast(UInt(1))
        mem_sram_re = memory_signals[CONTROL_LEN + 3:CONTROL_LEN + 3].bitcast(UInt(1))
        mem_sram_addr = memory_signals[CONTROL_LEN + 4:CONTROL_LEN + 35].bitcast(UInt(XLEN))
        mem_sram_wdata = memory_signals[CONTROL_LEN + 36:CONTROL_LEN + 67].bitcast(UInt(XLEN))
        
        # SRAM仲裁：MEM优先，IF在MEM不用时才能访问
        final_sram_we = mem_sram_we
        final_sram_re = mem_sram_busy.select(mem_sram_re, if_needs_sram)
        final_sram_addr = mem_sram_busy.select(mem_sram_addr, if_sram_addr)
        final_sram_wdata = mem_sram_wdata
        
        log("SRAM_ARBITER: mem_busy={}, mem_we={}, mem_re={}, if_needs={}, final_re={}, final_addr={:08x}",
            mem_sram_busy, mem_sram_we, mem_sram_re, if_needs_sram, final_sram_re, final_sram_addr)
        
        # 统一调用SRAM.build - 作为 Downstream，这在同一周期内完成
        unified_sram.build(
            we=final_sram_we,
            re=final_sram_re,
            addr=final_sram_addr,
            wdata=final_sram_wdata
        )

class HazardUnit(Downstream):
    """冒险检测单元 - 包含分支预测器更新逻辑"""
    def __init__(self):
        super().__init__()

    @downstream.combinational
    def build(self, pc, stall, if_id_valid, if_id_pc, if_id_instruction, if_id_prediction_info, id_ex_pc, id_ex_control, id_ex_valid, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, id_ex_prediction_info, id_ex_need_rs1, id_ex_need_rs2, ex_mem_valid, mem_wb_valid, btb, bht, btb_valid, fetch_signals, decode_signals, execute_signals, memory_signals, writeback_signals, mul_in_progress, mul_cycle_counter, mul_rd_reg, div_state, div_iter_count, div_rd_reg):

        # 计算新的信号长度 (增加3位乘法信号和3位除法信号)
        # pc_change(1) + target_pc(32) + control(48) + prediction_result(103) + mul_signals(3) + div_signals(3) = 190
        EXECUTE_SIGNALS_LEN = XLEN + 1 + CONTROL_LEN + 103 + 6  # 32 + 1 + 48 + 103 + 6 = 190
        DECODE_SIGNALS_LEN = 2 + CONTROL_LEN + 5 + 5 + XLEN + PREDICTION_INFO_LEN  # need_rs1(1) + need_rs2(1) + control(45) + rs1(5) + rs2(5) + immediate(32) + prediction_info(34)
        # 冯诺依曼架构：fetch_signals 包含指令和IF的SRAM请求信号
        # 布局: [31:0]=instruction, [32]=if_needs_sram, [64:33]=if_sram_addr
        FETCH_SIGNALS_LEN = XLEN + 1 + XLEN  # 32 + 1 + 32 = 65

        execute_signals = execute_signals.optional(Bits(EXECUTE_SIGNALS_LEN)(0))
        decode_signals = decode_signals.optional(Bits(DECODE_SIGNALS_LEN)(0))
        fetch_signals = fetch_signals.optional(Bits(FETCH_SIGNALS_LEN)(0))
        # 冯诺依曼架构：memory_signals 包含 SRAM 控制信号
        # 布局: [47:0]=control, [48]=sb_sh_active, [49]=mem_sram_busy, [50]=we, [51]=re, [83:52]=addr, [115:84]=wdata
        MEMORY_SIGNALS_LEN = CONTROL_LEN + 2 + 2 + XLEN + XLEN  # 48 + 4 + 64 = 116
        memory_signals = memory_signals.optional(Bits(MEMORY_SIGNALS_LEN)(0))
        writeback_signals = writeback_signals.optional(Bits(CONTROL_LEN)(0))

        # 解析execute_signals
        pc_change = execute_signals[0:0].bitcast(UInt(1))
        target_pc = execute_signals[1:XLEN].bitcast(UInt(XLEN))
        
        # 解析预测结果 (从execute_signals中提取)
        # execute_signals布局: [0]: pc_change, [1:32]: target_pc, [33:80]: control(48), [81:183]: prediction_result(103), [184:186]: mul_signals(3), [187:189]: div_signals(3)
        pred_result_start = XLEN + 1 + CONTROL_LEN
        prediction_result = execute_signals[pred_result_start:pred_result_start + 102].bitcast(UInt(103))
        
        # 解析乘法器信号
        mul_signals_start = pred_result_start + 103
        mul_busy_sig = execute_signals[mul_signals_start:mul_signals_start].bitcast(UInt(1))
        mul_done_sig = execute_signals[mul_signals_start + 1:mul_signals_start + 1].bitcast(UInt(1))
        mul_stall_sig = execute_signals[mul_signals_start + 2:mul_signals_start + 2].bitcast(UInt(1))
        
        # 解析除法器信号
        div_signals_start = mul_signals_start + 3
        div_busy_sig = execute_signals[div_signals_start:div_signals_start].bitcast(UInt(1))
        div_done_sig = execute_signals[div_signals_start + 1:div_signals_start + 1].bitcast(UInt(1))
        div_stall_sig = execute_signals[div_signals_start + 2:div_signals_start + 2].bitcast(UInt(1))
        
        # 解析prediction_result:
        # [0]: mispredict, [1:32]: correct_pc, [33]: actual_taken, [34:65]: actual_target_pc
        # [66]: btb_hit, [67]: predict_taken, [68:99]: pc_in, [100]: is_branch, [101]: is_jump, [102]: is_jumpr
        mispredict = prediction_result[0:0].bitcast(UInt(1))
        correct_pc = prediction_result[1:32].bitcast(UInt(XLEN))
        actual_taken = prediction_result[33:33].bitcast(UInt(1))
        actual_target_pc = prediction_result[34:65].bitcast(UInt(XLEN))
        pred_btb_hit = prediction_result[66:66].bitcast(UInt(1))
        pred_predict_taken = prediction_result[67:67].bitcast(UInt(1))
        branch_pc = prediction_result[68:99].bitcast(UInt(XLEN))
        is_branch_ex = prediction_result[100:100].bitcast(UInt(1))
        is_jump_ex = prediction_result[101:101].bitcast(UInt(1))
        is_jumpr_ex = prediction_result[102:102].bitcast(UInt(1))
        
        # 从fetch_signals中提取指令部分
        # fetch_signals布局: [31:0]=instruction, [32]=if_needs_sram, [64:33]=if_sram_addr
        instruction = fetch_signals[0:XLEN - 1].bitcast(UInt(XLEN))
        
        # 解析decode_signals (新布局)
        control_in = decode_signals[0:CONTROL_LEN - 1].bitcast(UInt(CONTROL_LEN))
        rs1 = decode_signals[CONTROL_LEN:CONTROL_LEN + 4].bitcast(UInt(5))
        rs2 = decode_signals[CONTROL_LEN + 5:CONTROL_LEN + 9].bitcast(UInt(5))
        immediate = decode_signals[CONTROL_LEN + 10:CONTROL_LEN + 10 + XLEN - 1].bitcast(UInt(XLEN))
        needs_rs1 = decode_signals[CONTROL_LEN + 10 + XLEN:CONTROL_LEN + 10 + XLEN].bitcast(UInt(1))
        needs_rs2 = decode_signals[CONTROL_LEN + 10 + XLEN + 1:CONTROL_LEN + 10 + XLEN + 1].bitcast(UInt(1))
        prediction_info_id = decode_signals[CONTROL_LEN + 10 + XLEN + 2:CONTROL_LEN + 10 + XLEN + 2 + PREDICTION_INFO_LEN - 1].bitcast(UInt(PREDICTION_INFO_LEN))

        # 解析 execute_signals 中的控制信号（用于即将进入MEM阶段的指令，用于转发判断）
        ex_out_control = execute_signals[XLEN + 1:XLEN + CONTROL_LEN].bitcast(UInt(CONTROL_LEN))
        log("HAZARD ex_out_control: id_ex_valid={}, ex_out_control={:012x}",
            id_ex_valid[0], ex_out_control)
        
        # 解析 memory_signals，获取 MEM 阶段当前指令的控制信号
        # [0:CONTROL_LEN-1] = control (48位) - 来自 ex_mem_control[0]，是MEM阶段流水线寄存器
        # [CONTROL_LEN] = sb_sh_active (1位)
        # [CONTROL_LEN+1] = mem_sram_busy (1位)
        # [CONTROL_LEN+2] = we, [CONTROL_LEN+3] = re, [CONTROL_LEN+4:35] = addr, [CONTROL_LEN+36:67] = wdata
        mem_sig_control = memory_signals[0:CONTROL_LEN-1].bitcast(UInt(CONTROL_LEN))
        sb_sh_stall = memory_signals[CONTROL_LEN:CONTROL_LEN].bitcast(UInt(1))
        mem_sram_busy_sig = memory_signals[CONTROL_LEN + 1:CONTROL_LEN + 1].bitcast(UInt(1))  # MEM是否正在发起SRAM请求
        
        # **重要修复**: 使用 mem_sig_control（来自 ex_mem_control[0]）来检测 Load-Use 冒险
        # 之前错误地使用了 execute_signals 中的控制信号，这是 EX 阶段输出的下一条指令的信号
        # 而我们需要的是 MEM 阶段当前正在执行的指令的控制信号
        # 
        # **进一步修复**: 不再用 ex_mem_valid 来 mask mem_stage_control
        # ex_mem_valid 控制的是 EX→MEM 的传递，不是 MEM 阶段当前持有的指令的有效性
        # MEM 阶段的控制信号总是有效的（来自 ex_mem_control[0]）
        mem_stage_control = mem_sig_control
        rd_mem = mem_stage_control[25:29]
        reg_write_mem = mem_stage_control[7:7]
        mem_read_mem = mem_stage_control[5:5]  # MEM 阶段当前指令是否为 Load
        log("HAZARD memory_control: id_ex_valid={}, memory_control={:012x}, rd_mem={}, reg_write_mem={}, mem_read_mem={}",
            id_ex_valid[0], mem_stage_control, rd_mem, reg_write_mem, mem_read_mem)
        
        # WB 阶段控制信号（保持原有逻辑）
        wb_control = mem_stage_control  # wb_control 与 mem_stage_control 相同（用于 WB 阶段转发）
        rd_wb = wb_control[25:29]
        reg_write_wb = wb_control[7:7]
        mem_read_wb = wb_control[5:5]  # WB 阶段的 mem_read 信号
        
        # ==================== Load-Use 冒险检测 ====================
        # 只有 Load-Use 冒险需要暂停，其他数据冒险通过 bypass/forwarding 解决
        # Load-Use 冒险：MEM 阶段为 Load 指令（mem_read=1）且目标寄存器与 ID 阶段源寄存器相同
        # 1-cycle Load: MEM 阶段发送 SRAM 请求，WB 阶段才有数据，所以需要检测 Load-Use 冒险
        # 
        # **关键修复**: 只有当 MEM 阶段真的在执行 load（mem_sram_busy=1）时，才检测 load-use hazard
        # 当 is_new_instruction=0 时，load 不执行，mem_sram_busy=0，不应该检测 hazard
        # 因为此时 load 的结果已经在 WB 阶段可用（可以前递）
        load_use_hazard_mem = (mem_sram_busy_sig & mem_read_mem & reg_write_mem & (rd_mem != UInt(5)(0)) & 
                               ((needs_rs1 & (rs1 == rd_mem)) | (needs_rs2 & (rs2 == rd_mem))))
        log("LOAD_USE_HAZARD_MEM: mem_sram_busy={}, mem_read_mem={}, reg_write_mem={}, rd_mem={}, rs1={}, rs2={}, needs_rs1={}, needs_rs2={}, rs1==rd_mem={}, rs2==rd_mem={}, result={}",
            mem_sram_busy_sig, mem_read_mem, reg_write_mem, rd_mem, rs1, rs2, needs_rs1, needs_rs2, (rs1 == rd_mem), (rs2 == rd_mem), load_use_hazard_mem)
        
        # ==================== Von Neumann SRAM 冲突检测 ====================
        # 当 MEM 阶段发起 SRAM 请求时（mem_sram_busy=1），IF 无法取指，需要暂停 IF/ID/EX
        mem_sram_stall = mem_sram_busy_sig
        log("VON_NEUMANN_STALL: mem_sram_busy={}, mem_sram_stall={}", 
            mem_sram_busy_sig, mem_sram_stall)
        
        # WB 阶段 Load-Use 冒险（理论上通过前递可以解决，但作为安全检测保留）
        load_use_hazard_wb = (mem_read_wb & reg_write_wb & (rd_wb != UInt(5)(0)) & 
                              ((needs_rs1 & (rs1 == rd_wb)) | (needs_rs2 & (rs2 == rd_wb))))
        
        # ==================== 乘法冒险检测 ====================
        # 检测EX阶段是否有乘法指令
        ex_control = id_ex_control[0]
        ex_rd = ex_control[25:29]
        ex_mul_op = ex_control[42:44]
        ex_div_op = ex_control[45:47]
        is_ex_mul = (ex_mul_op != UInt(3)(MUL_OP_NONE))
        is_ex_div = (ex_div_op != UInt(3)(DIV_OP_NONE))
        
        # 乘法暂停条件：
        # 乘法器正在执行中(cycle 1, 2, 或 3)，需要暂停IF/ID阶段
        # cycle 3 (mul_done) 时也需要暂停，因为结果还在 MEM/WB 阶段传递
        mul_cycle = mul_cycle_counter[0]
        # 包含 cycle 1, 2, 3 - 只有 cycle 0 时才不暂停
        mul_executing = (mul_cycle != UInt(2)(0)).select(UInt(1)(1), UInt(1)(0))
        
        # 检测乘法结果冒险：ID阶段的指令依赖于正在执行的乘法结果
        # 注意：只有在乘法器实际在执行（mul_cycle != 0）或者乘法刚开始（mul_in_progress=1）时才检测
        # 这是为了避免在 id_ex_valid=0 但 id_ex_control 还保留旧 MUL 指令时产生误判
        # 条件：(乘法器正在执行中 且 rd != 0 且 ID阶段指令依赖于rd) 或者
        #       (EX阶段有新的MUL指令 且 id_ex_valid=1 且 rd != 0 且 ID阶段指令依赖于rd)
        mul_in_progress_val = mul_in_progress[0]
        mul_rd = mul_rd_reg[0]  # 正在执行的乘法的目标寄存器
        
        # 乘法器执行中的冒险检测（使用保存的 rd）
        mul_executing_hazard = (mul_executing & (mul_rd != UInt(5)(0)) &
                                ((needs_rs1 & (rs1 == mul_rd)) | (needs_rs2 & (rs2 == mul_rd))))
        
        # 新 MUL 指令的冒险检测（只在 id_ex_valid=1 时有效）
        # 注意：这里使用 id_ex_valid[0]，它是上一周期设置的值
        # 如果当前正在暂停（data_hazard=1），则 id_ex_valid[0] 已经是 0，不会再次检测
        mul_new_inst_hazard = (is_ex_mul & id_ex_valid[0] & (ex_rd != UInt(5)(0)) &
                              ((needs_rs1 & (rs1 == ex_rd)) | (needs_rs2 & (rs2 == ex_rd))))
        
        mul_result_hazard = mul_executing_hazard | mul_new_inst_hazard
        
        # ==================== 除法冒险检测 ====================
        # 检测EX阶段是否有除法指令
        div_cycle = div_iter_count[0]
        # 除法器状态：0=IDLE, 1=INIT, 2-17=ITERATE, 18=FINAL_CORRECTION, 19=DONE
        # 除法器执行中：state != 0 (IDLE)
        div_state_val = div_state[0]
        div_executing = (div_state_val != UInt(6)(0)).select(UInt(1)(1), UInt(1)(0))
        
        # 检测除法结果冒险：ID阶段的指令依赖于正在执行的除法结果
        # 与乘法类似，需要检查除法器是否正在执行以及是否有新的 DIV 指令
        div_rd = div_rd_reg[0]  # 正在执行的除法的目标寄存器
        
        # 除法器执行中的冒险检测（使用保存的 rd）
        div_executing_hazard = (div_executing & (div_rd != UInt(5)(0)) &
                                ((needs_rs1 & (rs1 == div_rd)) | (needs_rs2 & (rs2 == div_rd))))
        
        # 新 DIV 指令的冒险检测（只在 id_ex_valid=1 时有效）
        div_new_inst_hazard = (is_ex_div & id_ex_valid[0] & (ex_rd != UInt(5)(0)) &
                              ((needs_rs1 & (rs1 == ex_rd)) | (needs_rs2 & (rs2 == ex_rd))))
        
        div_result_hazard = div_executing_hazard | div_new_inst_hazard
        
        # 需要刷新的情况: mispredict || is_jump || is_jumpr
        need_flush = (mispredict | is_jump_ex | is_jumpr_ex).select(UInt(1)(1), UInt(1)(0))
        
        # 综合暂停逻辑：
        # 1. Load-Use 冒险
        # 2. 乘法器执行中（cycle 1或2，需要等待乘法完成）
        # 3. 乘法结果冒险（下一条指令依赖乘法结果）
        # 4. 除法器执行中（state != 0，需要等待除法完成）
        # 5. 除法结果冒险（下一条指令依赖除法结果）
        # 6. Von Neumann SRAM 冲突：MEM 发起 SRAM 请求时暂停 IF/ID/EX（Load 完成时不暂停）
        data_hazard = ((load_use_hazard_mem | mul_executing | mul_result_hazard | div_executing | div_result_hazard | sb_sh_stall | mem_sram_stall) & ~need_flush)
        log("HAZARD: data_hazard={}, load_use_hazard_mem={}, sb_sh_stall={}, mem_sram_stall={}, need_flush={}",
            data_hazard, load_use_hazard_mem, sb_sh_stall, mem_sram_stall, need_flush)
        log("HAZARD DETAIL: rd_mem={}, rs1={}, rs2={}, needs_rs1={}, needs_rs2={}, mem_read_mem={}, reg_write_mem={}",
            rd_mem, rs1, rs2, needs_rs1, needs_rs2, mem_read_mem, reg_write_mem)
        log("HAZARD MUL/DIV: mul_cycle={}, mul_executing={}, mul_result_hazard={}, mul_exec_haz={}, mul_new_haz={}, div_state={}, div_executing={}, div_result_hazard={}",
            mul_cycle, mul_executing, mul_result_hazard, mul_executing_hazard, mul_new_inst_hazard, div_state_val, div_executing, div_result_hazard)
        
        # id_ex_valid 的含义：EX阶段是否有有效指令需要执行
        # - need_flush时，EX阶段指令作废，设为0
        # - data_hazard时，EX阶段指令仍然有效（只是暂停），保持当前值
        # - 正常时，从 if_id_valid 传递（模拟流水线行为）
        # 
        # **关键修复**：正常情况下从 if_id_valid 传递，而不是强制设为1！
        # 原因：flush后，if_id_valid被设为0，下一周期id_ex_valid应该继承这个0，
        #       而不是强制设为1导致被flush的指令又被执行。
        # 这修复了分支指令被重复执行的bug。
        id_ex_valid[0] = need_flush.select(UInt(1)(0), data_hazard.select(id_ex_valid[0], if_id_valid[0]))
        if_id_valid[0] = need_flush.select(UInt(1)(0), data_hazard.select(if_id_valid[0], UInt(1)(1)))
        # ex_mem_valid: 
        # **关键修复**: ex_mem_valid 只用于控制 SB/SH 的 RMW 逻辑
        # - SB/SH stall时设为0（RMW操作进行中），防止新的 SB/SH 开始
        # - 不再因 mem_sram_stall 设为0，因为:
        #   1. Load/SW 是单周期操作，不需要 ex_mem_valid 防止重复执行
        #   2. SB/SH RMW 期间，is_idle=0 已经防止了 load/SW 的误执行
        ex_mem_valid[0] = (~sb_sh_stall)
        mem_wb_valid[0] = UInt(1)(1)
        stall[0] = data_hazard
        nop_control = UInt(CONTROL_LEN)(0)

        # BTB索引计算
        btb_update_index = branch_pc[2:7].bitcast(UInt(BTB_INDEX_BITS))
        
        # 分支预测器更新逻辑 (仅在is_branch == 1时更新)
        # 根据branch_prediction_rules.md:
        # - 更新BTB: btb[index] = actual_target_pc, btb_valid[index] = 1
        # - 更新BHT (2-bit饱和计数器):
        #   - actual_taken == 1: bht[index] = (bht[index] == 3) ? 3 : bht[index] + 1
        #   - actual_taken == 0: bht[index] = (bht[index] == 0) ? 0 : bht[index] - 1
        current_bht = bht[btb_update_index]
        new_bht_taken = (current_bht == UInt(2)(3)).select(UInt(2)(3), current_bht + UInt(2)(1))
        new_bht_not_taken = (current_bht == UInt(2)(0)).select(UInt(2)(0), current_bht - UInt(2)(1))
        new_bht = actual_taken.select(new_bht_taken, new_bht_not_taken)
        
        with Condition(is_branch_ex):
            btb[btb_update_index] = actual_target_pc
            btb_valid[btb_update_index] = UInt(1)(1)
            bht[btb_update_index] = new_bht

        # PC更新逻辑 (根据branch_prediction_rules.md)
        # need_flush == 1:
        #   - JALR指令: pc[0] = (rs1_data + immediate_in) & ~1 (已在target_pc中计算)
        #   - 其他情况: pc[0] = correct_pc
        # need_flush == 0:
        #   - 数据冒险: pc[0] = pc[0] (保持不变)
        #   - 无数据冒险: 
        #     - 如果有预测且预测跳转，使用预测的PC
        #     - 否则 pc[0] = pc[0] + 4
        
        # 从IF阶段获取当前指令的预测信息
        current_btb_hit = if_id_prediction_info[0][0:0].bitcast(UInt(1))
        current_predict_taken = if_id_prediction_info[0][1:1].bitcast(UInt(1))
        current_predicted_pc = if_id_prediction_info[0][2:33].bitcast(UInt(XLEN))
        
        # BTB预测跳转标志：当ID阶段的指令有BTB预测且预测跳转时
        # 需要清空IF/ID中已经取到的顺序指令（因为PC已经被更新为预测目标）
        btb_predict_taken = current_btb_hit & current_predict_taken
        
        # 正常情况下的下一个PC
        # 如果BTB命中且预测跳转，使用预测的目标PC
        normal_next_pc = btb_predict_taken.select(current_predicted_pc, pc[0] + UInt(XLEN)(4))
        
        # PC更新
        # JALR时使用target_pc (因为在EX阶段已经计算为 (rs1 + imm) & ~1)
        # JAL时使用actual_target_pc (pc + immediate)
        # 分支误预测时使用correct_pc
        flush_pc = is_jumpr_ex.select(target_pc, is_jump_ex.select(actual_target_pc, correct_pc))
        
        # PC更新
        # JALR时使用target_pc (因为在EX阶段已经计算为 (rs1 + imm) & ~1)
        flush_pc = is_jumpr_ex.select(target_pc, is_jump_ex.select(actual_target_pc, correct_pc))
        pc[0] = need_flush.select(flush_pc, data_hazard.select(pc[0], normal_next_pc))
        
        # 流水线刷新 (根据branch_prediction_rules.md)
        # IF/ID阶段刷新: if_id_valid[0] = 0, if_id_pc[0] = 0, if_id_instruction[0] = NOP
        # ID/EX阶段刷新: 清空所有寄存器
        # 
        # 关键修复：当BTB预测跳转时，也需要清空IF/ID寄存器
        # 原因：BTB预测发生在IF阶段，但PC更新要等到HazardUnit处理
        # 在这一周期内，FetchStage可能已经用旧的PC（顺序PC）取了下一条指令
        # 这条顺序取的指令是错误的延迟槽指令，需要被清空
        # 
        # **重要**: 只在 data_hazard=0 时清空 IF/ID！
        # 当 data_hazard=1 时，IF/ID 中的指令还在等待被执行，不能清空！
        # 如果在 data_hazard=1 时清空 IF/ID，会导致指令丢失。
        need_flush_ifid = (need_flush | btb_predict_taken) & (~data_hazard)
        log("FLUSH DEBUG: need_flush={}, btb_predict_taken={}, data_hazard={}, need_flush_ifid={}",
            need_flush, btb_predict_taken, data_hazard, need_flush_ifid)
        
        with Condition(if_id_valid[0] & need_flush_ifid):
            if_id_instruction[0] = UInt(XLEN)(0x00000013)  # NOP指令
            if_id_prediction_info[0] = UInt(PREDICTION_INFO_LEN)(0)
        
        # 注意：不再需要在 data_hazard 时恢复 IF/ID 寄存器
        # 因为 FetchStage 现在在 mem_sram_busy=1 时不会更新 IF/ID
        
        # **关键修复**: 当 data_hazard 时，恢复 ID/EX 寄存器
        # DecodeStage 可能已经更新了 id_ex_pc, id_ex_need_rs1, id_ex_need_rs2
        # 我们需要覆盖这些更新，保持旧值
        current_id_ex_pc = id_ex_pc[0]
        current_id_ex_need_rs1 = id_ex_need_rs1[0]
        current_id_ex_need_rs2 = id_ex_need_rs2[0]
        current_id_ex_prediction_info_ex = id_ex_prediction_info[0]
        
        with Condition(data_hazard):
            # stall 时保持 ID/EX 寄存器不变（覆盖 DecodeStage 的更新）
            id_ex_pc[0] = current_id_ex_pc
            id_ex_need_rs1[0] = current_id_ex_need_rs1
            id_ex_need_rs2[0] = current_id_ex_need_rs2
            # id_ex_prediction_info 已经在下面的条件中处理
            log("HAZARD: Restoring ID/EX registers during stall, PC={:08x}", current_id_ex_pc)
        
        # **关键修复**: 只在 id_ex_valid=1 且 data_hazard=0 时更新 ID/EX 寄存器
        # 当 data_hazard=1 时，流水线暂停，ID/EX 寄存器应该保持不变
        # 这样 ExecuteStage 会继续处理旧指令，而不是接收新指令
        with Condition(id_ex_valid[0] & (~data_hazard)):
            id_ex_control[0] = need_flush.select(nop_control, control_in)
            id_ex_immediate[0] = need_flush.select(UInt(XLEN)(0), immediate)
            id_ex_rs1_idx[0] = need_flush.select(UInt(5)(0), rs1)
            id_ex_rs2_idx[0] = need_flush.select(UInt(5)(0), rs2)
            id_ex_prediction_info[0] = need_flush.select(UInt(PREDICTION_INFO_LEN)(0), prediction_info_id)

# ==================== 顶层CPU模块 ===================
class Driver(Module):
    """五级流水线RV32I CPU"""
    def __init__(self, program_file="test_program.txt"):
        super().__init__(ports={})

    @module.combinational
    def build(self, fetch_stage):
        fetch_stage.async_called()
        
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

def generate_unified_hex(program_file="test_program.txt", output_file="unified_memory.hex"):
    """生成统一的hex文件用于冯诺依曼架构SRAM初始化
    
    从 test_program.txt 读取程序（带0x前缀），生成不带前缀的hex文件
    """
    try:
        with open(program_file, 'r') as f:
            lines = f.readlines()
        
        hex_lines = []
        for line in lines:
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            # 支持十六进制格式（带或不带0x前缀）
            if line.startswith('0x') or line.startswith('0X'):
                value = int(line, 16)
            else:
                value = int(line, 0)  # 自动检测进制
            # 生成不带前缀的8位十六进制字符串
            hex_lines.append(f"{value:08x}")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(hex_lines))
        
        print(f"Generated {output_file} with {len(hex_lines)} entries from {program_file}")
        
    except FileNotFoundError:
        print(f"Warning: Program file {program_file} not found. Creating empty hex file.")
        with open(output_file, 'w') as f:
            f.write('')
    except Exception as e:
        print(f"Error generating hex file: {e}")     

def build_cpu(program_file="test_program.txt"):
    """构建RV32I CPU系统 - 包含BTB分支预测器"""
    sys = SysBuilder('rv32i_cpu')
    with sys:
        # 创建单独的流水线寄存器，每个寄存器使用适合的宽度
        
        # IF/ID阶段寄存器
        if_id_pc = RegArray(UInt(XLEN), 1, initializer=[0])           # PC (32位)
        if_id_instruction = RegArray(UInt(XLEN), 1, initializer=[0])  # 指令 (32位)
        if_id_valid = RegArray(UInt(1), 1, initializer=[1])            # 有效标志 (1位)
        if_id_prediction_info = RegArray(UInt(PREDICTION_INFO_LEN), 1, initializer=[0])  # 预测信息 (34位)

        # ID/EX阶段寄存器
        id_ex_pc = RegArray(UInt(XLEN), 1, initializer=[0])           # PC (32位)
        id_ex_control = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])  # 控制信号 (45位)
        id_ex_valid = RegArray(UInt(1), 1, initializer=[1])            # 有效标志 (1位)
        id_ex_rs1_idx = RegArray(UInt(5), 1, initializer=[0])         # rs1索引 (5位)
        id_ex_rs2_idx = RegArray(UInt(5), 1, initializer=[0])         # rs2索引 (5位)
        id_ex_immediate = RegArray(UInt(XLEN), 1, initializer=[0])    # 立即数 (32位)
        id_ex_need_rs1 = RegArray(UInt(1), 1, initializer=[0])        # 是否需要rs1 (1位)
        id_ex_need_rs2 = RegArray(UInt(1), 1, initializer=[0])        # 是否需要rs2 (1位)
        id_ex_prediction_info = RegArray(UInt(PREDICTION_INFO_LEN), 1, initializer=[0])  # 预测信息 (34位)

        # EX/MEM阶段寄存器
        ex_mem_pc = RegArray(UInt(XLEN), 1, initializer=[0])           # PC (32位)
        ex_mem_control = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])  # 控制信号 (45位)
        ex_mem_valid = RegArray(UInt(1), 1, initializer=[1])            # 有效标志 (1位)
        ex_mem_result = RegArray(UInt(XLEN), 1, initializer=[0])       # ALU结果 (32位)
        ex_mem_data = RegArray(UInt(XLEN), 1, initializer=[0])          # 数据 (32位)
        # execute_signals 相关寄存器 (用于暂停时保持旧值)
        ex_mem_pc_change = RegArray(UInt(1), 1, initializer=[0])        # PC变化标志 (1位)
        ex_mem_target_pc = RegArray(UInt(XLEN), 1, initializer=[0])     # 目标PC (32位)
        ex_mem_prediction_result = RegArray(UInt(103), 1, initializer=[0])  # 预测结果 (103位)
        ex_sig_control = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])  # execute_signals用的控制信号 (48位)

        # MEM/WB阶段寄存器
        mem_wb_control = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])  # 控制信号 (45位)
        mem_wb_valid = RegArray(UInt(1), 1, initializer=[1])            # 有效标志 (1位)
        mem_wb_mem_data = RegArray(UInt(XLEN), 1, initializer=[0])     # 内存数据 (32位)
        mem_wb_ex_result = RegArray(UInt(XLEN), 1, initializer=[0])     # EX阶段结果 (32位)
        mem_wb_addr = RegArray(UInt(XLEN), 1, initializer=[0])          # 内存地址 (32位)
        
        # SB/SH读-修改-写状态机寄存器
        sb_sh_state = RegArray(UInt(2), 1, initializer=[0])    # 0=IDLE, 1=READ, 2=WRITE
        sb_sh_addr = RegArray(UInt(XLEN), 1, initializer=[0])  # 保存的地址
        sb_sh_data = RegArray(UInt(XLEN), 1, initializer=[0])  # 保存的数据
        sb_sh_type = RegArray(UInt(2), 1, initializer=[0])     # 保存的store类型
        
        # MEM 阶段执行标志：防止 load/SW 重复执行
        mem_last_pc = RegArray(UInt(XLEN), 1, initializer=[0xFFFFFFFF])  # 上一周期 MEM 执行的 PC

        # ==================== 冯诺依曼架构：取指状态寄存器 ====================
        # 用于处理SRAM 1-cycle latency
        if_fetch_pending = RegArray(UInt(1), 1, initializer=[0])                # 是否有待完成的取指请求
        if_fetch_pc = RegArray(UInt(XLEN), 1, initializer=[0])                  # 待取指的PC地址
        if_fetch_prediction_info = RegArray(UInt(PREDICTION_INFO_LEN), 1, initializer=[0])  # 待取指的预测信息
        if_last_can_fetch = RegArray(UInt(1), 1, initializer=[0])               # 上周期IF是否成功访问SRAM
        # 取指缓冲区：当 SRAM stall 时保存取到的指令
        if_stall_buffer_valid = RegArray(UInt(1), 1, initializer=[0])           # 缓冲区是否有效
        if_stall_buffer_pc = RegArray(UInt(XLEN), 1, initializer=[0])           # 缓冲的PC
        if_stall_buffer_instruction = RegArray(UInt(XLEN), 1, initializer=[0])  # 缓冲的指令
        if_stall_buffer_pred_info = RegArray(UInt(PREDICTION_INFO_LEN), 1, initializer=[0])  # 缓冲的预测信息

        # ==================== 乘法器寄存器 ====================
        # Wallace Tree 乘法器流水线寄存器
        mul_a = RegArray(UInt(32), 1, initializer=[0])                # 乘法操作数A
        mul_b = RegArray(UInt(32), 1, initializer=[0])                # 乘法操作数B
        mul_op_reg = RegArray(UInt(3), 1, initializer=[0])            # 乘法操作码
        mul_start = RegArray(UInt(1), 1, initializer=[0])             # 乘法开始信号
        mul_cycle_counter = RegArray(UInt(2), 1, initializer=[0])     # 乘法周期计数器 (0=空闲, 1/2/3=执行中)
        mul_stage1_sum = RegArray(UInt(64), 1, initializer=[0])       # 第一级CSA压缩结果-sum
        mul_stage1_carry = RegArray(UInt(64), 1, initializer=[0])     # 第一级CSA压缩结果-carry
        mul_stage2_sum = RegArray(UInt(64), 1, initializer=[0])       # 第二级CSA压缩结果-sum
        mul_stage2_carry = RegArray(UInt(64), 1, initializer=[0])     # 第二级CSA压缩结果-carry
        mul_valid = RegArray(UInt(1), 1, initializer=[0])             # 乘法结果有效
        mul_result_reg = RegArray(UInt(32), 1, initializer=[0])       # 乘法结果
        mul_in_progress = RegArray(UInt(1), 1, initializer=[0])       # 乘法执行中标志
        mul_rd_reg = RegArray(UInt(5), 1, initializer=[0])            # 乘法目标寄存器
        mul_control_reg = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])  # 乘法控制信号
        mul_pc_reg = RegArray(UInt(XLEN), 1, initializer=[0])         # 乘法指令PC
        
        # ==================== 除法器寄存器 ====================
        # Radix-4 SRT 不恢复除法器流水线寄存器（带查表）
        div_dividend = RegArray(UInt(32), 1, initializer=[0])            # 被除数（绝对值）
        div_divisor = RegArray(UInt(32), 1, initializer=[0])             # 除数（绝对值）
        div_op_reg = RegArray(UInt(3), 1, initializer=[0])            # 除法操作码
        div_state = RegArray(UInt(6), 1, initializer=[0])            # 除法器状态 (0=IDLE, 1=INIT, 2-17=ITERATE, 18=FINAL_CORRECTION, 19=DONE)
        div_remainder = RegArray(Int(64), 1, initializer=[0])         # 部分余数P（有符号，可为负）
        div_quotient_pos = RegArray(UInt(32), 1, initializer=[0])     # 正商累积 Q+
        div_quotient_neg = RegArray(UInt(32), 1, initializer=[0])     # 负商累积 Q-
        div_iter_count = RegArray(UInt(5), 1, initializer=[0])        # 迭代计数器 (0-15)
        div_sign = RegArray(UInt(1), 1, initializer=[0])              # 商结果符号
        div_dividend_sign = RegArray(UInt(1), 1, initializer=[0])     # 被除数原始符号（用于余数符号）
        div_valid = RegArray(UInt(1), 1, initializer=[0])             # 除法结果有效
        div_result_reg = RegArray(UInt(32), 1, initializer=[0])       # 除法结果
        div_rd_reg = RegArray(UInt(5), 1, initializer=[0])            # 除法目标寄存器
        div_control_reg = RegArray(UInt(CONTROL_LEN), 1, initializer=[0])  # 除法控制信号
        div_pc_reg = RegArray(UInt(XLEN), 1, initializer=[0])         # 除法指令PC
        div_norm_shift = RegArray(UInt(6), 1, initializer=[0])        # 归一化移位量
        div_divisor_norm = RegArray(UInt(32), 1, initializer=[0])     # 归一化后的除数

        # 分支预测器 - BTB + BHT + 有效位
        btb = RegArray(UInt(XLEN), BTB_SIZE, initializer=[0]*BTB_SIZE)        # Branch Target Buffer (32位 x 64)
        bht = RegArray(UInt(2), BTB_SIZE, initializer=[1]*BTB_SIZE)           # 2-bit饱和计数器 (初始化为01=Weakly Not Taken)
        btb_valid = RegArray(UInt(1), BTB_SIZE, initializer=[0]*BTB_SIZE)     # BTB有效位

        # ==================== 冯诺依曼架构：统一SRAM ====================
        # 从 test_program.txt 加载程序到统一 SRAM
        # 首先生成适合 SRAM 的 hex 文件
        generate_unified_hex(program_file, "unified_memory.hex")
        
        # 创建统一的 SRAM（指令和数据共享）
        unified_sram = SRAM(width=XLEN, depth=65536, init_file="unified_memory.hex")
        unified_sram.name = 'unified_sram'
        
        # 创建寄存器文件
        reg_file = RegArray(UInt(XLEN), REG_COUNT, initializer=[0]*REG_COUNT)

        pc = RegArray(UInt(XLEN), 1, initializer=[0])
        stall = RegArray(UInt(1), 1, initializer=[0])
        
        # 创建模块实例
        hazard_unit = HazardUnit()
        fetch_stage = FetchStage()
        decode_stage = DecodeStage()
        execute_stage = ExecuteStage()
        memory_stage = MemoryStage()
        writeback_stage = WriteBackStage()
        driver = Driver()

        # ==================== 冯诺依曼架构：SRAM仲裁和构建 ====================
        # 策略：
        # 1. FetchStage 内部从流水线寄存器计算 mem_sram_busy，返回 IF 的 SRAM 请求
        # 2. MemoryStage 返回 MEM 的 SRAM 请求
        # 3. SramArbiter 作为 Downstream 统一处理仲裁和 SRAM.build()
        
        # 创建 SramArbiter 实例
        sram_arbiter = SramArbiter()
        
        # 构建各个模块
        writeback_signals = writeback_stage.build(mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, mem_wb_control, mem_wb_addr, reg_file, unified_sram)
        memory_signals = memory_stage.build(ex_mem_valid, ex_mem_result, ex_mem_pc, ex_mem_data, ex_mem_control, mem_wb_control, mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, mem_wb_addr, sb_sh_state, sb_sh_addr, sb_sh_data, sb_sh_type, writeback_stage, unified_sram, mem_last_pc)
        execute_signals = execute_stage.build(id_ex_valid, id_ex_pc, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, id_ex_control, id_ex_prediction_info, ex_mem_pc, ex_mem_control, ex_mem_valid, ex_mem_result, ex_mem_data, reg_file, memory_stage, mem_wb_control, mem_wb_valid, mem_wb_mem_data, mem_wb_ex_result, unified_sram, mem_wb_addr, ex_mem_pc_change, ex_mem_target_pc, ex_mem_prediction_result, ex_sig_control, mul_a, mul_b, mul_op_reg, mul_start, mul_cycle_counter, mul_stage1_sum, mul_stage1_carry, mul_stage2_sum, mul_stage2_carry, mul_valid, mul_result_reg, mul_in_progress, mul_rd_reg, mul_control_reg, mul_pc_reg, div_dividend, div_divisor, div_op_reg, div_state, div_remainder, div_quotient_pos, div_quotient_neg, div_iter_count, div_sign, div_dividend_sign, div_valid, div_result_reg, div_rd_reg, div_control_reg, div_pc_reg, div_norm_shift, div_divisor_norm)
        decode_signals = decode_stage.build(if_id_valid, if_id_pc, if_id_instruction, if_id_prediction_info, id_ex_pc, id_ex_control, id_ex_valid, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, id_ex_need_rs1, id_ex_need_rs2, id_ex_prediction_info, reg_file, execute_stage)
        
        # 构建 FetchStage，传入流水线寄存器用于内部计算 mem_sram_busy
        fetch_signals = fetch_stage.build(pc, stall, if_id_pc, if_id_instruction, if_id_valid, if_id_prediction_info, if_fetch_pending, if_fetch_pc, if_fetch_prediction_info, if_last_can_fetch, if_stall_buffer_valid, if_stall_buffer_pc, if_stall_buffer_instruction, if_stall_buffer_pred_info, btb, bht, btb_valid, decode_stage, unified_sram, ex_mem_valid, ex_mem_control, mem_wb_valid, sb_sh_state, ex_mem_pc, mem_last_pc)
        
        # 构建 SramArbiter，统一处理 SRAM 仲裁
        sram_arbiter.build(fetch_signals, memory_signals, unified_sram)
        
        # 构建 HazardUnit
        # fetch_signals 包含 [31:0]=instruction, [32]=if_needs_sram, [64:33]=if_sram_addr
        hazard_unit.build(pc, stall, if_id_valid, if_id_pc, if_id_instruction, if_id_prediction_info, id_ex_pc, id_ex_control, id_ex_valid, id_ex_rs1_idx, id_ex_rs2_idx, id_ex_immediate, id_ex_prediction_info, id_ex_need_rs1, id_ex_need_rs2, ex_mem_valid, mem_wb_valid, btb, bht, btb_valid, fetch_signals, decode_signals, execute_signals, memory_signals, writeback_signals, mul_in_progress, mul_cycle_counter, mul_rd_reg, div_state, div_iter_count, div_rd_reg)
        
        # 构建Driver模块，处理PC更新
        driver.build(fetch_stage)
    
    return sys

def test_rv32i_cpu(program_file="test_program.txt"):
    """测试RV32I CPU"""
    sys = build_cpu(program_file)
    
    # 生成模拟器
    simulator_path, _ = elaborate(sys, verilog=False, sim_threshold=2500, resource_base='.')
    raw = utils.run_simulator(simulator_path)
    with open("result.out", 'w', encoding='utf-8') as f:
        print(raw, file=f)

if __name__ == "__main__":
    test_rv32i_cpu(program_file="test_program.txt")