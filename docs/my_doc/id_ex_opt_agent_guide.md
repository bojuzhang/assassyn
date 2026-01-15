# RV32I ID/EX 优化技术指导手册（面向 Agent）

本手册仅包含并保留以下优化项：
- 1.1 统一立即数生成
- 1.2 控制信号生成优化
- 2.2 ALU 单元优化
- 2.3 分支单元优化

目标：在不改变功能的前提下，减少硬件占用并缩短最长组合链。所有改动仅限于 rv32i_cpu.py 的 ID/EX 相关逻辑。禁止新增功能或改变指令语义。

---

## 目录

- 1. 总览与约束
- 2. 1.1 统一立即数生成（ID 阶段）
- 3. 1.2 控制信号生成优化（ID 阶段）
- 4. 2.2 ALU 单元优化（EX 阶段）
- 5. 2.3 分支单元优化（EX 阶段）
- 6. 验证清单

---

## 1. 总览与约束

### 1.1 范围
- 文件：rv32i_cpu.py
- 修改范围：
  - DecodeStage.build 内的立即数生成与控制信号生成
  - ExecuteStage.alu_unit
  - ExecuteStage.branch_unit

### 1.2 不允许的改动
- 不调整控制信号的位宽和布局（CONTROL_LEN 与控制信号字段位置必须保持）。
- 不改变已有功能（包括 M 扩展、分支预测、Hazard Unit 的逻辑等）。
- 不引入新的模块或改变流水线寄存器数量。

### 1.3 代码风格
- 保持现有命名与缩进风格。
- 使用现有类型与位宽，避免引入 new type。
- 只在必要的局部区域修改，不做全局重构。

---

## 2. 1.1 统一立即数生成（ID 阶段）

### 2.1 问题点
当前实现为 I/S/B/U/J 各自做符号扩展，重复生成多个符号扩展器，组合逻辑较多，且增加最长链。

### 2.2 目标
- 使用 instruction[31] 作为唯一符号位源。
- 复用符号扩展高位常量，减少重复逻辑。
- 输出的 immediate_i/immediate_s/immediate_b/immediate_u/immediate_j 与原语义一致。

### 2.3 具体操作步骤
1) 在 DecodeStage.build 中，替换 “提取立即数 - 使用手动符号扩展” 逻辑块。
2) 使用统一的 sign_bit 及其扩展位：
   - sign_bit = instruction[31:31]
   - sign_ext_20 = sign_bit.select(Bits(20)(0xFFFFF), Bits(20)(0))
   - sign_ext_19 = sign_bit.select(Bits(19)(0x7FFFF), Bits(19)(0))
   - sign_ext_11 = sign_bit.select(Bits(11)(0x7FF), Bits(11)(0))
3) 用 concat 拼接形成立即数：
   - I 型：concat(sign_ext_20, instruction[20:31])
   - S 型：concat(sign_ext_20, concat(instruction[25:31], instruction[7:11]))
   - B 型：concat(sign_ext_19, concat(instruction[31:31], instruction[7:7], instruction[25:30], instruction[8:11], UInt(1)(0)))
   - U 型：concat(instruction[12:31], Bits(12)(0)) 或保留 “<< 12” 形式（必须一致）
   - J 型：concat(sign_ext_11, concat(instruction[31:31], instruction[12:19], instruction[20:20], instruction[21:30], UInt(1)(0)))
4) 立即数全部以 UInt(32) 输出。

### 2.4 必须保持的语义
- B/J 立即数仍包含左移 1 位（bit0=0）。
- U 型立即数不需要符号扩展。

### 2.5 局部替换参考（非完整代码）
- 仅替换原来五段立即数生成逻辑。
- 原变量名 immediate_i/immediate_s/immediate_b/immediate_u/immediate_j 保持不变。

---

## 3. 1.2 控制信号生成优化（ID 阶段）

### 3.1 问题点
当前控制信号（reg_write, mem_read, mem_write, mem_to_reg, alu_src 等）由大量链式 select 逐条覆盖。组合层级深且重复比较逻辑。

### 3.2 目标
- 用“聚合条件”一次性生成 reg_write。
- mem_read/mem_write/mem_to_reg 直接由类型判定生成。
- alu_src 使用优先级明确的两段赋值而不是多级覆盖。
- 不改变控制信号位字段与功能。

### 3.3 具体操作步骤
1) 生成 writes_reg 与 rd_nonzero：
   - writes_reg = is_r_type | is_i_type | is_l_type | is_lui_type | is_auipc_type | is_j_type | is_jr_type | is_mul_inst | is_div_inst
   - rd_nonzero = (rd != UInt(5)(0))
   - reg_write = (writes_reg & rd_nonzero).select(UInt(1)(1), UInt(1)(0))
2) mem_read/mem_write/mem_to_reg：
   - mem_read = is_l_type.select(UInt(1)(1), UInt(1)(0))
   - mem_write = is_s_type.select(UInt(1)(1), UInt(1)(0))
   - mem_to_reg = is_l_type.select(UInt(1)(1), UInt(1)(0))
3) alu_src 统一选择策略：
   - 默认 0
   - 对所有“立即数类”指令设为 1
   - 对 AUIPC 设为 2

建议的 alu_src 逻辑：
- alu_src = UInt(2)(0)
- alu_src = (is_i_type | is_l_type | is_s_type | is_lui_type | is_j_type | is_jr_type).select(UInt(2)(1), alu_src)
- alu_src = is_auipc_type.select(UInt(2)(2), alu_src)
- 乘除法指令：保持寄存器输入（0）

4) immediate 的选择逻辑保持语义，但应减少重复覆盖：
- 先设默认 0
- 对 I/L/S/B/U/J/JR 覆盖（与原行为一致）

### 3.4 注意事项
- 不要改动 mul_op/div_op 生成逻辑。
- 不改动控制信号打包位置。
- 如果已有 reg_write 的单独覆盖（如 is_mul_inst/is_div_inst），可移除并由 writes_reg 覆盖。

---

## 4. 2.2 ALU 单元优化（EX 阶段）

### 4.1 问题点
现有 alu_unit 使用大量“op 比较 + select”串联，每个操作一个比较器，资源多且组合链长。

### 4.2 目标
- 共享加法器/减法器。
- 共享移位器与比较器。
- 减少对 op 的比较次数。
- 保持当前 ALU 操作码定义和输出结果完全一致。

### 4.3 具体操作步骤
1) 在 alu_unit 内部建立共享运算：
- 共享加/减：根据 op[0] 或显式 is_sub 选择 b 或 -b
- 共享移位：shift_amount = b[0:4]
- 共享比较：lt_signed, lt_unsigned

2) 使用少量选择器合并结果：
- 对 ADD/SUB、SLL/SRL/SRA、AND/OR/XOR、SLT/SLTU 分组选择。
- 保持原操作码映射：
  - 00000 ADD
  - 00001 SUB
  - 00010 SLL
  - 00011 SLT
  - 00100 XOR
  - 00101 SRL
  - 00110 SRA
  - 00111 SLTU
  - 01000 OR
  - 01001 AND

### 4.4 必须保持的语义
- 所有算术/逻辑结果与旧实现一致。
- 移位量仍取 b 的低 5 位。
- 结果类型为 UInt(XLEN)。

---

## 5. 2.3 分支单元优化（EX 阶段）

### 5.1 问题点
branch_unit 对每个分支类型独立比较，重复了等于/小于判断。

### 5.2 目标
- 复用等于和大小比较器。
- 仅通过少量 select 生成 taken。
- 保持当前 branch_op 编码与语义一致。

### 5.3 具体操作步骤
1) 预计算比较结果：
- eq = (a == b)
- lt_signed = (a_signed < b_signed)
- lt_unsigned = (a < b)

2) 根据 branch_op 映射 taken：
- BEQ (001): eq
- BNE (010): ~eq
- BLT (011): lt_signed
- BGE (100): ~lt_signed
- BLTU (101): lt_unsigned
- BGEU (110): ~lt_unsigned

3) 使用 select 链组合：
- 初始化 taken = 0
- 按照以上规则依次覆盖

### 5.4 必须保持的语义
- branch_op 位编码不变。
- 返回的 taken 为 UInt(1)。

---

## 6. 验证清单

在完成修改后，Agent 必须逐条确认：

1) 立即数输出与原来一致：
- I/S/B/U/J 立即数值一致（含符号扩展与左移规则）。

2) 控制信号等效：
- 对每类指令，mem_read/mem_write/mem_to_reg/reg_write/alu_src/branch_op/jump_op/jumpr_op/alu_op 与原功能一致。
- rd = x0 时 reg_write 为 0。

3) ALU 与分支结果一致：
- ALU 操作码映射不变。
- branch_unit 结果与原来一致。

4) 控制信号打包位置未变：
- 仍然使用现有的 48 位 control_signals 格式。

---

## 附：定位提示

- 立即数生成与控制信号生成位置：DecodeStage.build 中“提取立即数”与“控制信号解码”。
- ALU 与分支单元位置：ExecuteStage.alu_unit 和 ExecuteStage.branch_unit。

完成后无需新增测试文件。仅提交对 rv32i_cpu.py 的局部修改。