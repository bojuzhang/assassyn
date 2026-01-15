use crate::simulator::Simulator;
use sim_runtime::num_bigint::{BigInt, BigUint};
use sim_runtime::*;
use std::ffi::c_void;

// Elaborating module ExecuteStageInstance
pub fn ExecuteStageInstance(sim: &mut Simulator) -> bool {
  let pc_in_1 = { sim.id_ex_pc.payload[false as usize].clone() };
  let rs1_idx = { sim.id_ex_rs1_idx.payload[false as usize].clone() };
  let rs2_idx = { sim.id_ex_rs2_idx.payload[false as usize].clone() };
  let alu_b = { sim.id_ex_immediate.payload[false as usize].clone() };
  let control_in_2 = { sim.id_ex_control.payload[false as usize].clone() };
  let prediction_info_in = { sim.id_ex_prediction_info.payload[false as usize].clone() };
  let rs1_data = { sim.reg_file.payload[rs1_idx as usize].clone() };
  let rs2_data = { sim.reg_file.payload[rs2_idx as usize].clone() };
  let mem_control = { sim.ex_sig_control.payload[false as usize].clone() };
  let mem_reg_write = {
    {
      let a = ValueCastTo::<u64>::cast(&mem_control);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 7) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let mem_rd = {
    {
      let a = ValueCastTo::<u64>::cast(&mem_control);
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 25) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let mem_result = { sim.ex_mem_result.payload[false as usize].clone() };
  let wb_control = { sim.mem_wb_control.payload[false as usize].clone() };
  let wb_reg_write = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_control);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 7) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let wb_mem_to_reg = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_control);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 8) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let wb_rd_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_control);
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 25) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let wb_ex_result = { sim.mem_wb_ex_result.payload[false as usize].clone() };
  let wb_lw_data = { sim.SRAM_rdata.payload[false as usize].clone() };
  let wb_load_type = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_control);
      let mask = u64::from_str_radix("111", 2).unwrap();
      let res = (a >> 14) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let mem_wb_rd_8 = { sim.mem_wb_addr.payload[false as usize].clone() };
  let wb_byte_offset = {
    {
      let a = ValueCastTo::<u64>::cast(&mem_wb_rd_8);
      let mask = u64::from_str_radix("11", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let wb_byte0 = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_lw_data);
      let mask = u64::from_str_radix("11111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let wb_byte1 = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_lw_data);
      let mask = u64::from_str_radix("11111111", 2).unwrap();
      let res = (a >> 8) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let wb_byte2 = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_lw_data);
      let mask = u64::from_str_radix("11111111", 2).unwrap();
      let res = (a >> 16) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let wb_byte3 = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_lw_data);
      let mask = u64::from_str_radix("11111111", 2).unwrap();
      let res = (a >> 24) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let wb_byte_eq = { ValueCastTo::<u8>::cast(&wb_byte_offset) == ValueCastTo::<u8>::cast(&0u8) };
  let wb_byte_eq_1 = { ValueCastTo::<u8>::cast(&wb_byte_offset) == ValueCastTo::<u8>::cast(&1u8) };
  let wb_byte_eq_2 = { ValueCastTo::<u8>::cast(&wb_byte_offset) == ValueCastTo::<u8>::cast(&2u8) };
  let wb_byte_mux = {
    if wb_byte_eq_2 {
      wb_byte2
    } else {
      wb_byte3
    }
  };
  let wb_byte_mux_1 = {
    if wb_byte_eq_1 {
      wb_byte1
    } else {
      wb_byte_mux
    }
  };
  let wb_selected_byte = {
    if wb_byte_eq {
      wb_byte0
    } else {
      wb_byte_mux_1
    }
  };
  let wb_half0 = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_lw_data);
      let mask = u64::from_str_radix("1111111111111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u16>::cast(&res)
    }
  };
  let wb_half1 = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_lw_data);
      let mask = u64::from_str_radix("1111111111111111", 2).unwrap();
      let res = (a >> 16) & mask;
      ValueCastTo::<u16>::cast(&res)
    }
  };
  let wb_byte_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_byte_offset);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 1) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let wb_byte_eq_3 =
    { ValueCastTo::<bool>::cast(&wb_byte_slice) == ValueCastTo::<bool>::cast(&false) };
  let wb_selected_half = {
    if wb_byte_eq_3 {
      wb_half0
    } else {
      wb_half1
    }
  };
  let wb_byte_sign = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_selected_byte);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 7) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let wb_byte_mux_4 = {
    if wb_byte_sign {
      16777215u32
    } else {
      0u32
    }
  };
  let wb_byte_cat_wb_selected = {
    {
      let a = ValueCastTo::<BigUint>::cast(&wb_byte_mux_4);
      let b = ValueCastTo::<BigUint>::cast(&wb_selected_byte);
      let c = (a << 8) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let wb_lb_data = { ValueCastTo::<u32>::cast(&wb_byte_cat_wb_selected) };
  let cat_wb_selected = {
    {
      let a = ValueCastTo::<BigUint>::cast(&0u32);
      let b = ValueCastTo::<BigUint>::cast(&wb_selected_byte);
      let c = (a << 8) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let wb_lbu_data = { ValueCastTo::<u32>::cast(&cat_wb_selected) };
  let wb_half_sign = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_selected_half);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 15) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let wb_half_mux = {
    if wb_half_sign {
      65535u16
    } else {
      0u16
    }
  };
  let wb_half_cat_wb_selected = {
    {
      let a = ValueCastTo::<BigUint>::cast(&wb_half_mux);
      let b = ValueCastTo::<BigUint>::cast(&wb_selected_half);
      let c = (a << 16) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let wb_lh_data = { ValueCastTo::<u32>::cast(&wb_half_cat_wb_selected) };
  let cat_wb_selected_1 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&0u16);
      let b = ValueCastTo::<BigUint>::cast(&wb_selected_half);
      let c = (a << 16) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let wb_lhu_data = { ValueCastTo::<u32>::cast(&cat_wb_selected_1) };
  let wb_load_eq = { ValueCastTo::<u8>::cast(&wb_load_type) == ValueCastTo::<u8>::cast(&0u8) };
  let wb_load_eq_1 = { ValueCastTo::<u8>::cast(&wb_load_type) == ValueCastTo::<u8>::cast(&1u8) };
  let wb_load_eq_2 = { ValueCastTo::<u8>::cast(&wb_load_type) == ValueCastTo::<u8>::cast(&2u8) };
  let wb_load_eq_3 = { ValueCastTo::<u8>::cast(&wb_load_type) == ValueCastTo::<u8>::cast(&4u8) };
  let wb_load_mux = {
    if wb_load_eq_3 {
      wb_lbu_data
    } else {
      wb_lhu_data
    }
  };
  let wb_load_mux_1 = {
    if wb_load_eq_2 {
      wb_lw_data
    } else {
      wb_load_mux
    }
  };
  let wb_load_mux_2 = {
    if wb_load_eq_1 {
      wb_lh_data
    } else {
      wb_load_mux_1
    }
  };
  let wb_mem_data = {
    if wb_load_eq {
      wb_lb_data
    } else {
      wb_load_mux_2
    }
  };
  let wb_data_1 = {
    if wb_mem_to_reg {
      wb_mem_data
    } else {
      wb_ex_result
    }
  };
  let ex_mem_rd_9 = { sim.ex_mem_valid.payload[false as usize].clone() };
  let ex_mem_and_mem_reg =
    { ValueCastTo::<bool>::cast(&ex_mem_rd_9) & ValueCastTo::<bool>::cast(&mem_reg_write) };
  let rs1_idx_eq_mem_rd = { ValueCastTo::<u8>::cast(&rs1_idx) == ValueCastTo::<u8>::cast(&mem_rd) };
  let ex_mem_and_rs1_idx = {
    ValueCastTo::<bool>::cast(&ex_mem_and_mem_reg) & ValueCastTo::<bool>::cast(&rs1_idx_eq_mem_rd)
  };
  let mem_rd_neq = { ValueCastTo::<u8>::cast(&mem_rd) != ValueCastTo::<u8>::cast(&0u8) };
  let rs1_forward_mem =
    { ValueCastTo::<bool>::cast(&ex_mem_and_rs1_idx) & ValueCastTo::<bool>::cast(&mem_rd_neq) };
  let mem_wb_rd_9 = { sim.mem_wb_valid.payload[false as usize].clone() };
  let mem_wb_and_wb_reg =
    { ValueCastTo::<bool>::cast(&mem_wb_rd_9) & ValueCastTo::<bool>::cast(&wb_reg_write) };
  let rs1_idx_eq_wb_rd = { ValueCastTo::<u8>::cast(&rs1_idx) == ValueCastTo::<u8>::cast(&wb_rd_1) };
  let mem_wb_and_rs1_idx = {
    ValueCastTo::<bool>::cast(&mem_wb_and_wb_reg) & ValueCastTo::<bool>::cast(&rs1_idx_eq_wb_rd)
  };
  let wb_rd_neq = { ValueCastTo::<u8>::cast(&wb_rd_1) != ValueCastTo::<u8>::cast(&0u8) };
  let rs1_forward_wb =
    { ValueCastTo::<bool>::cast(&mem_wb_and_rs1_idx) & ValueCastTo::<bool>::cast(&wb_rd_neq) };
  let rs1_data_1 = {
    if rs1_forward_wb {
      wb_data_1
    } else {
      rs1_data
    }
  };
  let alu_a = {
    if rs1_forward_mem {
      mem_result
    } else {
      rs1_data_1
    }
  };
  let ex_mem_and_mem_reg_1 =
    { ValueCastTo::<bool>::cast(&ex_mem_rd_9) & ValueCastTo::<bool>::cast(&mem_reg_write) };
  let rs2_idx_eq_mem_rd = { ValueCastTo::<u8>::cast(&rs2_idx) == ValueCastTo::<u8>::cast(&mem_rd) };
  let ex_mem_and_rs2_idx = {
    ValueCastTo::<bool>::cast(&ex_mem_and_mem_reg_1) & ValueCastTo::<bool>::cast(&rs2_idx_eq_mem_rd)
  };
  let mem_rd_neq_1 = { ValueCastTo::<u8>::cast(&mem_rd) != ValueCastTo::<u8>::cast(&0u8) };
  let rs2_forward_mem =
    { ValueCastTo::<bool>::cast(&ex_mem_and_rs2_idx) & ValueCastTo::<bool>::cast(&mem_rd_neq_1) };
  let mem_wb_and_wb_reg_1 =
    { ValueCastTo::<bool>::cast(&mem_wb_rd_9) & ValueCastTo::<bool>::cast(&wb_reg_write) };
  let rs2_idx_eq_wb_rd = { ValueCastTo::<u8>::cast(&rs2_idx) == ValueCastTo::<u8>::cast(&wb_rd_1) };
  let mem_wb_and_rs2_idx = {
    ValueCastTo::<bool>::cast(&mem_wb_and_wb_reg_1) & ValueCastTo::<bool>::cast(&rs2_idx_eq_wb_rd)
  };
  let wb_rd_neq_1 = { ValueCastTo::<u8>::cast(&wb_rd_1) != ValueCastTo::<u8>::cast(&0u8) };
  let rs2_forward_wb =
    { ValueCastTo::<bool>::cast(&mem_wb_and_rs2_idx) & ValueCastTo::<bool>::cast(&wb_rd_neq_1) };
  let rs2_data_1 = {
    if rs2_forward_wb {
      wb_data_1
    } else {
      rs2_data
    }
  };
  let rs2_data_2 = {
    if rs2_forward_mem {
      mem_result
    } else {
      rs2_data_1
    }
  };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:576
  print!("@line:{:<5} {:<10}: [ExecuteStageInstance]\t", line!(), cyclize(sim.stamp));
  println!("EX FORWARD: PC={:08x}, rs1_idx={}, rs2_idx={}, rs1_fwd_mem={}, rs1_fwd_wb={}, rs2_fwd_mem={}, rs2_fwd_wb={}", pc_in_1, rs1_idx, rs2_idx, if rs1_forward_mem { 1 } else { 0 }, if rs1_forward_wb { 1 } else { 0 }, if rs2_forward_mem { 1 } else { 0 }, if rs2_forward_wb { 1 } else { 0 }, );
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:578
  print!("@line:{:<5} {:<10}: [ExecuteStageInstance]\t", line!(), cyclize(sim.stamp));
  println!("EX FORWARD DATA: rs1_reg={:08x}, rs2_reg={:08x}, mem_result={:08x}, wb_data={:08x}, rs1_data={:08x}, rs2_data={:08x}", rs1_data, rs2_data, mem_result, wb_data_1, alu_a, rs2_data_2, );
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:580
  print!("@line:{:<5} {:<10}: [ExecuteStageInstance]\t", line!(), cyclize(sim.stamp));
  println!("EX FORWARD COND: ex_mem_valid={}, mem_wb_valid={}, mem_rd={}, wb_rd={}, mem_reg_write={}, wb_reg_write={}", if ex_mem_rd_9 { 1 } else { 0 }, if mem_wb_rd_9 { 1 } else { 0 }, mem_rd, wb_rd_1, if mem_reg_write { 1 } else { 0 }, if wb_reg_write { 1 } else { 0 }, );
  let target_pc = { ValueCastTo::<u32>::cast(&pc_in_1) + ValueCastTo::<u32>::cast(&4u32) };
  let alu_op = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let mem_read_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 5) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let mem_write_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 6) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let reg_write_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 7) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let mem_to_reg_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 8) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let alu_src = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("11", 2).unwrap();
      let res = (a >> 9) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let branch_op = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("111", 2).unwrap();
      let res = (a >> 17) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let jump_op = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 20) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let jumpr_op = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 21) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let rd_addr = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 25) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let alu_a_zero = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 24) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let immediate = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("1111111111", 2).unwrap();
      let res = (a >> 22) & mask;
      ValueCastTo::<u16>::cast(&res)
    }
  };
  let mul_op = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("111", 2).unwrap();
      let res = (a >> 42) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let div_op = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_2);
      let mask = u64::from_str_radix("111", 2).unwrap();
      let res = (a >> 45) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let is_mul_inst = { ValueCastTo::<u8>::cast(&mul_op) != ValueCastTo::<u8>::cast(&0u8) };
  let is_div_inst = { ValueCastTo::<u8>::cast(&div_op) != ValueCastTo::<u8>::cast(&0u8) };
  let btb_hit = {
    {
      let a = ValueCastTo::<u64>::cast(&prediction_info_in);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let predict_taken = {
    {
      let a = ValueCastTo::<u64>::cast(&prediction_info_in);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 1) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let prediction_info_slice_2 = {
    {
      let a = ValueCastTo::<u64>::cast(&prediction_info_in);
      let mask = u64::from_str_radix("11111111111111111111111111111111", 2).unwrap();
      let res = (a >> 2) & mask;
      ValueCastTo::<u32>::cast(&res)
    }
  };
  let predicted_pc = { ValueCastTo::<u32>::cast(&prediction_info_slice_2) };
  let alu_src_eq = { ValueCastTo::<u8>::cast(&alu_src) == ValueCastTo::<u8>::cast(&0u8) };
  let alu_b_1 = {
    if alu_src_eq {
      rs2_data_2
    } else {
      alu_b
    }
  };
  let alu_src_eq_1 = { ValueCastTo::<u8>::cast(&alu_src) == ValueCastTo::<u8>::cast(&1u8) };
  let alu_b_2 = {
    if alu_src_eq_1 {
      alu_b
    } else {
      alu_b_1
    }
  };
  let alu_src_eq_2 = { ValueCastTo::<u8>::cast(&alu_src) == ValueCastTo::<u8>::cast(&2u8) };
  let alu_b_3 = {
    if alu_src_eq_2 {
      alu_b
    } else {
      alu_b_2
    }
  };
  let is_branch = { ValueCastTo::<u8>::cast(&branch_op) != ValueCastTo::<u8>::cast(&0u8) };
  let is_jump = { ValueCastTo::<bool>::cast(&jump_op) == ValueCastTo::<bool>::cast(&true) };
  let is_jumpr = { ValueCastTo::<bool>::cast(&jumpr_op) == ValueCastTo::<bool>::cast(&true) };
  let alu_src_eq_3 = { ValueCastTo::<u8>::cast(&alu_src) == ValueCastTo::<u8>::cast(&0u8) };
  let alu_a_1 = {
    if alu_src_eq_3 {
      alu_a
    } else {
      alu_a
    }
  };
  let alu_src_eq_4 = { ValueCastTo::<u8>::cast(&alu_src) == ValueCastTo::<u8>::cast(&1u8) };
  let alu_a_2 = {
    if alu_src_eq_4 {
      alu_a
    } else {
      alu_a_1
    }
  };
  let alu_src_eq_5 = { ValueCastTo::<u8>::cast(&alu_src) == ValueCastTo::<u8>::cast(&2u8) };
  let alu_a_3 = {
    if alu_src_eq_5 {
      pc_in_1
    } else {
      alu_a_2
    }
  };
  let alu_a_4 = {
    if alu_a_zero {
      0u32
    } else {
      alu_a_3
    }
  };
  let is_jump_or_is_jumpr =
    { ValueCastTo::<bool>::cast(&is_jump) | ValueCastTo::<bool>::cast(&is_jumpr) };
  let alu_a_5 = {
    if is_jump_or_is_jumpr {
      pc_in_1
    } else {
      alu_a_4
    }
  };
  let is_jump_or_is_jumpr_1 =
    { ValueCastTo::<bool>::cast(&is_jump) | ValueCastTo::<bool>::cast(&is_jumpr) };
  let alu_b_4 = {
    if is_jump_or_is_jumpr_1 {
      4u32
    } else {
      alu_b_3
    }
  };
  let alu_a_cast = { ValueCastTo::<i32>::cast(&alu_a) };
  let rs2_data_cast = { ValueCastTo::<i32>::cast(&rs2_data_2) };
  let branch_op_eq = { ValueCastTo::<u8>::cast(&branch_op) == ValueCastTo::<u8>::cast(&1u8) };
  let alu_a_eq_rs2_data =
    { ValueCastTo::<u32>::cast(&alu_a) == ValueCastTo::<u32>::cast(&rs2_data_2) };
  let alu_a_mux_1 = {
    if alu_a_eq_rs2_data {
      true
    } else {
      false
    }
  };
  let branch_op_mux = {
    if branch_op_eq {
      alu_a_mux_1
    } else {
      false
    }
  };
  let branch_op_eq_1 = { ValueCastTo::<u8>::cast(&branch_op) == ValueCastTo::<u8>::cast(&2u8) };
  let alu_a_neq_rs2_data =
    { ValueCastTo::<u32>::cast(&alu_a) != ValueCastTo::<u32>::cast(&rs2_data_2) };
  let alu_a_mux_2 = {
    if alu_a_neq_rs2_data {
      true
    } else {
      false
    }
  };
  let branch_op_mux_1 = {
    if branch_op_eq_1 {
      alu_a_mux_2
    } else {
      branch_op_mux
    }
  };
  let branch_op_eq_2 = { ValueCastTo::<u8>::cast(&branch_op) == ValueCastTo::<u8>::cast(&3u8) };
  let alu_a_lt_rs2_data =
    { ValueCastTo::<i32>::cast(&alu_a_cast) < ValueCastTo::<i32>::cast(&rs2_data_cast) };
  let alu_a_mux_3 = {
    if alu_a_lt_rs2_data {
      true
    } else {
      false
    }
  };
  let branch_op_mux_2 = {
    if branch_op_eq_2 {
      alu_a_mux_3
    } else {
      branch_op_mux_1
    }
  };
  let branch_op_eq_3 = { ValueCastTo::<u8>::cast(&branch_op) == ValueCastTo::<u8>::cast(&4u8) };
  let alu_a_ge_rs2_data =
    { ValueCastTo::<i32>::cast(&alu_a_cast) >= ValueCastTo::<i32>::cast(&rs2_data_cast) };
  let alu_a_mux_4 = {
    if alu_a_ge_rs2_data {
      true
    } else {
      false
    }
  };
  let branch_op_mux_3 = {
    if branch_op_eq_3 {
      alu_a_mux_4
    } else {
      branch_op_mux_2
    }
  };
  let branch_op_eq_4 = { ValueCastTo::<u8>::cast(&branch_op) == ValueCastTo::<u8>::cast(&5u8) };
  let alu_a_lt_rs2_data_1 =
    { ValueCastTo::<u32>::cast(&alu_a) < ValueCastTo::<u32>::cast(&rs2_data_2) };
  let alu_a_mux_5 = {
    if alu_a_lt_rs2_data_1 {
      true
    } else {
      false
    }
  };
  let branch_op_mux_4 = {
    if branch_op_eq_4 {
      alu_a_mux_5
    } else {
      branch_op_mux_3
    }
  };
  let branch_op_eq_5 = { ValueCastTo::<u8>::cast(&branch_op) == ValueCastTo::<u8>::cast(&6u8) };
  let alu_a_ge_rs2_data_1 =
    { ValueCastTo::<u32>::cast(&alu_a) >= ValueCastTo::<u32>::cast(&rs2_data_2) };
  let alu_a_mux_6 = {
    if alu_a_ge_rs2_data_1 {
      true
    } else {
      false
    }
  };
  let branch_op_mux_5 = {
    if branch_op_eq_5 {
      alu_a_mux_6
    } else {
      branch_op_mux_4
    }
  };
  let actual_taken = {
    if is_branch {
      branch_op_mux_5
    } else {
      false
    }
  };
  let actual_target_pc = { ValueCastTo::<u32>::cast(&pc_in_1) + ValueCastTo::<u32>::cast(&alu_b) };
  let new_pc_temp = { ValueCastTo::<u32>::cast(&alu_a) + ValueCastTo::<u32>::cast(&alu_b) };
  let new_pc_and = { ValueCastTo::<u32>::cast(&new_pc_temp) & ValueCastTo::<u32>::cast(&1u32) };
  let new_pc = { ValueCastTo::<u32>::cast(&new_pc_temp) ^ ValueCastTo::<u32>::cast(&new_pc_and) };
  let pc_in_add_1 = { ValueCastTo::<u32>::cast(&pc_in_1) + ValueCastTo::<u32>::cast(&4u32) };
  let correct_pc = {
    if actual_taken {
      actual_target_pc
    } else {
      pc_in_add_1
    }
  };
  let predict_taken_eq_actual_t =
    { ValueCastTo::<bool>::cast(&predict_taken) == ValueCastTo::<bool>::cast(&actual_taken) };
  let predicted_pc_eq_correct_p =
    { ValueCastTo::<u32>::cast(&predicted_pc) == ValueCastTo::<u32>::cast(&correct_pc) };
  let predict_taken_and_predict = {
    ValueCastTo::<bool>::cast(&predict_taken_eq_actual_t)
      & ValueCastTo::<bool>::cast(&predicted_pc_eq_correct_p)
  };
  let prediction_correct_hit = {
    if predict_taken_and_predict {
      true
    } else {
      false
    }
  };
  let not_actual_taken = { !actual_taken };
  let prediction_correct_miss = {
    if not_actual_taken {
      true
    } else {
      false
    }
  };
  let prediction_correct = {
    if btb_hit {
      prediction_correct_hit
    } else {
      prediction_correct_miss
    }
  };
  let not_prediction_correct = { !prediction_correct };
  let is_branch_and_not_predict =
    { ValueCastTo::<bool>::cast(&is_branch) & ValueCastTo::<bool>::cast(&not_prediction_correct) };
  let mispredict = {
    if is_branch_and_not_predict {
      true
    } else {
      false
    }
  };
  let mul_cycle = { sim.mul_cycle_counter.payload[false as usize].clone() };
  let mul_cycle_neq = { ValueCastTo::<u8>::cast(&mul_cycle) != ValueCastTo::<u8>::cast(&0u8) };
  let mul_busy = {
    if mul_cycle_neq {
      true
    } else {
      false
    }
  };
  let mul_cycle_eq = { ValueCastTo::<u8>::cast(&mul_cycle) == ValueCastTo::<u8>::cast(&3u8) };
  let mul_done = {
    if mul_cycle_eq {
      true
    } else {
      false
    }
  };
  let id_ex_rd_6 = { sim.id_ex_valid.payload[false as usize].clone() };
  let is_mul_and_id_ex =
    { ValueCastTo::<bool>::cast(&is_mul_inst) & ValueCastTo::<bool>::cast(&id_ex_rd_6) };
  let not_mul_busy = { !mul_busy };
  let is_mul_and_not_mul =
    { ValueCastTo::<bool>::cast(&is_mul_and_id_ex) & ValueCastTo::<bool>::cast(&not_mul_busy) };
  let start_new_mul = {
    if is_mul_and_not_mul {
      true
    } else {
      false
    }
  };
  if start_new_mul {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:675
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, alu_a.clone(), "ExecuteStageInstance");
      sim.mul_a.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:676
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, rs2_data_2.clone(), "ExecuteStageInstance");
      sim.mul_b.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:677
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, mul_op.clone(), "ExecuteStageInstance");
      sim.mul_op_reg.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:678
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, rd_addr.clone(), "ExecuteStageInstance");
      sim.mul_rd_reg.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:679
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, control_in_2.clone(), "ExecuteStageInstance");
      sim.mul_control_reg.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:680
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, pc_in_1.clone(), "ExecuteStageInstance");
      sim.mul_pc_reg.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:681
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, true.clone(), "ExecuteStageInstance");
      sim.mul_in_progress.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:682
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, 1u8.clone(), "ExecuteStageInstance");
      sim.mul_cycle_counter.write(0, write);
    };
  }
  let mul_cycle_eq_1 = { ValueCastTo::<u8>::cast(&mul_cycle) == ValueCastTo::<u8>::cast(&1u8) };
  if mul_cycle_eq_1 {
    let a = { sim.mul_a.payload[false as usize].clone() };
    let b = { sim.mul_b.payload[false as usize].clone() };
    let saved_op = { sim.mul_op_reg.payload[false as usize].clone() };
    let saved_op_eq = { ValueCastTo::<u8>::cast(&saved_op) == ValueCastTo::<u8>::cast(&1u8) };
    let saved_op_eq_1 = { ValueCastTo::<u8>::cast(&saved_op) == ValueCastTo::<u8>::cast(&2u8) };
    let saved_op_or_saved_op =
      { ValueCastTo::<bool>::cast(&saved_op_eq) | ValueCastTo::<bool>::cast(&saved_op_eq_1) };
    let saved_op_eq_2 = { ValueCastTo::<u8>::cast(&saved_op) == ValueCastTo::<u8>::cast(&3u8) };
    let saved_op_or_saved_op_1 = {
      ValueCastTo::<bool>::cast(&saved_op_or_saved_op) | ValueCastTo::<bool>::cast(&saved_op_eq_2)
    };
    let a_signed = {
      if saved_op_or_saved_op_1 {
        true
      } else {
        false
      }
    };
    let saved_op_eq_3 = { ValueCastTo::<u8>::cast(&saved_op) == ValueCastTo::<u8>::cast(&1u8) };
    let saved_op_eq_4 = { ValueCastTo::<u8>::cast(&saved_op) == ValueCastTo::<u8>::cast(&2u8) };
    let saved_op_or_saved_op_2 =
      { ValueCastTo::<bool>::cast(&saved_op_eq_3) | ValueCastTo::<bool>::cast(&saved_op_eq_4) };
    let b_signed = {
      if saved_op_or_saved_op_2 {
        true
      } else {
        false
      }
    };
    let a_sign = {
      {
        let a = ValueCastTo::<u64>::cast(&a);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 31) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_sign = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 31) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let a_signed_and_a_sign =
      { ValueCastTo::<bool>::cast(&a_signed) & ValueCastTo::<bool>::cast(&a_sign) };
    let a_high = {
      if a_signed_and_a_sign {
        4294967295u32
      } else {
        0u32
      }
    };
    let b_signed_and_b_sign =
      { ValueCastTo::<bool>::cast(&b_signed) & ValueCastTo::<bool>::cast(&b_sign) };
    let b_high = {
      if b_signed_and_b_sign {
        4294967295u32
      } else {
        0u32
      }
    };
    let a_64 = { ValueCastTo::<u64>::cast(&a) };
    let b_64 = { ValueCastTo::<u64>::cast(&b) };
    let b_slice_1 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 0) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq = { ValueCastTo::<bool>::cast(&b_slice_1) == ValueCastTo::<bool>::cast(&true) };
    let pp0 = {
      if b_slice_eq {
        a_64
      } else {
        0u64
      }
    };
    let b_slice_2 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 1) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_1 =
      { ValueCastTo::<bool>::cast(&b_slice_2) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&1u64) };
    let a_64_cast = { ValueCastTo::<u64>::cast(&a_64_shl) };
    let pp1 = {
      if b_slice_eq_1 {
        a_64_cast
      } else {
        0u64
      }
    };
    let b_slice_3 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 2) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_2 =
      { ValueCastTo::<bool>::cast(&b_slice_3) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_1 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&2u64) };
    let a_64_cast_1 = { ValueCastTo::<u64>::cast(&a_64_shl_1) };
    let pp2 = {
      if b_slice_eq_2 {
        a_64_cast_1
      } else {
        0u64
      }
    };
    let b_slice_4 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 3) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_3 =
      { ValueCastTo::<bool>::cast(&b_slice_4) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_2 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&3u64) };
    let a_64_cast_2 = { ValueCastTo::<u64>::cast(&a_64_shl_2) };
    let pp3 = {
      if b_slice_eq_3 {
        a_64_cast_2
      } else {
        0u64
      }
    };
    let b_slice_5 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 4) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_4 =
      { ValueCastTo::<bool>::cast(&b_slice_5) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_3 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&4u64) };
    let a_64_cast_3 = { ValueCastTo::<u64>::cast(&a_64_shl_3) };
    let pp4 = {
      if b_slice_eq_4 {
        a_64_cast_3
      } else {
        0u64
      }
    };
    let b_slice_6 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 5) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_5 =
      { ValueCastTo::<bool>::cast(&b_slice_6) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_4 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&5u64) };
    let a_64_cast_4 = { ValueCastTo::<u64>::cast(&a_64_shl_4) };
    let pp5 = {
      if b_slice_eq_5 {
        a_64_cast_4
      } else {
        0u64
      }
    };
    let b_slice_7 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 6) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_6 =
      { ValueCastTo::<bool>::cast(&b_slice_7) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_5 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&6u64) };
    let a_64_cast_5 = { ValueCastTo::<u64>::cast(&a_64_shl_5) };
    let pp6 = {
      if b_slice_eq_6 {
        a_64_cast_5
      } else {
        0u64
      }
    };
    let b_slice_8 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 7) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_7 =
      { ValueCastTo::<bool>::cast(&b_slice_8) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_6 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&7u64) };
    let a_64_cast_6 = { ValueCastTo::<u64>::cast(&a_64_shl_6) };
    let pp7 = {
      if b_slice_eq_7 {
        a_64_cast_6
      } else {
        0u64
      }
    };
    let b_slice_9 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 8) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_8 =
      { ValueCastTo::<bool>::cast(&b_slice_9) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_7 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&8u64) };
    let a_64_cast_7 = { ValueCastTo::<u64>::cast(&a_64_shl_7) };
    let pp8 = {
      if b_slice_eq_8 {
        a_64_cast_7
      } else {
        0u64
      }
    };
    let b_slice_10 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 9) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_9 =
      { ValueCastTo::<bool>::cast(&b_slice_10) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_8 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&9u64) };
    let a_64_cast_8 = { ValueCastTo::<u64>::cast(&a_64_shl_8) };
    let pp9 = {
      if b_slice_eq_9 {
        a_64_cast_8
      } else {
        0u64
      }
    };
    let b_slice_11 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 10) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_10 =
      { ValueCastTo::<bool>::cast(&b_slice_11) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_9 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&10u64) };
    let a_64_cast_9 = { ValueCastTo::<u64>::cast(&a_64_shl_9) };
    let pp10 = {
      if b_slice_eq_10 {
        a_64_cast_9
      } else {
        0u64
      }
    };
    let b_slice_12 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 11) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_11 =
      { ValueCastTo::<bool>::cast(&b_slice_12) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_10 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&11u64) };
    let a_64_cast_10 = { ValueCastTo::<u64>::cast(&a_64_shl_10) };
    let pp11 = {
      if b_slice_eq_11 {
        a_64_cast_10
      } else {
        0u64
      }
    };
    let b_slice_13 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 12) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_12 =
      { ValueCastTo::<bool>::cast(&b_slice_13) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_11 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&12u64) };
    let a_64_cast_11 = { ValueCastTo::<u64>::cast(&a_64_shl_11) };
    let pp12 = {
      if b_slice_eq_12 {
        a_64_cast_11
      } else {
        0u64
      }
    };
    let b_slice_14 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 13) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_13 =
      { ValueCastTo::<bool>::cast(&b_slice_14) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_12 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&13u64) };
    let a_64_cast_12 = { ValueCastTo::<u64>::cast(&a_64_shl_12) };
    let pp13 = {
      if b_slice_eq_13 {
        a_64_cast_12
      } else {
        0u64
      }
    };
    let b_slice_15 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 14) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_14 =
      { ValueCastTo::<bool>::cast(&b_slice_15) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_13 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&14u64) };
    let a_64_cast_13 = { ValueCastTo::<u64>::cast(&a_64_shl_13) };
    let pp14 = {
      if b_slice_eq_14 {
        a_64_cast_13
      } else {
        0u64
      }
    };
    let b_slice_16 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 15) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_15 =
      { ValueCastTo::<bool>::cast(&b_slice_16) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_14 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&15u64) };
    let a_64_cast_14 = { ValueCastTo::<u64>::cast(&a_64_shl_14) };
    let pp15 = {
      if b_slice_eq_15 {
        a_64_cast_14
      } else {
        0u64
      }
    };
    let b_slice_17 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 16) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_16 =
      { ValueCastTo::<bool>::cast(&b_slice_17) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_15 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&16u64) };
    let a_64_cast_15 = { ValueCastTo::<u64>::cast(&a_64_shl_15) };
    let pp16 = {
      if b_slice_eq_16 {
        a_64_cast_15
      } else {
        0u64
      }
    };
    let b_slice_18 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 17) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_17 =
      { ValueCastTo::<bool>::cast(&b_slice_18) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_16 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&17u64) };
    let a_64_cast_16 = { ValueCastTo::<u64>::cast(&a_64_shl_16) };
    let pp17 = {
      if b_slice_eq_17 {
        a_64_cast_16
      } else {
        0u64
      }
    };
    let b_slice_19 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 18) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_18 =
      { ValueCastTo::<bool>::cast(&b_slice_19) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_17 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&18u64) };
    let a_64_cast_17 = { ValueCastTo::<u64>::cast(&a_64_shl_17) };
    let pp18 = {
      if b_slice_eq_18 {
        a_64_cast_17
      } else {
        0u64
      }
    };
    let b_slice_20 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 19) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_19 =
      { ValueCastTo::<bool>::cast(&b_slice_20) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_18 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&19u64) };
    let a_64_cast_18 = { ValueCastTo::<u64>::cast(&a_64_shl_18) };
    let pp19 = {
      if b_slice_eq_19 {
        a_64_cast_18
      } else {
        0u64
      }
    };
    let b_slice_21 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 20) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_20 =
      { ValueCastTo::<bool>::cast(&b_slice_21) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_19 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&20u64) };
    let a_64_cast_19 = { ValueCastTo::<u64>::cast(&a_64_shl_19) };
    let pp20 = {
      if b_slice_eq_20 {
        a_64_cast_19
      } else {
        0u64
      }
    };
    let b_slice_22 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 21) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_21 =
      { ValueCastTo::<bool>::cast(&b_slice_22) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_20 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&21u64) };
    let a_64_cast_20 = { ValueCastTo::<u64>::cast(&a_64_shl_20) };
    let pp21 = {
      if b_slice_eq_21 {
        a_64_cast_20
      } else {
        0u64
      }
    };
    let b_slice_23 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 22) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_22 =
      { ValueCastTo::<bool>::cast(&b_slice_23) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_21 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&22u64) };
    let a_64_cast_21 = { ValueCastTo::<u64>::cast(&a_64_shl_21) };
    let pp22 = {
      if b_slice_eq_22 {
        a_64_cast_21
      } else {
        0u64
      }
    };
    let b_slice_24 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 23) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_23 =
      { ValueCastTo::<bool>::cast(&b_slice_24) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_22 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&23u64) };
    let a_64_cast_22 = { ValueCastTo::<u64>::cast(&a_64_shl_22) };
    let pp23 = {
      if b_slice_eq_23 {
        a_64_cast_22
      } else {
        0u64
      }
    };
    let b_slice_25 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 24) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_24 =
      { ValueCastTo::<bool>::cast(&b_slice_25) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_23 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&24u64) };
    let a_64_cast_23 = { ValueCastTo::<u64>::cast(&a_64_shl_23) };
    let pp24 = {
      if b_slice_eq_24 {
        a_64_cast_23
      } else {
        0u64
      }
    };
    let b_slice_26 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 25) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_25 =
      { ValueCastTo::<bool>::cast(&b_slice_26) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_24 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&25u64) };
    let a_64_cast_24 = { ValueCastTo::<u64>::cast(&a_64_shl_24) };
    let pp25 = {
      if b_slice_eq_25 {
        a_64_cast_24
      } else {
        0u64
      }
    };
    let b_slice_27 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 26) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_26 =
      { ValueCastTo::<bool>::cast(&b_slice_27) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_25 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&26u64) };
    let a_64_cast_25 = { ValueCastTo::<u64>::cast(&a_64_shl_25) };
    let pp26 = {
      if b_slice_eq_26 {
        a_64_cast_25
      } else {
        0u64
      }
    };
    let b_slice_28 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 27) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_27 =
      { ValueCastTo::<bool>::cast(&b_slice_28) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_26 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&27u64) };
    let a_64_cast_26 = { ValueCastTo::<u64>::cast(&a_64_shl_26) };
    let pp27 = {
      if b_slice_eq_27 {
        a_64_cast_26
      } else {
        0u64
      }
    };
    let b_slice_29 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 28) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_28 =
      { ValueCastTo::<bool>::cast(&b_slice_29) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_27 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&28u64) };
    let a_64_cast_27 = { ValueCastTo::<u64>::cast(&a_64_shl_27) };
    let pp28 = {
      if b_slice_eq_28 {
        a_64_cast_27
      } else {
        0u64
      }
    };
    let b_slice_30 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 29) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_29 =
      { ValueCastTo::<bool>::cast(&b_slice_30) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_28 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&29u64) };
    let a_64_cast_28 = { ValueCastTo::<u64>::cast(&a_64_shl_28) };
    let pp29 = {
      if b_slice_eq_29 {
        a_64_cast_28
      } else {
        0u64
      }
    };
    let b_slice_31 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 30) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_30 =
      { ValueCastTo::<bool>::cast(&b_slice_31) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_29 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&30u64) };
    let a_64_cast_29 = { ValueCastTo::<u64>::cast(&a_64_shl_29) };
    let pp30 = {
      if b_slice_eq_30 {
        a_64_cast_29
      } else {
        0u64
      }
    };
    let b_slice_32 = {
      {
        let a = ValueCastTo::<u64>::cast(&b);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 31) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let b_slice_eq_31 =
      { ValueCastTo::<bool>::cast(&b_slice_32) == ValueCastTo::<bool>::cast(&true) };
    let a_64_shl_30 = { ValueCastTo::<u64>::cast(&a_64) << ValueCastTo::<u64>::cast(&31u64) };
    let a_64_cast_30 = { ValueCastTo::<u64>::cast(&a_64_shl_30) };
    let pp31 = {
      if b_slice_eq_31 {
        a_64_cast_30
      } else {
        0u64
      }
    };
    let pp0_xor_pp1 = { ValueCastTo::<u64>::cast(&pp0) ^ ValueCastTo::<u64>::cast(&pp1) };
    let pp0_xor_xor_pp2 =
      { ValueCastTo::<u64>::cast(&pp0_xor_pp1) ^ ValueCastTo::<u64>::cast(&pp2) };
    let s = { ValueCastTo::<u64>::cast(&pp0_xor_xor_pp2) };
    let pp0_and_pp1 = { ValueCastTo::<u64>::cast(&pp0) & ValueCastTo::<u64>::cast(&pp1) };
    let pp1_and_pp2 = { ValueCastTo::<u64>::cast(&pp1) & ValueCastTo::<u64>::cast(&pp2) };
    let pp0_and_or_pp1_and =
      { ValueCastTo::<u64>::cast(&pp0_and_pp1) | ValueCastTo::<u64>::cast(&pp1_and_pp2) };
    let pp0_and_pp2 = { ValueCastTo::<u64>::cast(&pp0) & ValueCastTo::<u64>::cast(&pp2) };
    let pp0_and_or_pp0_and =
      { ValueCastTo::<u64>::cast(&pp0_and_or_pp1_and) | ValueCastTo::<u64>::cast(&pp0_and_pp2) };
    let pp0_and_shl =
      { ValueCastTo::<u64>::cast(&pp0_and_or_pp0_and) << ValueCastTo::<u64>::cast(&1u64) };
    let c = { ValueCastTo::<u64>::cast(&pp0_and_shl) };
    let pp3_xor_pp4 = { ValueCastTo::<u64>::cast(&pp3) ^ ValueCastTo::<u64>::cast(&pp4) };
    let pp3_xor_xor_pp5 =
      { ValueCastTo::<u64>::cast(&pp3_xor_pp4) ^ ValueCastTo::<u64>::cast(&pp5) };
    let s_1 = { ValueCastTo::<u64>::cast(&pp3_xor_xor_pp5) };
    let pp3_and_pp4 = { ValueCastTo::<u64>::cast(&pp3) & ValueCastTo::<u64>::cast(&pp4) };
    let pp4_and_pp5 = { ValueCastTo::<u64>::cast(&pp4) & ValueCastTo::<u64>::cast(&pp5) };
    let pp3_and_or_pp4_and =
      { ValueCastTo::<u64>::cast(&pp3_and_pp4) | ValueCastTo::<u64>::cast(&pp4_and_pp5) };
    let pp3_and_pp5 = { ValueCastTo::<u64>::cast(&pp3) & ValueCastTo::<u64>::cast(&pp5) };
    let pp3_and_or_pp3_and =
      { ValueCastTo::<u64>::cast(&pp3_and_or_pp4_and) | ValueCastTo::<u64>::cast(&pp3_and_pp5) };
    let pp3_and_shl =
      { ValueCastTo::<u64>::cast(&pp3_and_or_pp3_and) << ValueCastTo::<u64>::cast(&1u64) };
    let c_1 = { ValueCastTo::<u64>::cast(&pp3_and_shl) };
    let pp6_xor_pp7 = { ValueCastTo::<u64>::cast(&pp6) ^ ValueCastTo::<u64>::cast(&pp7) };
    let pp6_xor_xor_pp8 =
      { ValueCastTo::<u64>::cast(&pp6_xor_pp7) ^ ValueCastTo::<u64>::cast(&pp8) };
    let s_2 = { ValueCastTo::<u64>::cast(&pp6_xor_xor_pp8) };
    let pp6_and_pp7 = { ValueCastTo::<u64>::cast(&pp6) & ValueCastTo::<u64>::cast(&pp7) };
    let pp7_and_pp8 = { ValueCastTo::<u64>::cast(&pp7) & ValueCastTo::<u64>::cast(&pp8) };
    let pp6_and_or_pp7_and =
      { ValueCastTo::<u64>::cast(&pp6_and_pp7) | ValueCastTo::<u64>::cast(&pp7_and_pp8) };
    let pp6_and_pp8 = { ValueCastTo::<u64>::cast(&pp6) & ValueCastTo::<u64>::cast(&pp8) };
    let pp6_and_or_pp6_and =
      { ValueCastTo::<u64>::cast(&pp6_and_or_pp7_and) | ValueCastTo::<u64>::cast(&pp6_and_pp8) };
    let pp6_and_shl =
      { ValueCastTo::<u64>::cast(&pp6_and_or_pp6_and) << ValueCastTo::<u64>::cast(&1u64) };
    let c_2 = { ValueCastTo::<u64>::cast(&pp6_and_shl) };
    let pp9_xor_pp10 = { ValueCastTo::<u64>::cast(&pp9) ^ ValueCastTo::<u64>::cast(&pp10) };
    let pp9_xor_xor_pp11 =
      { ValueCastTo::<u64>::cast(&pp9_xor_pp10) ^ ValueCastTo::<u64>::cast(&pp11) };
    let s_3 = { ValueCastTo::<u64>::cast(&pp9_xor_xor_pp11) };
    let pp9_and_pp10 = { ValueCastTo::<u64>::cast(&pp9) & ValueCastTo::<u64>::cast(&pp10) };
    let pp10_and_pp11 = { ValueCastTo::<u64>::cast(&pp10) & ValueCastTo::<u64>::cast(&pp11) };
    let pp9_and_or_pp10_and =
      { ValueCastTo::<u64>::cast(&pp9_and_pp10) | ValueCastTo::<u64>::cast(&pp10_and_pp11) };
    let pp9_and_pp11 = { ValueCastTo::<u64>::cast(&pp9) & ValueCastTo::<u64>::cast(&pp11) };
    let pp9_and_or_pp9_and =
      { ValueCastTo::<u64>::cast(&pp9_and_or_pp10_and) | ValueCastTo::<u64>::cast(&pp9_and_pp11) };
    let pp9_and_shl =
      { ValueCastTo::<u64>::cast(&pp9_and_or_pp9_and) << ValueCastTo::<u64>::cast(&1u64) };
    let c_3 = { ValueCastTo::<u64>::cast(&pp9_and_shl) };
    let pp12_xor_pp13 = { ValueCastTo::<u64>::cast(&pp12) ^ ValueCastTo::<u64>::cast(&pp13) };
    let pp12_xor_xor_pp14 =
      { ValueCastTo::<u64>::cast(&pp12_xor_pp13) ^ ValueCastTo::<u64>::cast(&pp14) };
    let s_4 = { ValueCastTo::<u64>::cast(&pp12_xor_xor_pp14) };
    let pp12_and_pp13 = { ValueCastTo::<u64>::cast(&pp12) & ValueCastTo::<u64>::cast(&pp13) };
    let pp13_and_pp14 = { ValueCastTo::<u64>::cast(&pp13) & ValueCastTo::<u64>::cast(&pp14) };
    let pp12_and_or_pp13_and =
      { ValueCastTo::<u64>::cast(&pp12_and_pp13) | ValueCastTo::<u64>::cast(&pp13_and_pp14) };
    let pp12_and_pp14 = { ValueCastTo::<u64>::cast(&pp12) & ValueCastTo::<u64>::cast(&pp14) };
    let pp12_and_or_pp12_and = {
      ValueCastTo::<u64>::cast(&pp12_and_or_pp13_and) | ValueCastTo::<u64>::cast(&pp12_and_pp14)
    };
    let pp12_and_shl =
      { ValueCastTo::<u64>::cast(&pp12_and_or_pp12_and) << ValueCastTo::<u64>::cast(&1u64) };
    let c_4 = { ValueCastTo::<u64>::cast(&pp12_and_shl) };
    let pp15_xor_pp16 = { ValueCastTo::<u64>::cast(&pp15) ^ ValueCastTo::<u64>::cast(&pp16) };
    let pp15_xor_xor_pp17 =
      { ValueCastTo::<u64>::cast(&pp15_xor_pp16) ^ ValueCastTo::<u64>::cast(&pp17) };
    let s_5 = { ValueCastTo::<u64>::cast(&pp15_xor_xor_pp17) };
    let pp15_and_pp16 = { ValueCastTo::<u64>::cast(&pp15) & ValueCastTo::<u64>::cast(&pp16) };
    let pp16_and_pp17 = { ValueCastTo::<u64>::cast(&pp16) & ValueCastTo::<u64>::cast(&pp17) };
    let pp15_and_or_pp16_and =
      { ValueCastTo::<u64>::cast(&pp15_and_pp16) | ValueCastTo::<u64>::cast(&pp16_and_pp17) };
    let pp15_and_pp17 = { ValueCastTo::<u64>::cast(&pp15) & ValueCastTo::<u64>::cast(&pp17) };
    let pp15_and_or_pp15_and = {
      ValueCastTo::<u64>::cast(&pp15_and_or_pp16_and) | ValueCastTo::<u64>::cast(&pp15_and_pp17)
    };
    let pp15_and_shl =
      { ValueCastTo::<u64>::cast(&pp15_and_or_pp15_and) << ValueCastTo::<u64>::cast(&1u64) };
    let c_5 = { ValueCastTo::<u64>::cast(&pp15_and_shl) };
    let pp18_xor_pp19 = { ValueCastTo::<u64>::cast(&pp18) ^ ValueCastTo::<u64>::cast(&pp19) };
    let pp18_xor_xor_pp20 =
      { ValueCastTo::<u64>::cast(&pp18_xor_pp19) ^ ValueCastTo::<u64>::cast(&pp20) };
    let s_6 = { ValueCastTo::<u64>::cast(&pp18_xor_xor_pp20) };
    let pp18_and_pp19 = { ValueCastTo::<u64>::cast(&pp18) & ValueCastTo::<u64>::cast(&pp19) };
    let pp19_and_pp20 = { ValueCastTo::<u64>::cast(&pp19) & ValueCastTo::<u64>::cast(&pp20) };
    let pp18_and_or_pp19_and =
      { ValueCastTo::<u64>::cast(&pp18_and_pp19) | ValueCastTo::<u64>::cast(&pp19_and_pp20) };
    let pp18_and_pp20 = { ValueCastTo::<u64>::cast(&pp18) & ValueCastTo::<u64>::cast(&pp20) };
    let pp18_and_or_pp18_and = {
      ValueCastTo::<u64>::cast(&pp18_and_or_pp19_and) | ValueCastTo::<u64>::cast(&pp18_and_pp20)
    };
    let pp18_and_shl =
      { ValueCastTo::<u64>::cast(&pp18_and_or_pp18_and) << ValueCastTo::<u64>::cast(&1u64) };
    let c_6 = { ValueCastTo::<u64>::cast(&pp18_and_shl) };
    let pp21_xor_pp22 = { ValueCastTo::<u64>::cast(&pp21) ^ ValueCastTo::<u64>::cast(&pp22) };
    let pp21_xor_xor_pp23 =
      { ValueCastTo::<u64>::cast(&pp21_xor_pp22) ^ ValueCastTo::<u64>::cast(&pp23) };
    let s_7 = { ValueCastTo::<u64>::cast(&pp21_xor_xor_pp23) };
    let pp21_and_pp22 = { ValueCastTo::<u64>::cast(&pp21) & ValueCastTo::<u64>::cast(&pp22) };
    let pp22_and_pp23 = { ValueCastTo::<u64>::cast(&pp22) & ValueCastTo::<u64>::cast(&pp23) };
    let pp21_and_or_pp22_and =
      { ValueCastTo::<u64>::cast(&pp21_and_pp22) | ValueCastTo::<u64>::cast(&pp22_and_pp23) };
    let pp21_and_pp23 = { ValueCastTo::<u64>::cast(&pp21) & ValueCastTo::<u64>::cast(&pp23) };
    let pp21_and_or_pp21_and = {
      ValueCastTo::<u64>::cast(&pp21_and_or_pp22_and) | ValueCastTo::<u64>::cast(&pp21_and_pp23)
    };
    let pp21_and_shl =
      { ValueCastTo::<u64>::cast(&pp21_and_or_pp21_and) << ValueCastTo::<u64>::cast(&1u64) };
    let c_7 = { ValueCastTo::<u64>::cast(&pp21_and_shl) };
    let pp24_xor_pp25 = { ValueCastTo::<u64>::cast(&pp24) ^ ValueCastTo::<u64>::cast(&pp25) };
    let pp24_xor_xor_pp26 =
      { ValueCastTo::<u64>::cast(&pp24_xor_pp25) ^ ValueCastTo::<u64>::cast(&pp26) };
    let s_8 = { ValueCastTo::<u64>::cast(&pp24_xor_xor_pp26) };
    let pp24_and_pp25 = { ValueCastTo::<u64>::cast(&pp24) & ValueCastTo::<u64>::cast(&pp25) };
    let pp25_and_pp26 = { ValueCastTo::<u64>::cast(&pp25) & ValueCastTo::<u64>::cast(&pp26) };
    let pp24_and_or_pp25_and =
      { ValueCastTo::<u64>::cast(&pp24_and_pp25) | ValueCastTo::<u64>::cast(&pp25_and_pp26) };
    let pp24_and_pp26 = { ValueCastTo::<u64>::cast(&pp24) & ValueCastTo::<u64>::cast(&pp26) };
    let pp24_and_or_pp24_and = {
      ValueCastTo::<u64>::cast(&pp24_and_or_pp25_and) | ValueCastTo::<u64>::cast(&pp24_and_pp26)
    };
    let pp24_and_shl =
      { ValueCastTo::<u64>::cast(&pp24_and_or_pp24_and) << ValueCastTo::<u64>::cast(&1u64) };
    let c_8 = { ValueCastTo::<u64>::cast(&pp24_and_shl) };
    let pp27_xor_pp28 = { ValueCastTo::<u64>::cast(&pp27) ^ ValueCastTo::<u64>::cast(&pp28) };
    let pp27_xor_xor_pp29 =
      { ValueCastTo::<u64>::cast(&pp27_xor_pp28) ^ ValueCastTo::<u64>::cast(&pp29) };
    let s_9 = { ValueCastTo::<u64>::cast(&pp27_xor_xor_pp29) };
    let pp27_and_pp28 = { ValueCastTo::<u64>::cast(&pp27) & ValueCastTo::<u64>::cast(&pp28) };
    let pp28_and_pp29 = { ValueCastTo::<u64>::cast(&pp28) & ValueCastTo::<u64>::cast(&pp29) };
    let pp27_and_or_pp28_and =
      { ValueCastTo::<u64>::cast(&pp27_and_pp28) | ValueCastTo::<u64>::cast(&pp28_and_pp29) };
    let pp27_and_pp29 = { ValueCastTo::<u64>::cast(&pp27) & ValueCastTo::<u64>::cast(&pp29) };
    let pp27_and_or_pp27_and = {
      ValueCastTo::<u64>::cast(&pp27_and_or_pp28_and) | ValueCastTo::<u64>::cast(&pp27_and_pp29)
    };
    let pp27_and_shl =
      { ValueCastTo::<u64>::cast(&pp27_and_or_pp27_and) << ValueCastTo::<u64>::cast(&1u64) };
    let c_9 = { ValueCastTo::<u64>::cast(&pp27_and_shl) };
    let s_xor_c = { ValueCastTo::<u64>::cast(&s) ^ ValueCastTo::<u64>::cast(&c) };
    let s_xor_xor_s_1 = { ValueCastTo::<u64>::cast(&s_xor_c) ^ ValueCastTo::<u64>::cast(&s_1) };
    let s_10 = { ValueCastTo::<u64>::cast(&s_xor_xor_s_1) };
    let s_and_c = { ValueCastTo::<u64>::cast(&s) & ValueCastTo::<u64>::cast(&c) };
    let c_and_s_1 = { ValueCastTo::<u64>::cast(&c) & ValueCastTo::<u64>::cast(&s_1) };
    let s_and_or_c_and =
      { ValueCastTo::<u64>::cast(&s_and_c) | ValueCastTo::<u64>::cast(&c_and_s_1) };
    let s_and_s_1 = { ValueCastTo::<u64>::cast(&s) & ValueCastTo::<u64>::cast(&s_1) };
    let s_and_or_s_and =
      { ValueCastTo::<u64>::cast(&s_and_or_c_and) | ValueCastTo::<u64>::cast(&s_and_s_1) };
    let s_and_shl =
      { ValueCastTo::<u64>::cast(&s_and_or_s_and) << ValueCastTo::<u64>::cast(&1u64) };
    let c_10 = { ValueCastTo::<u64>::cast(&s_and_shl) };
    let c_1_xor_s_2 = { ValueCastTo::<u64>::cast(&c_1) ^ ValueCastTo::<u64>::cast(&s_2) };
    let c_1_xor_c_2 = { ValueCastTo::<u64>::cast(&c_1_xor_s_2) ^ ValueCastTo::<u64>::cast(&c_2) };
    let s_11 = { ValueCastTo::<u64>::cast(&c_1_xor_c_2) };
    let c_1_and_s_2 = { ValueCastTo::<u64>::cast(&c_1) & ValueCastTo::<u64>::cast(&s_2) };
    let s_2_and_c_2 = { ValueCastTo::<u64>::cast(&s_2) & ValueCastTo::<u64>::cast(&c_2) };
    let c_1_or_s_2 =
      { ValueCastTo::<u64>::cast(&c_1_and_s_2) | ValueCastTo::<u64>::cast(&s_2_and_c_2) };
    let c_1_and_c_2 = { ValueCastTo::<u64>::cast(&c_1) & ValueCastTo::<u64>::cast(&c_2) };
    let c_1_or_c_1 =
      { ValueCastTo::<u64>::cast(&c_1_or_s_2) | ValueCastTo::<u64>::cast(&c_1_and_c_2) };
    let c_1_shl = { ValueCastTo::<u64>::cast(&c_1_or_c_1) << ValueCastTo::<u64>::cast(&1u64) };
    let c_11 = { ValueCastTo::<u64>::cast(&c_1_shl) };
    let s_3_xor_c_3 = { ValueCastTo::<u64>::cast(&s_3) ^ ValueCastTo::<u64>::cast(&c_3) };
    let s_3_xor_s_4 = { ValueCastTo::<u64>::cast(&s_3_xor_c_3) ^ ValueCastTo::<u64>::cast(&s_4) };
    let s_12 = { ValueCastTo::<u64>::cast(&s_3_xor_s_4) };
    let s_3_and_c_3 = { ValueCastTo::<u64>::cast(&s_3) & ValueCastTo::<u64>::cast(&c_3) };
    let c_3_and_s_4 = { ValueCastTo::<u64>::cast(&c_3) & ValueCastTo::<u64>::cast(&s_4) };
    let s_3_or_c_3 =
      { ValueCastTo::<u64>::cast(&s_3_and_c_3) | ValueCastTo::<u64>::cast(&c_3_and_s_4) };
    let s_3_and_s_4 = { ValueCastTo::<u64>::cast(&s_3) & ValueCastTo::<u64>::cast(&s_4) };
    let s_3_or_s_3 =
      { ValueCastTo::<u64>::cast(&s_3_or_c_3) | ValueCastTo::<u64>::cast(&s_3_and_s_4) };
    let s_3_shl = { ValueCastTo::<u64>::cast(&s_3_or_s_3) << ValueCastTo::<u64>::cast(&1u64) };
    let c_12 = { ValueCastTo::<u64>::cast(&s_3_shl) };
    let c_4_xor_s_5 = { ValueCastTo::<u64>::cast(&c_4) ^ ValueCastTo::<u64>::cast(&s_5) };
    let c_4_xor_c_5 = { ValueCastTo::<u64>::cast(&c_4_xor_s_5) ^ ValueCastTo::<u64>::cast(&c_5) };
    let s_13 = { ValueCastTo::<u64>::cast(&c_4_xor_c_5) };
    let c_4_and_s_5 = { ValueCastTo::<u64>::cast(&c_4) & ValueCastTo::<u64>::cast(&s_5) };
    let s_5_and_c_5 = { ValueCastTo::<u64>::cast(&s_5) & ValueCastTo::<u64>::cast(&c_5) };
    let c_4_or_s_5 =
      { ValueCastTo::<u64>::cast(&c_4_and_s_5) | ValueCastTo::<u64>::cast(&s_5_and_c_5) };
    let c_4_and_c_5 = { ValueCastTo::<u64>::cast(&c_4) & ValueCastTo::<u64>::cast(&c_5) };
    let c_4_or_c_4 =
      { ValueCastTo::<u64>::cast(&c_4_or_s_5) | ValueCastTo::<u64>::cast(&c_4_and_c_5) };
    let c_4_shl = { ValueCastTo::<u64>::cast(&c_4_or_c_4) << ValueCastTo::<u64>::cast(&1u64) };
    let c_13 = { ValueCastTo::<u64>::cast(&c_4_shl) };
    let s_6_xor_c_6 = { ValueCastTo::<u64>::cast(&s_6) ^ ValueCastTo::<u64>::cast(&c_6) };
    let s_6_xor_s_7 = { ValueCastTo::<u64>::cast(&s_6_xor_c_6) ^ ValueCastTo::<u64>::cast(&s_7) };
    let s_14 = { ValueCastTo::<u64>::cast(&s_6_xor_s_7) };
    let s_6_and_c_6 = { ValueCastTo::<u64>::cast(&s_6) & ValueCastTo::<u64>::cast(&c_6) };
    let c_6_and_s_7 = { ValueCastTo::<u64>::cast(&c_6) & ValueCastTo::<u64>::cast(&s_7) };
    let s_6_or_c_6 =
      { ValueCastTo::<u64>::cast(&s_6_and_c_6) | ValueCastTo::<u64>::cast(&c_6_and_s_7) };
    let s_6_and_s_7 = { ValueCastTo::<u64>::cast(&s_6) & ValueCastTo::<u64>::cast(&s_7) };
    let s_6_or_s_6 =
      { ValueCastTo::<u64>::cast(&s_6_or_c_6) | ValueCastTo::<u64>::cast(&s_6_and_s_7) };
    let s_6_shl = { ValueCastTo::<u64>::cast(&s_6_or_s_6) << ValueCastTo::<u64>::cast(&1u64) };
    let c_14 = { ValueCastTo::<u64>::cast(&s_6_shl) };
    let c_7_xor_s_8 = { ValueCastTo::<u64>::cast(&c_7) ^ ValueCastTo::<u64>::cast(&s_8) };
    let c_7_xor_c_8 = { ValueCastTo::<u64>::cast(&c_7_xor_s_8) ^ ValueCastTo::<u64>::cast(&c_8) };
    let s_15 = { ValueCastTo::<u64>::cast(&c_7_xor_c_8) };
    let c_7_and_s_8 = { ValueCastTo::<u64>::cast(&c_7) & ValueCastTo::<u64>::cast(&s_8) };
    let s_8_and_c_8 = { ValueCastTo::<u64>::cast(&s_8) & ValueCastTo::<u64>::cast(&c_8) };
    let c_7_or_s_8 =
      { ValueCastTo::<u64>::cast(&c_7_and_s_8) | ValueCastTo::<u64>::cast(&s_8_and_c_8) };
    let c_7_and_c_8 = { ValueCastTo::<u64>::cast(&c_7) & ValueCastTo::<u64>::cast(&c_8) };
    let c_7_or_c_7 =
      { ValueCastTo::<u64>::cast(&c_7_or_s_8) | ValueCastTo::<u64>::cast(&c_7_and_c_8) };
    let c_7_shl = { ValueCastTo::<u64>::cast(&c_7_or_c_7) << ValueCastTo::<u64>::cast(&1u64) };
    let c_15 = { ValueCastTo::<u64>::cast(&c_7_shl) };
    let s_9_xor_c_9 = { ValueCastTo::<u64>::cast(&s_9) ^ ValueCastTo::<u64>::cast(&c_9) };
    let s_9_xor_pp30 = { ValueCastTo::<u64>::cast(&s_9_xor_c_9) ^ ValueCastTo::<u64>::cast(&pp30) };
    let s_16 = { ValueCastTo::<u64>::cast(&s_9_xor_pp30) };
    let s_9_and_c_9 = { ValueCastTo::<u64>::cast(&s_9) & ValueCastTo::<u64>::cast(&c_9) };
    let c_9_and_pp30 = { ValueCastTo::<u64>::cast(&c_9) & ValueCastTo::<u64>::cast(&pp30) };
    let s_9_or_c_9 =
      { ValueCastTo::<u64>::cast(&s_9_and_c_9) | ValueCastTo::<u64>::cast(&c_9_and_pp30) };
    let s_9_and_pp30 = { ValueCastTo::<u64>::cast(&s_9) & ValueCastTo::<u64>::cast(&pp30) };
    let s_9_or_s_9 =
      { ValueCastTo::<u64>::cast(&s_9_or_c_9) | ValueCastTo::<u64>::cast(&s_9_and_pp30) };
    let s_9_shl = { ValueCastTo::<u64>::cast(&s_9_or_s_9) << ValueCastTo::<u64>::cast(&1u64) };
    let c_16 = { ValueCastTo::<u64>::cast(&s_9_shl) };
    let s_10_xor_c_10 = { ValueCastTo::<u64>::cast(&s_10) ^ ValueCastTo::<u64>::cast(&c_10) };
    let s_10_xor_s_11 =
      { ValueCastTo::<u64>::cast(&s_10_xor_c_10) ^ ValueCastTo::<u64>::cast(&s_11) };
    let s_17 = { ValueCastTo::<u64>::cast(&s_10_xor_s_11) };
    let s_10_and_c_10 = { ValueCastTo::<u64>::cast(&s_10) & ValueCastTo::<u64>::cast(&c_10) };
    let c_10_and_s_11 = { ValueCastTo::<u64>::cast(&c_10) & ValueCastTo::<u64>::cast(&s_11) };
    let s_10_or_c_10 =
      { ValueCastTo::<u64>::cast(&s_10_and_c_10) | ValueCastTo::<u64>::cast(&c_10_and_s_11) };
    let s_10_and_s_11 = { ValueCastTo::<u64>::cast(&s_10) & ValueCastTo::<u64>::cast(&s_11) };
    let s_10_or_s_10 =
      { ValueCastTo::<u64>::cast(&s_10_or_c_10) | ValueCastTo::<u64>::cast(&s_10_and_s_11) };
    let s_10_shl = { ValueCastTo::<u64>::cast(&s_10_or_s_10) << ValueCastTo::<u64>::cast(&1u64) };
    let c_17 = { ValueCastTo::<u64>::cast(&s_10_shl) };
    let c_11_xor_s_12 = { ValueCastTo::<u64>::cast(&c_11) ^ ValueCastTo::<u64>::cast(&s_12) };
    let c_11_xor_c_12 =
      { ValueCastTo::<u64>::cast(&c_11_xor_s_12) ^ ValueCastTo::<u64>::cast(&c_12) };
    let s_18 = { ValueCastTo::<u64>::cast(&c_11_xor_c_12) };
    let c_11_and_s_12 = { ValueCastTo::<u64>::cast(&c_11) & ValueCastTo::<u64>::cast(&s_12) };
    let s_12_and_c_12 = { ValueCastTo::<u64>::cast(&s_12) & ValueCastTo::<u64>::cast(&c_12) };
    let c_11_or_s_12 =
      { ValueCastTo::<u64>::cast(&c_11_and_s_12) | ValueCastTo::<u64>::cast(&s_12_and_c_12) };
    let c_11_and_c_12 = { ValueCastTo::<u64>::cast(&c_11) & ValueCastTo::<u64>::cast(&c_12) };
    let c_11_or_c_11 =
      { ValueCastTo::<u64>::cast(&c_11_or_s_12) | ValueCastTo::<u64>::cast(&c_11_and_c_12) };
    let c_11_shl = { ValueCastTo::<u64>::cast(&c_11_or_c_11) << ValueCastTo::<u64>::cast(&1u64) };
    let c_18 = { ValueCastTo::<u64>::cast(&c_11_shl) };
    let s_13_xor_c_13 = { ValueCastTo::<u64>::cast(&s_13) ^ ValueCastTo::<u64>::cast(&c_13) };
    let s_13_xor_s_14 =
      { ValueCastTo::<u64>::cast(&s_13_xor_c_13) ^ ValueCastTo::<u64>::cast(&s_14) };
    let s_19 = { ValueCastTo::<u64>::cast(&s_13_xor_s_14) };
    let s_13_and_c_13 = { ValueCastTo::<u64>::cast(&s_13) & ValueCastTo::<u64>::cast(&c_13) };
    let c_13_and_s_14 = { ValueCastTo::<u64>::cast(&c_13) & ValueCastTo::<u64>::cast(&s_14) };
    let s_13_or_c_13 =
      { ValueCastTo::<u64>::cast(&s_13_and_c_13) | ValueCastTo::<u64>::cast(&c_13_and_s_14) };
    let s_13_and_s_14 = { ValueCastTo::<u64>::cast(&s_13) & ValueCastTo::<u64>::cast(&s_14) };
    let s_13_or_s_13 =
      { ValueCastTo::<u64>::cast(&s_13_or_c_13) | ValueCastTo::<u64>::cast(&s_13_and_s_14) };
    let s_13_shl = { ValueCastTo::<u64>::cast(&s_13_or_s_13) << ValueCastTo::<u64>::cast(&1u64) };
    let c_19 = { ValueCastTo::<u64>::cast(&s_13_shl) };
    let c_14_xor_s_15 = { ValueCastTo::<u64>::cast(&c_14) ^ ValueCastTo::<u64>::cast(&s_15) };
    let c_14_xor_c_15 =
      { ValueCastTo::<u64>::cast(&c_14_xor_s_15) ^ ValueCastTo::<u64>::cast(&c_15) };
    let s_20 = { ValueCastTo::<u64>::cast(&c_14_xor_c_15) };
    let c_14_and_s_15 = { ValueCastTo::<u64>::cast(&c_14) & ValueCastTo::<u64>::cast(&s_15) };
    let s_15_and_c_15 = { ValueCastTo::<u64>::cast(&s_15) & ValueCastTo::<u64>::cast(&c_15) };
    let c_14_or_s_15 =
      { ValueCastTo::<u64>::cast(&c_14_and_s_15) | ValueCastTo::<u64>::cast(&s_15_and_c_15) };
    let c_14_and_c_15 = { ValueCastTo::<u64>::cast(&c_14) & ValueCastTo::<u64>::cast(&c_15) };
    let c_14_or_c_14 =
      { ValueCastTo::<u64>::cast(&c_14_or_s_15) | ValueCastTo::<u64>::cast(&c_14_and_c_15) };
    let c_14_shl = { ValueCastTo::<u64>::cast(&c_14_or_c_14) << ValueCastTo::<u64>::cast(&1u64) };
    let c_20 = { ValueCastTo::<u64>::cast(&c_14_shl) };
    let s_16_xor_c_16 = { ValueCastTo::<u64>::cast(&s_16) ^ ValueCastTo::<u64>::cast(&c_16) };
    let s_16_xor_pp31 =
      { ValueCastTo::<u64>::cast(&s_16_xor_c_16) ^ ValueCastTo::<u64>::cast(&pp31) };
    let s_21 = { ValueCastTo::<u64>::cast(&s_16_xor_pp31) };
    let s_16_and_c_16 = { ValueCastTo::<u64>::cast(&s_16) & ValueCastTo::<u64>::cast(&c_16) };
    let c_16_and_pp31 = { ValueCastTo::<u64>::cast(&c_16) & ValueCastTo::<u64>::cast(&pp31) };
    let s_16_or_c_16 =
      { ValueCastTo::<u64>::cast(&s_16_and_c_16) | ValueCastTo::<u64>::cast(&c_16_and_pp31) };
    let s_16_and_pp31 = { ValueCastTo::<u64>::cast(&s_16) & ValueCastTo::<u64>::cast(&pp31) };
    let s_16_or_s_16 =
      { ValueCastTo::<u64>::cast(&s_16_or_c_16) | ValueCastTo::<u64>::cast(&s_16_and_pp31) };
    let s_16_shl = { ValueCastTo::<u64>::cast(&s_16_or_s_16) << ValueCastTo::<u64>::cast(&1u64) };
    let c_21 = { ValueCastTo::<u64>::cast(&s_16_shl) };
    let s_17_xor_c_17 = { ValueCastTo::<u64>::cast(&s_17) ^ ValueCastTo::<u64>::cast(&c_17) };
    let s_17_xor_s_18 =
      { ValueCastTo::<u64>::cast(&s_17_xor_c_17) ^ ValueCastTo::<u64>::cast(&s_18) };
    let s_22 = { ValueCastTo::<u64>::cast(&s_17_xor_s_18) };
    let s_17_and_c_17 = { ValueCastTo::<u64>::cast(&s_17) & ValueCastTo::<u64>::cast(&c_17) };
    let c_17_and_s_18 = { ValueCastTo::<u64>::cast(&c_17) & ValueCastTo::<u64>::cast(&s_18) };
    let s_17_or_c_17 =
      { ValueCastTo::<u64>::cast(&s_17_and_c_17) | ValueCastTo::<u64>::cast(&c_17_and_s_18) };
    let s_17_and_s_18 = { ValueCastTo::<u64>::cast(&s_17) & ValueCastTo::<u64>::cast(&s_18) };
    let s_17_or_s_17 =
      { ValueCastTo::<u64>::cast(&s_17_or_c_17) | ValueCastTo::<u64>::cast(&s_17_and_s_18) };
    let s_17_shl = { ValueCastTo::<u64>::cast(&s_17_or_s_17) << ValueCastTo::<u64>::cast(&1u64) };
    let c_22 = { ValueCastTo::<u64>::cast(&s_17_shl) };
    let c_18_xor_s_19 = { ValueCastTo::<u64>::cast(&c_18) ^ ValueCastTo::<u64>::cast(&s_19) };
    let c_18_xor_c_19 =
      { ValueCastTo::<u64>::cast(&c_18_xor_s_19) ^ ValueCastTo::<u64>::cast(&c_19) };
    let s_23 = { ValueCastTo::<u64>::cast(&c_18_xor_c_19) };
    let c_18_and_s_19 = { ValueCastTo::<u64>::cast(&c_18) & ValueCastTo::<u64>::cast(&s_19) };
    let s_19_and_c_19 = { ValueCastTo::<u64>::cast(&s_19) & ValueCastTo::<u64>::cast(&c_19) };
    let c_18_or_s_19 =
      { ValueCastTo::<u64>::cast(&c_18_and_s_19) | ValueCastTo::<u64>::cast(&s_19_and_c_19) };
    let c_18_and_c_19 = { ValueCastTo::<u64>::cast(&c_18) & ValueCastTo::<u64>::cast(&c_19) };
    let c_18_or_c_18 =
      { ValueCastTo::<u64>::cast(&c_18_or_s_19) | ValueCastTo::<u64>::cast(&c_18_and_c_19) };
    let c_18_shl = { ValueCastTo::<u64>::cast(&c_18_or_c_18) << ValueCastTo::<u64>::cast(&1u64) };
    let c_23 = { ValueCastTo::<u64>::cast(&c_18_shl) };
    let s_20_xor_c_20 = { ValueCastTo::<u64>::cast(&s_20) ^ ValueCastTo::<u64>::cast(&c_20) };
    let s_20_xor_s_21 =
      { ValueCastTo::<u64>::cast(&s_20_xor_c_20) ^ ValueCastTo::<u64>::cast(&s_21) };
    let s_24 = { ValueCastTo::<u64>::cast(&s_20_xor_s_21) };
    let s_20_and_c_20 = { ValueCastTo::<u64>::cast(&s_20) & ValueCastTo::<u64>::cast(&c_20) };
    let c_20_and_s_21 = { ValueCastTo::<u64>::cast(&c_20) & ValueCastTo::<u64>::cast(&s_21) };
    let s_20_or_c_20 =
      { ValueCastTo::<u64>::cast(&s_20_and_c_20) | ValueCastTo::<u64>::cast(&c_20_and_s_21) };
    let s_20_and_s_21 = { ValueCastTo::<u64>::cast(&s_20) & ValueCastTo::<u64>::cast(&s_21) };
    let s_20_or_s_20 =
      { ValueCastTo::<u64>::cast(&s_20_or_c_20) | ValueCastTo::<u64>::cast(&s_20_and_s_21) };
    let s_20_shl = { ValueCastTo::<u64>::cast(&s_20_or_s_20) << ValueCastTo::<u64>::cast(&1u64) };
    let c_24 = { ValueCastTo::<u64>::cast(&s_20_shl) };
    let s_22_xor_c_22 = { ValueCastTo::<u64>::cast(&s_22) ^ ValueCastTo::<u64>::cast(&c_22) };
    let s_22_xor_s_23 =
      { ValueCastTo::<u64>::cast(&s_22_xor_c_22) ^ ValueCastTo::<u64>::cast(&s_23) };
    let s_25 = { ValueCastTo::<u64>::cast(&s_22_xor_s_23) };
    let s_22_and_c_22 = { ValueCastTo::<u64>::cast(&s_22) & ValueCastTo::<u64>::cast(&c_22) };
    let c_22_and_s_23 = { ValueCastTo::<u64>::cast(&c_22) & ValueCastTo::<u64>::cast(&s_23) };
    let s_22_or_c_22 =
      { ValueCastTo::<u64>::cast(&s_22_and_c_22) | ValueCastTo::<u64>::cast(&c_22_and_s_23) };
    let s_22_and_s_23 = { ValueCastTo::<u64>::cast(&s_22) & ValueCastTo::<u64>::cast(&s_23) };
    let s_22_or_s_22 =
      { ValueCastTo::<u64>::cast(&s_22_or_c_22) | ValueCastTo::<u64>::cast(&s_22_and_s_23) };
    let s_22_shl = { ValueCastTo::<u64>::cast(&s_22_or_s_22) << ValueCastTo::<u64>::cast(&1u64) };
    let c_25 = { ValueCastTo::<u64>::cast(&s_22_shl) };
    let c_23_xor_s_24 = { ValueCastTo::<u64>::cast(&c_23) ^ ValueCastTo::<u64>::cast(&s_24) };
    let c_23_xor_c_24 =
      { ValueCastTo::<u64>::cast(&c_23_xor_s_24) ^ ValueCastTo::<u64>::cast(&c_24) };
    let s_26 = { ValueCastTo::<u64>::cast(&c_23_xor_c_24) };
    let c_23_and_s_24 = { ValueCastTo::<u64>::cast(&c_23) & ValueCastTo::<u64>::cast(&s_24) };
    let s_24_and_c_24 = { ValueCastTo::<u64>::cast(&s_24) & ValueCastTo::<u64>::cast(&c_24) };
    let c_23_or_s_24 =
      { ValueCastTo::<u64>::cast(&c_23_and_s_24) | ValueCastTo::<u64>::cast(&s_24_and_c_24) };
    let c_23_and_c_24 = { ValueCastTo::<u64>::cast(&c_23) & ValueCastTo::<u64>::cast(&c_24) };
    let c_23_or_c_23 =
      { ValueCastTo::<u64>::cast(&c_23_or_s_24) | ValueCastTo::<u64>::cast(&c_23_and_c_24) };
    let c_23_shl = { ValueCastTo::<u64>::cast(&c_23_or_c_23) << ValueCastTo::<u64>::cast(&1u64) };
    let c_26 = { ValueCastTo::<u64>::cast(&c_23_shl) };
    let s_25_xor_c_25 = { ValueCastTo::<u64>::cast(&s_25) ^ ValueCastTo::<u64>::cast(&c_25) };
    let s_25_xor_s_26 =
      { ValueCastTo::<u64>::cast(&s_25_xor_c_25) ^ ValueCastTo::<u64>::cast(&s_26) };
    let s_27 = { ValueCastTo::<u64>::cast(&s_25_xor_s_26) };
    let s_25_and_c_25 = { ValueCastTo::<u64>::cast(&s_25) & ValueCastTo::<u64>::cast(&c_25) };
    let c_25_and_s_26 = { ValueCastTo::<u64>::cast(&c_25) & ValueCastTo::<u64>::cast(&s_26) };
    let s_25_or_c_25 =
      { ValueCastTo::<u64>::cast(&s_25_and_c_25) | ValueCastTo::<u64>::cast(&c_25_and_s_26) };
    let s_25_and_s_26 = { ValueCastTo::<u64>::cast(&s_25) & ValueCastTo::<u64>::cast(&s_26) };
    let s_25_or_s_25 =
      { ValueCastTo::<u64>::cast(&s_25_or_c_25) | ValueCastTo::<u64>::cast(&s_25_and_s_26) };
    let s_25_shl = { ValueCastTo::<u64>::cast(&s_25_or_s_25) << ValueCastTo::<u64>::cast(&1u64) };
    let c_27 = { ValueCastTo::<u64>::cast(&s_25_shl) };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:802
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, s_27.clone(), "ExecuteStageInstance");
      sim.mul_stage1_sum.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:803
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, c_27.clone(), "ExecuteStageInstance");
      sim.mul_stage1_carry.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:804
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, c_26.clone(), "ExecuteStageInstance");
      sim.mul_stage2_sum.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:805
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, c_21.clone(), "ExecuteStageInstance");
      sim.mul_stage2_carry.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:807
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, 2u8.clone(), "ExecuteStageInstance");
      sim.mul_cycle_counter.write(0, write);
    };
  }
  let mul_cycle_eq_2 = { ValueCastTo::<u8>::cast(&mul_cycle) == ValueCastTo::<u8>::cast(&2u8) };
  if mul_cycle_eq_2 {
    let q0_r = { sim.mul_stage1_sum.payload[false as usize].clone() };
    let q1_r = { sim.mul_stage1_carry.payload[false as usize].clone() };
    let z3_r = { sim.mul_stage2_sum.payload[false as usize].clone() };
    let w4_r = { sim.mul_stage2_carry.payload[false as usize].clone() };
    let q0_r_xor_q1_r = { ValueCastTo::<u64>::cast(&q0_r) ^ ValueCastTo::<u64>::cast(&q1_r) };
    let q0_r_xor_z3_r =
      { ValueCastTo::<u64>::cast(&q0_r_xor_q1_r) ^ ValueCastTo::<u64>::cast(&z3_r) };
    let s_28 = { ValueCastTo::<u64>::cast(&q0_r_xor_z3_r) };
    let q0_r_and_q1_r = { ValueCastTo::<u64>::cast(&q0_r) & ValueCastTo::<u64>::cast(&q1_r) };
    let q1_r_and_z3_r = { ValueCastTo::<u64>::cast(&q1_r) & ValueCastTo::<u64>::cast(&z3_r) };
    let q0_r_or_q1_r =
      { ValueCastTo::<u64>::cast(&q0_r_and_q1_r) | ValueCastTo::<u64>::cast(&q1_r_and_z3_r) };
    let q0_r_and_z3_r = { ValueCastTo::<u64>::cast(&q0_r) & ValueCastTo::<u64>::cast(&z3_r) };
    let q0_r_or_q0_r =
      { ValueCastTo::<u64>::cast(&q0_r_or_q1_r) | ValueCastTo::<u64>::cast(&q0_r_and_z3_r) };
    let q0_r_shl = { ValueCastTo::<u64>::cast(&q0_r_or_q0_r) << ValueCastTo::<u64>::cast(&1u64) };
    let c_28 = { ValueCastTo::<u64>::cast(&q0_r_shl) };
    let s_28_xor_c_28 = { ValueCastTo::<u64>::cast(&s_28) ^ ValueCastTo::<u64>::cast(&c_28) };
    let s_28_xor_w4_r =
      { ValueCastTo::<u64>::cast(&s_28_xor_c_28) ^ ValueCastTo::<u64>::cast(&w4_r) };
    let s_29 = { ValueCastTo::<u64>::cast(&s_28_xor_w4_r) };
    let s_28_and_c_28 = { ValueCastTo::<u64>::cast(&s_28) & ValueCastTo::<u64>::cast(&c_28) };
    let c_28_and_w4_r = { ValueCastTo::<u64>::cast(&c_28) & ValueCastTo::<u64>::cast(&w4_r) };
    let s_28_or_c_28 =
      { ValueCastTo::<u64>::cast(&s_28_and_c_28) | ValueCastTo::<u64>::cast(&c_28_and_w4_r) };
    let s_28_and_w4_r = { ValueCastTo::<u64>::cast(&s_28) & ValueCastTo::<u64>::cast(&w4_r) };
    let s_28_or_s_28 =
      { ValueCastTo::<u64>::cast(&s_28_or_c_28) | ValueCastTo::<u64>::cast(&s_28_and_w4_r) };
    let s_28_shl = { ValueCastTo::<u64>::cast(&s_28_or_s_28) << ValueCastTo::<u64>::cast(&1u64) };
    let c_29 = { ValueCastTo::<u64>::cast(&s_28_shl) };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:832
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, s_29.clone(), "ExecuteStageInstance");
      sim.mul_stage1_sum.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:833
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, c_29.clone(), "ExecuteStageInstance");
      sim.mul_stage1_carry.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:834
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, 3u8.clone(), "ExecuteStageInstance");
      sim.mul_cycle_counter.write(0, write);
    };
  }
  let mul_cycle_eq_3 = { ValueCastTo::<u8>::cast(&mul_cycle) == ValueCastTo::<u8>::cast(&3u8) };
  if mul_cycle_eq_3 {
    let mul_stage1_rd_2 = { sim.mul_stage1_sum.payload[false as usize].clone() };
    let mul_stage1_rd_3 = { sim.mul_stage1_carry.payload[false as usize].clone() };
    let final_result =
      { ValueCastTo::<u64>::cast(&mul_stage1_rd_2) + ValueCastTo::<u64>::cast(&mul_stage1_rd_3) };
    let saved_op_1 = { sim.mul_op_reg.payload[false as usize].clone() };
    let final_result_slice = {
      {
        let a = ValueCastTo::<u64>::cast(&final_result);
        let mask = u64::from_str_radix("11111111111111111111111111111111", 2).unwrap();
        let res = (a >> 0) & mask;
        ValueCastTo::<u32>::cast(&res)
      }
    };
    let result_low = { ValueCastTo::<u32>::cast(&final_result_slice) };
    let final_result_slice_1 = {
      {
        let a = ValueCastTo::<u64>::cast(&final_result);
        let mask = u64::from_str_radix("11111111111111111111111111111111", 2).unwrap();
        let res = (a >> 32) & mask;
        ValueCastTo::<u32>::cast(&res)
      }
    };
    let result_high = { ValueCastTo::<u32>::cast(&final_result_slice_1) };
    let saved_op_eq_5 = { ValueCastTo::<u8>::cast(&saved_op_1) == ValueCastTo::<u8>::cast(&1u8) };
    let mul_result_val = {
      if saved_op_eq_5 {
        result_low
      } else {
        result_high
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:849
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, mul_result_val.clone(), "ExecuteStageInstance");
      sim.mul_result_reg.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:850
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, true.clone(), "ExecuteStageInstance");
      sim.mul_valid.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:851
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, 0u8.clone(), "ExecuteStageInstance");
      sim.mul_cycle_counter.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:852
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, false.clone(), "ExecuteStageInstance");
      sim.mul_in_progress.write(0, write);
    };
  }
  let mul_stage1_rd_4 = { sim.mul_stage1_sum.payload[false as usize].clone() };
  let mul_stage1_rd_5 = { sim.mul_stage1_carry.payload[false as usize].clone() };
  let current_final_result =
    { ValueCastTo::<u64>::cast(&mul_stage1_rd_4) + ValueCastTo::<u64>::cast(&mul_stage1_rd_5) };
  let current_final_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&current_final_result);
      let mask = u64::from_str_radix("11111111111111111111111111111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u32>::cast(&res)
    }
  };
  let current_result_low = { ValueCastTo::<u32>::cast(&current_final_slice) };
  let current_final_slice_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&current_final_result);
      let mask = u64::from_str_radix("11111111111111111111111111111111", 2).unwrap();
      let res = (a >> 32) & mask;
      ValueCastTo::<u32>::cast(&res)
    }
  };
  let current_result_high = { ValueCastTo::<u32>::cast(&current_final_slice_1) };
  let current_saved_op = { sim.mul_op_reg.payload[false as usize].clone() };
  let current_saved_eq =
    { ValueCastTo::<u8>::cast(&current_saved_op) == ValueCastTo::<u8>::cast(&1u8) };
  let current_mul_result = {
    if current_saved_eq {
      current_result_low
    } else {
      current_result_high
    }
  };
  let mul_cycle_eq_4 = { ValueCastTo::<u8>::cast(&mul_cycle) == ValueCastTo::<u8>::cast(&0u8) };
  if mul_cycle_eq_4 {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:864
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, false.clone(), "ExecuteStageInstance");
      sim.mul_valid.write(0, write);
    };
  }
  let div_state_val = { sim.div_state.payload[false as usize].clone() };
  let div_state_neq = { ValueCastTo::<u8>::cast(&div_state_val) != ValueCastTo::<u8>::cast(&0u8) };
  let div_busy = {
    if div_state_neq {
      true
    } else {
      false
    }
  };
  let div_state_eq = { ValueCastTo::<u8>::cast(&div_state_val) == ValueCastTo::<u8>::cast(&35u8) };
  let div_done = {
    if div_state_eq {
      true
    } else {
      false
    }
  };
  let is_div_and_id_ex =
    { ValueCastTo::<bool>::cast(&is_div_inst) & ValueCastTo::<bool>::cast(&id_ex_rd_6) };
  let not_div_busy = { !div_busy };
  let is_div_and_not_div =
    { ValueCastTo::<bool>::cast(&is_div_and_id_ex) & ValueCastTo::<bool>::cast(&not_div_busy) };
  let start_new_div = {
    if is_div_and_not_div {
      true
    } else {
      false
    }
  };
  if start_new_div {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:878
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, alu_a.clone(), "ExecuteStageInstance");
      sim.div_dividend.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:879
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, rs2_data_2.clone(), "ExecuteStageInstance");
      sim.div_divisor.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:880
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, div_op.clone(), "ExecuteStageInstance");
      sim.div_op_reg.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:881
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, rd_addr.clone(), "ExecuteStageInstance");
      sim.div_rd_reg.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:882
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, control_in_2.clone(), "ExecuteStageInstance");
      sim.div_control_reg.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:883
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, pc_in_1.clone(), "ExecuteStageInstance");
      sim.div_pc_reg.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:884
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, 1u8.clone(), "ExecuteStageInstance");
      sim.div_state.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:885
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, 0u8.clone(), "ExecuteStageInstance");
      sim.div_iter_count.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:886
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, false.clone(), "ExecuteStageInstance");
      sim.div_sign.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:887
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, false.clone(), "ExecuteStageInstance");
      sim.div_neg_result.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:888
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, false.clone(), "ExecuteStageInstance");
      sim.div_valid.write(0, write);
    };
  }
  let div_state_eq_1 = { ValueCastTo::<u8>::cast(&div_state_val) == ValueCastTo::<u8>::cast(&1u8) };
  let not_start_new = { !start_new_div };
  let div_state_and_not_start =
    { ValueCastTo::<bool>::cast(&div_state_eq_1) & ValueCastTo::<bool>::cast(&not_start_new) };
  if div_state_and_not_start {
    let saved_op_2 = { sim.div_op_reg.payload[false as usize].clone() };
    let rem_result_val = { sim.div_dividend.payload[false as usize].clone() };
    let divisor = { sim.div_divisor.payload[false as usize].clone() };
    let div_zero = { ValueCastTo::<u32>::cast(&divisor) == ValueCastTo::<u32>::cast(&0u32) };
    let saved_op_eq_6 = { ValueCastTo::<u8>::cast(&saved_op_2) == ValueCastTo::<u8>::cast(&1u8) };
    let saved_op_eq_7 = { ValueCastTo::<u8>::cast(&saved_op_2) == ValueCastTo::<u8>::cast(&3u8) };
    let saved_op_or_saved_op_3 =
      { ValueCastTo::<bool>::cast(&saved_op_eq_6) | ValueCastTo::<bool>::cast(&saved_op_eq_7) };
    let is_signed = {
      if saved_op_or_saved_op_3 {
        true
      } else {
        false
      }
    };
    let dividend_sign = {
      {
        let a = ValueCastTo::<u64>::cast(&rem_result_val);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 31) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let divisor_sign = {
      {
        let a = ValueCastTo::<u64>::cast(&divisor);
        let mask = u64::from_str_radix("1", 2).unwrap();
        let res = (a >> 31) & mask;
        ValueCastTo::<bool>::cast(&res)
      }
    };
    let dividend_sign_xor_divisor =
      { ValueCastTo::<bool>::cast(&dividend_sign) ^ ValueCastTo::<bool>::cast(&divisor_sign) };
    let is_signed_and_dividend_si = {
      ValueCastTo::<bool>::cast(&is_signed) & ValueCastTo::<bool>::cast(&dividend_sign_xor_divisor)
    };
    let result_sign = {
      if is_signed_and_dividend_si {
        true
      } else {
        false
      }
    };
    let dividend_signed = { ValueCastTo::<i32>::cast(&rem_result_val) };
    let divisor_signed = { ValueCastTo::<i32>::cast(&divisor) };
    let dividend_sign_eq =
      { ValueCastTo::<bool>::cast(&dividend_sign) == ValueCastTo::<bool>::cast(&true) };
    let not_dividend = { !rem_result_val };
    let not_dividend_cast = { ValueCastTo::<u32>::cast(&not_dividend) };
    let not_dividend_add =
      { ValueCastTo::<u32>::cast(&not_dividend_cast) + ValueCastTo::<u32>::cast(&1u32) };
    let dividend_abs = {
      if dividend_sign_eq {
        not_dividend_add
      } else {
        rem_result_val
      }
    };
    let divisor_sign_eq =
      { ValueCastTo::<bool>::cast(&divisor_sign) == ValueCastTo::<bool>::cast(&true) };
    let not_divisor = { !divisor };
    let not_divisor_cast = { ValueCastTo::<u32>::cast(&not_divisor) };
    let not_divisor_add =
      { ValueCastTo::<u32>::cast(&not_divisor_cast) + ValueCastTo::<u32>::cast(&1u32) };
    let divisor_abs = {
      if divisor_sign_eq {
        not_divisor_add
      } else {
        divisor
      }
    };
    if div_zero {
      let div_result_val = {
        if is_signed {
          4294967295u32
        } else {
          4294967295u32
        }
      };
      let saved_op_eq_8 = { ValueCastTo::<u8>::cast(&saved_op_2) == ValueCastTo::<u8>::cast(&1u8) };
      let saved_op_eq_9 = { ValueCastTo::<u8>::cast(&saved_op_2) == ValueCastTo::<u8>::cast(&2u8) };
      let saved_op_or_saved_op_4 =
        { ValueCastTo::<bool>::cast(&saved_op_eq_8) | ValueCastTo::<bool>::cast(&saved_op_eq_9) };
      let final_div_result = {
        if saved_op_or_saved_op_4 {
          div_result_val
        } else {
          rem_result_val
        }
      };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:926
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write =
          ArrayWrite::new(stamp, false as usize, final_div_result.clone(), "ExecuteStageInstance");
        sim.div_result_reg.write(0, write);
      };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:927
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write = ArrayWrite::new(stamp, false as usize, true.clone(), "ExecuteStageInstance");
        sim.div_valid.write(0, write);
      };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:928
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write = ArrayWrite::new(stamp, false as usize, 35u8.clone(), "ExecuteStageInstance");
        sim.div_state.write(0, write);
      };
    }
    let not_div_zero = { !div_zero };
    if not_div_zero {
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:934
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write =
          ArrayWrite::new(stamp, false as usize, divisor_abs.clone(), "ExecuteStageInstance");
        sim.div_divisor.write(0, write);
      };
      let dividend_abs_cast = { ValueCastTo::<u64>::cast(&dividend_abs) };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:936
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write =
          ArrayWrite::new(stamp, false as usize, dividend_abs_cast.clone(), "ExecuteStageInstance");
        sim.div_remainder.write(0, write);
      };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:937
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write = ArrayWrite::new(stamp, false as usize, 0u32.clone(), "ExecuteStageInstance");
        sim.div_quotient.write(0, write);
      };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:939
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write =
          ArrayWrite::new(stamp, false as usize, result_sign.clone(), "ExecuteStageInstance");
        sim.div_sign.write(0, write);
      };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:940
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write =
          ArrayWrite::new(stamp, false as usize, result_sign.clone(), "ExecuteStageInstance");
        sim.div_neg_result.write(0, write);
      };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:941
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write = ArrayWrite::new(stamp, false as usize, 2u8.clone(), "ExecuteStageInstance");
        sim.div_state.write(0, write);
      };
    }
  }
  let div_state_ge = { ValueCastTo::<u8>::cast(&div_state_val) >= ValueCastTo::<u8>::cast(&2u8) };
  let div_state_lt = { ValueCastTo::<u8>::cast(&div_state_val) < ValueCastTo::<u8>::cast(&34u8) };
  let div_state_and_div_state =
    { ValueCastTo::<bool>::cast(&div_state_ge) & ValueCastTo::<bool>::cast(&div_state_lt) };
  let not_start_new_1 = { !start_new_div };
  let div_state_in_iterate = {
    ValueCastTo::<bool>::cast(&div_state_and_div_state)
      & ValueCastTo::<bool>::cast(&not_start_new_1)
  };
  if div_state_in_iterate {
    let iter_num = { sim.div_iter_count.payload[false as usize].clone() };
    let current_remainder = { sim.div_remainder.payload[false as usize].clone() };
    let current_divisor = { sim.div_divisor.payload[false as usize].clone() };
    let current_quotient = { sim.div_quotient.payload[false as usize].clone() };
    let current_remainder_shl =
      { ValueCastTo::<u64>::cast(&current_remainder) << ValueCastTo::<u64>::cast(&1u64) };
    let shifted_remainder = { ValueCastTo::<u64>::cast(&current_remainder_shl) };
    let shifted_remainder_slice = {
      {
        let a = ValueCastTo::<u64>::cast(&shifted_remainder);
        let mask = u64::from_str_radix("11111111111111111111111111111111", 2).unwrap();
        let res = (a >> 32) & mask;
        ValueCastTo::<u32>::cast(&res)
      }
    };
    let remainder_part = { ValueCastTo::<u32>::cast(&shifted_remainder_slice) };
    let can_subtract =
      { ValueCastTo::<u32>::cast(&remainder_part) >= ValueCastTo::<u32>::cast(&current_divisor) };
    let remainder_part_sub_curren =
      { ValueCastTo::<u32>::cast(&remainder_part) - ValueCastTo::<u32>::cast(&current_divisor) };
    let new_remainder_part = {
      if can_subtract {
        remainder_part_sub_curren
      } else {
        remainder_part
      }
    };
    let new_remainder_cast = { ValueCastTo::<u32>::cast(&new_remainder_part) };
    let shifted_remainder_slice_1 = {
      {
        let a = ValueCastTo::<u64>::cast(&shifted_remainder);
        let mask = u64::from_str_radix("11111111111111111111111111111111", 2).unwrap();
        let res = (a >> 0) & mask;
        ValueCastTo::<u32>::cast(&res)
      }
    };
    let new_remainder_cat_shifted = {
      {
        let a = ValueCastTo::<BigUint>::cast(&new_remainder_cast);
        let b = ValueCastTo::<BigUint>::cast(&shifted_remainder_slice_1);
        let c = (a << 32) | b;
        ValueCastTo::<u64>::cast(&c)
      }
    };
    let new_remainder = { ValueCastTo::<u64>::cast(&new_remainder_cat_shifted) };
    let current_quotient_shl =
      { ValueCastTo::<u32>::cast(&current_quotient) << ValueCastTo::<u32>::cast(&1u32) };
    let can_subtract_mux_1 = {
      if can_subtract {
        1u32
      } else {
        0u32
      }
    };
    let current_quotient_or_can_s = {
      ValueCastTo::<u32>::cast(&current_quotient_shl)
        | ValueCastTo::<u32>::cast(&can_subtract_mux_1)
    };
    let new_quotient = { ValueCastTo::<u32>::cast(&current_quotient_or_can_s) };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:977
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, new_remainder.clone(), "ExecuteStageInstance");
      sim.div_remainder.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:978
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, new_quotient.clone(), "ExecuteStageInstance");
      sim.div_quotient.write(0, write);
    };
    let iter_num_add = { ValueCastTo::<u8>::cast(&iter_num) + ValueCastTo::<u8>::cast(&1u8) };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:979
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, iter_num_add.clone(), "ExecuteStageInstance");
      sim.div_iter_count.write(0, write);
    };
    let iter_done = { ValueCastTo::<u8>::cast(&iter_num) >= ValueCastTo::<u8>::cast(&31u8) };
    let div_state_add = { ValueCastTo::<u8>::cast(&div_state_val) + ValueCastTo::<u8>::cast(&1u8) };
    let iter_done_mux = {
      if iter_done {
        34u8
      } else {
        div_state_add
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:983
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, iter_done_mux.clone(), "ExecuteStageInstance");
      sim.div_state.write(0, write);
    };
  }
  let div_state_eq_2 =
    { ValueCastTo::<u8>::cast(&div_state_val) == ValueCastTo::<u8>::cast(&34u8) };
  let not_start_new_2 = { !start_new_div };
  let div_state_and_not_start_2 =
    { ValueCastTo::<bool>::cast(&div_state_eq_2) & ValueCastTo::<bool>::cast(&not_start_new_2) };
  if div_state_and_not_start_2 {
    let saved_op_3 = { sim.div_op_reg.payload[false as usize].clone() };
    let quotient_corrected = { sim.div_quotient.payload[false as usize].clone() };
    let current_remainder_1 = { sim.div_remainder.payload[false as usize].clone() };
    let neg_result = { sim.div_neg_result.payload[false as usize].clone() };
    let quotient_signed = { ValueCastTo::<i32>::cast(&quotient_corrected) };
    let not_quotient_signed = { !quotient_signed };
    let quotient_neg =
      { ValueCastTo::<u32>::cast(&not_quotient_signed) + ValueCastTo::<u32>::cast(&1i32) };
    let quotient_neg_cast = { ValueCastTo::<u32>::cast(&quotient_neg) };
    let quotient_final = {
      if neg_result {
        quotient_neg_cast
      } else {
        quotient_corrected
      }
    };
    let current_remainder_slice = {
      {
        let a = ValueCastTo::<u64>::cast(&current_remainder_1);
        let mask = u64::from_str_radix("11111111111111111111111111111111", 2).unwrap();
        let res = (a >> 32) & mask;
        ValueCastTo::<u32>::cast(&res)
      }
    };
    let remainder_high = { ValueCastTo::<u32>::cast(&current_remainder_slice) };
    let dividend_sign_saved = { sim.div_sign.payload[false as usize].clone() };
    let remainder_signed = { ValueCastTo::<i32>::cast(&remainder_high) };
    let not_remainder_signed = { !remainder_signed };
    let remainder_neg =
      { ValueCastTo::<u32>::cast(&not_remainder_signed) + ValueCastTo::<u32>::cast(&1i32) };
    let remainder_neg_cast = { ValueCastTo::<u32>::cast(&remainder_neg) };
    let remainder_final = {
      if neg_result {
        remainder_neg_cast
      } else {
        remainder_high
      }
    };
    let saved_op_eq_10 = { ValueCastTo::<u8>::cast(&saved_op_3) == ValueCastTo::<u8>::cast(&1u8) };
    let saved_op_eq_11 = { ValueCastTo::<u8>::cast(&saved_op_3) == ValueCastTo::<u8>::cast(&2u8) };
    let saved_op_or_saved_op_5 =
      { ValueCastTo::<bool>::cast(&saved_op_eq_10) | ValueCastTo::<bool>::cast(&saved_op_eq_11) };
    let is_div_op = {
      if saved_op_or_saved_op_5 {
        true
      } else {
        false
      }
    };
    let final_div_result_1 = {
      if is_div_op {
        quotient_final
      } else {
        remainder_final
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1019
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, final_div_result_1.clone(), "ExecuteStageInstance");
      sim.div_result_reg.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1020
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, true.clone(), "ExecuteStageInstance");
      sim.div_valid.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1021
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, 35u8.clone(), "ExecuteStageInstance");
      sim.div_state.write(0, write);
    };
  }
  let div_state_eq_3 =
    { ValueCastTo::<u8>::cast(&div_state_val) == ValueCastTo::<u8>::cast(&35u8) };
  let not_start_new_3 = { !start_new_div };
  let div_state_and_not_start_3 =
    { ValueCastTo::<bool>::cast(&div_state_eq_3) & ValueCastTo::<bool>::cast(&not_start_new_3) };
  if div_state_and_not_start_3 {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1025
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, 0u8.clone(), "ExecuteStageInstance");
      sim.div_state.write(0, write);
    };
  }
  let div_state_eq_4 = { ValueCastTo::<u8>::cast(&div_state_val) == ValueCastTo::<u8>::cast(&0u8) };
  let not_start_new_4 = { !start_new_div };
  let div_state_and_not_start_4 =
    { ValueCastTo::<bool>::cast(&div_state_eq_4) & ValueCastTo::<bool>::cast(&not_start_new_4) };
  if div_state_and_not_start_4 {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1029
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, false.clone(), "ExecuteStageInstance");
      sim.div_valid.write(0, write);
    };
  }
  let is_jump_or_is_jumpr_2 =
    { ValueCastTo::<bool>::cast(&is_jump) | ValueCastTo::<bool>::cast(&is_jumpr) };
  let pc_in_add_2 = { ValueCastTo::<u32>::cast(&pc_in_1) + ValueCastTo::<u32>::cast(&4u32) };
  let alu_a_cast_1 = { ValueCastTo::<i32>::cast(&alu_a_5) };
  let alu_b_cast = { ValueCastTo::<i32>::cast(&alu_b_4) };
  let alu_op_eq = { ValueCastTo::<u8>::cast(&alu_op) == ValueCastTo::<u8>::cast(&0u8) };
  let alu_a_add_alu_b_1 =
    { ValueCastTo::<u32>::cast(&alu_a_5) + ValueCastTo::<u32>::cast(&alu_b_4) };
  let alu_op_mux = {
    if alu_op_eq {
      alu_a_add_alu_b_1
    } else {
      0u32
    }
  };
  let alu_op_eq_1 = { ValueCastTo::<u8>::cast(&alu_op) == ValueCastTo::<u8>::cast(&1u8) };
  let alu_a_sub_alu_b = { ValueCastTo::<u32>::cast(&alu_a_5) - ValueCastTo::<u32>::cast(&alu_b_4) };
  let alu_op_mux_1 = {
    if alu_op_eq_1 {
      alu_a_sub_alu_b
    } else {
      alu_op_mux
    }
  };
  let alu_op_eq_2 = { ValueCastTo::<u8>::cast(&alu_op) == ValueCastTo::<u8>::cast(&2u8) };
  let alu_b_and = { ValueCastTo::<u32>::cast(&alu_b_4) & ValueCastTo::<u32>::cast(&31u32) };
  let alu_a_shl_alu_b =
    { ValueCastTo::<u32>::cast(&alu_a_5) << ValueCastTo::<u32>::cast(&alu_b_and) };
  let alu_a_cast_2 = { ValueCastTo::<u32>::cast(&alu_a_shl_alu_b) };
  let alu_op_mux_2 = {
    if alu_op_eq_2 {
      alu_a_cast_2
    } else {
      alu_op_mux_1
    }
  };
  let alu_op_eq_3 = { ValueCastTo::<u8>::cast(&alu_op) == ValueCastTo::<u8>::cast(&3u8) };
  let alu_a_lt_alu_b =
    { ValueCastTo::<i32>::cast(&alu_a_cast_1) < ValueCastTo::<i32>::cast(&alu_b_cast) };
  let alu_a_mux_7 = {
    if alu_a_lt_alu_b {
      1u32
    } else {
      0u32
    }
  };
  let alu_op_mux_3 = {
    if alu_op_eq_3 {
      alu_a_mux_7
    } else {
      alu_op_mux_2
    }
  };
  let alu_op_eq_4 = { ValueCastTo::<u8>::cast(&alu_op) == ValueCastTo::<u8>::cast(&4u8) };
  let alu_a_xor_alu_b = { ValueCastTo::<u32>::cast(&alu_a_5) ^ ValueCastTo::<u32>::cast(&alu_b_4) };
  let alu_a_cast_3 = { ValueCastTo::<u32>::cast(&alu_a_xor_alu_b) };
  let alu_op_mux_4 = {
    if alu_op_eq_4 {
      alu_a_cast_3
    } else {
      alu_op_mux_3
    }
  };
  let alu_op_eq_5 = { ValueCastTo::<u8>::cast(&alu_op) == ValueCastTo::<u8>::cast(&5u8) };
  let alu_b_and_1 = { ValueCastTo::<u32>::cast(&alu_b_4) & ValueCastTo::<u32>::cast(&31u32) };
  let alu_a_shr_alu_b =
    { ValueCastTo::<u32>::cast(&alu_a_5) >> ValueCastTo::<u32>::cast(&alu_b_and_1) };
  let alu_a_cast_4 = { ValueCastTo::<u32>::cast(&alu_a_shr_alu_b) };
  let alu_op_mux_5 = {
    if alu_op_eq_5 {
      alu_a_cast_4
    } else {
      alu_op_mux_4
    }
  };
  let alu_op_eq_6 = { ValueCastTo::<u8>::cast(&alu_op) == ValueCastTo::<u8>::cast(&6u8) };
  let alu_b_and_2 = { ValueCastTo::<u32>::cast(&alu_b_4) & ValueCastTo::<u32>::cast(&31u32) };
  let alu_a_shr_alu_b_1 =
    { ValueCastTo::<i32>::cast(&alu_a_cast_1) >> ValueCastTo::<i32>::cast(&alu_b_and_2) };
  let alu_a_cast_5 = { ValueCastTo::<u32>::cast(&alu_a_shr_alu_b_1) };
  let alu_op_mux_6 = {
    if alu_op_eq_6 {
      alu_a_cast_5
    } else {
      alu_op_mux_5
    }
  };
  let alu_op_eq_7 = { ValueCastTo::<u8>::cast(&alu_op) == ValueCastTo::<u8>::cast(&7u8) };
  let alu_a_lt_alu_b_1 =
    { ValueCastTo::<u32>::cast(&alu_a_5) < ValueCastTo::<u32>::cast(&alu_b_4) };
  let alu_a_mux_8 = {
    if alu_a_lt_alu_b_1 {
      1u32
    } else {
      0u32
    }
  };
  let alu_op_mux_7 = {
    if alu_op_eq_7 {
      alu_a_mux_8
    } else {
      alu_op_mux_6
    }
  };
  let alu_op_eq_8 = { ValueCastTo::<u8>::cast(&alu_op) == ValueCastTo::<u8>::cast(&8u8) };
  let alu_a_or_alu_b = { ValueCastTo::<u32>::cast(&alu_a_5) | ValueCastTo::<u32>::cast(&alu_b_4) };
  let alu_a_cast_6 = { ValueCastTo::<u32>::cast(&alu_a_or_alu_b) };
  let alu_op_mux_8 = {
    if alu_op_eq_8 {
      alu_a_cast_6
    } else {
      alu_op_mux_7
    }
  };
  let alu_op_eq_9 = { ValueCastTo::<u8>::cast(&alu_op) == ValueCastTo::<u8>::cast(&9u8) };
  let alu_a_and_alu_b = { ValueCastTo::<u32>::cast(&alu_a_5) & ValueCastTo::<u32>::cast(&alu_b_4) };
  let alu_a_cast_7 = { ValueCastTo::<u32>::cast(&alu_a_and_alu_b) };
  let alu_op_mux_9 = {
    if alu_op_eq_9 {
      alu_a_cast_7
    } else {
      alu_op_mux_8
    }
  };
  let is_jump_mux_2 = {
    if is_jump_or_is_jumpr_2 {
      pc_in_add_2
    } else {
      alu_op_mux_9
    }
  };
  let normal_alu_result = {
    if is_branch {
      0u32
    } else {
      is_jump_mux_2
    }
  };
  let div_result_val_1 = { sim.div_result_reg.payload[false as usize].clone() };
  let mul_done_mux = {
    if mul_done {
      current_mul_result
    } else {
      normal_alu_result
    }
  };
  let alu_result_1 = {
    if div_done {
      div_result_val_1
    } else {
      mul_done_mux
    }
  };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1039
  print!("@line:{:<5} {:<10}: [ExecuteStageInstance]\t", line!(), cyclize(sim.stamp));
  println!("EX RESULT: PC={:08x}, alu_op={:05b}, alu_a={:08x}, alu_b={:08x}, normal_alu_result={:08x}, final_alu_result={:08x}", pc_in_1, alu_op, alu_a_5, alu_b_4, normal_alu_result, alu_result_1, );
  let is_branch_or_is_jump =
    { ValueCastTo::<bool>::cast(&is_branch) | ValueCastTo::<bool>::cast(&is_jump) };
  let target_pc_1 = {
    if is_branch_or_is_jump {
      actual_target_pc
    } else {
      target_pc
    }
  };
  let new_pc_cast = { ValueCastTo::<u32>::cast(&new_pc) };
  let target_pc_2 = {
    if is_jumpr {
      new_pc_cast
    } else {
      target_pc_1
    }
  };
  let mispredict_or_is_jump =
    { ValueCastTo::<bool>::cast(&mispredict) | ValueCastTo::<bool>::cast(&is_jump) };
  let mispredict_or_or_is_jumpr =
    { ValueCastTo::<bool>::cast(&mispredict_or_is_jump) | ValueCastTo::<bool>::cast(&is_jumpr) };
  let pc_change_1 = {
    if mispredict_or_or_is_jumpr {
      true
    } else {
      false
    }
  };
  let alu_b_eq_alu_result = { ValueCastTo::<u32>::cast(&alu_b) == ValueCastTo::<u32>::cast(&0u32) };
  let is_jump_and_alu_b =
    { ValueCastTo::<bool>::cast(&is_jump) & ValueCastTo::<bool>::cast(&alu_b_eq_alu_result) };
  if is_jump_and_alu_b {
    let reg_file_rd_2 = { sim.reg_file.payload[10u8 as usize].clone() };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1062
    print!("@line:{:<5} {:<10}: [ExecuteStageInstance]\t", line!(), cyclize(sim.stamp));
    println!("Finish Execution. The result is {}", reg_file_rd_2,);
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1063
    std::process::exit(0);
  }
  let mul_in_ex_stage =
    { ValueCastTo::<bool>::cast(&is_mul_inst) & ValueCastTo::<bool>::cast(&id_ex_rd_6) };
  let not_mul_done = { !mul_done };
  let mul_wait =
    { ValueCastTo::<bool>::cast(&mul_in_ex_stage) & ValueCastTo::<bool>::cast(&not_mul_done) };
  let div_in_ex_stage =
    { ValueCastTo::<bool>::cast(&is_div_inst) & ValueCastTo::<bool>::cast(&id_ex_rd_6) };
  let not_div_done = { !div_done };
  let div_wait =
    { ValueCastTo::<bool>::cast(&div_in_ex_stage) & ValueCastTo::<bool>::cast(&not_div_done) };
  let mul_control = { sim.mul_control_reg.payload[false as usize].clone() };
  let mul_pc = { sim.mul_pc_reg.payload[false as usize].clone() };
  let div_control = { sim.div_control_reg.payload[false as usize].clone() };
  let div_pc = { sim.div_pc_reg.payload[false as usize].clone() };
  let not_mul_wait = { !mul_wait };
  let id_ex_and_not_mul =
    { ValueCastTo::<bool>::cast(&id_ex_rd_6) & ValueCastTo::<bool>::cast(&not_mul_wait) };
  let not_div_wait = { !div_wait };
  let should_pass =
    { ValueCastTo::<bool>::cast(&id_ex_and_not_mul) & ValueCastTo::<bool>::cast(&not_div_wait) };
  let should_pass_or_mul_done =
    { ValueCastTo::<bool>::cast(&should_pass) | ValueCastTo::<bool>::cast(&mul_done) };
  let pass_or_done =
    { ValueCastTo::<bool>::cast(&should_pass_or_mul_done) | ValueCastTo::<bool>::cast(&div_done) };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1099
  print!("@line:{:<5} {:<10}: [ExecuteStageInstance]\t", line!(), cyclize(sim.stamp));
  println!("EX PASS LOGIC: PC={:08x}, id_ex_valid={}, should_pass={}, mul_wait={}, div_wait={}, mul_done={}, div_done={}, pass_or_done={}", pc_in_1, if id_ex_rd_6 { 1 } else { 0 }, if should_pass { 1 } else { 0 }, if mul_wait { 1 } else { 0 }, if div_wait { 1 } else { 0 }, if mul_done { 1 } else { 0 }, if div_done { 1 } else { 0 }, if pass_or_done { 1 } else { 0 }, );
  let div_done_mux_1 = {
    if div_done {
      div_pc
    } else {
      pc_in_1
    }
  };
  let final_pc = {
    if mul_done {
      mul_pc
    } else {
      div_done_mux_1
    }
  };
  let div_done_mux_2 = {
    if div_done {
      div_control
    } else {
      control_in_2
    }
  };
  let final_control = {
    if mul_done {
      mul_control
    } else {
      div_done_mux_2
    }
  };
  if ex_mem_rd_9 {
    let pass_or_mux = {
      if pass_or_done {
        final_pc
      } else {
        0u32
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1109
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, pass_or_mux.clone(), "ExecuteStageInstance");
      sim.ex_mem_pc.write(0, write);
    };
    let pass_or_mux_1 = {
      if pass_or_done {
        final_control
      } else {
        0u64
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1110
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, pass_or_mux_1.clone(), "ExecuteStageInstance");
      sim.ex_mem_control.write(0, write);
    };
    let pass_or_mux_2 = {
      if pass_or_done {
        alu_result_1
      } else {
        0u32
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1111
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, pass_or_mux_2.clone(), "ExecuteStageInstance");
      sim.ex_mem_result.write(0, write);
    };
    let pass_or_mux_3 = {
      if pass_or_done {
        rs2_data_2
      } else {
        0u32
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1112
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, pass_or_mux_3.clone(), "ExecuteStageInstance");
      sim.ex_mem_data.write(0, write);
    };
    let ex_mem_rd_10 = { sim.ex_mem_data.payload[false as usize].clone() };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1113
    print!("@line:{:<5} {:<10}: [ExecuteStageInstance]\t", line!(), cyclize(sim.stamp));
    println!(
      "EX DATA PATH: PC={:08x}, rs2_data={:08x}, pass_or_done={}, ex_mem_data={:08x}",
      pc_in_1,
      rs2_data_2,
      if pass_or_done { 1 } else { 0 },
      ex_mem_rd_10,
    );
  }
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1118
  ();
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1118
  {
    let stamp = sim.stamp - sim.stamp % 100 + 100;
    sim.MemoryStageInstance_event.push_back(stamp)
  };
  let is_jumpr_cast = { ValueCastTo::<bool>::cast(&is_jumpr) };
  let is_jump_cast = { ValueCastTo::<bool>::cast(&is_jump) };
  let is_branch_cast = { ValueCastTo::<bool>::cast(&is_branch) };
  let pc_in_cast = { ValueCastTo::<u32>::cast(&pc_in_1) };
  let predict_taken_cast = { ValueCastTo::<bool>::cast(&predict_taken) };
  let btb_hit_cast = { ValueCastTo::<bool>::cast(&btb_hit) };
  let actual_target_cast = { ValueCastTo::<u32>::cast(&actual_target_pc) };
  let actual_taken_cast = { ValueCastTo::<bool>::cast(&actual_taken) };
  let correct_pc_cast = { ValueCastTo::<u32>::cast(&correct_pc) };
  let mispredict_cast = { ValueCastTo::<bool>::cast(&mispredict) };
  let is_jumpr_cat_is_jump = {
    {
      let a = ValueCastTo::<BigUint>::cast(&is_jumpr_cast);
      let b = ValueCastTo::<BigUint>::cast(&is_jump_cast);
      let c = (a << 1) | b;
      ValueCastTo::<u8>::cast(&c)
    }
  };
  let is_jumpr_cat_is_branch = {
    {
      let a = ValueCastTo::<BigUint>::cast(&is_jumpr_cat_is_jump);
      let b = ValueCastTo::<BigUint>::cast(&is_branch_cast);
      let c = (a << 1) | b;
      ValueCastTo::<u8>::cast(&c)
    }
  };
  let is_jumpr_cat_pc_in = {
    {
      let a = ValueCastTo::<BigUint>::cast(&is_jumpr_cat_is_branch);
      let b = ValueCastTo::<BigUint>::cast(&pc_in_cast);
      let c = (a << 32) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let is_jumpr_cat_predict_take = {
    {
      let a = ValueCastTo::<BigUint>::cast(&is_jumpr_cat_pc_in);
      let b = ValueCastTo::<BigUint>::cast(&predict_taken_cast);
      let c = (a << 1) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let is_jumpr_cat_btb_hit = {
    {
      let a = ValueCastTo::<BigUint>::cast(&is_jumpr_cat_predict_take);
      let b = ValueCastTo::<BigUint>::cast(&btb_hit_cast);
      let c = (a << 1) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let is_jumpr_cat_actual_targe = {
    {
      let a = ValueCastTo::<BigUint>::cast(&is_jumpr_cat_btb_hit);
      let b = ValueCastTo::<BigUint>::cast(&actual_target_cast);
      let c = (a << 32) | b;
      ValueCastTo::<BigUint>::cast(&c)
    }
  };
  let is_jumpr_cat_actual_taken = {
    {
      let a = ValueCastTo::<BigUint>::cast(&is_jumpr_cat_actual_targe.clone());
      let b = ValueCastTo::<BigUint>::cast(&actual_taken_cast);
      let c = (a << 1) | b;
      ValueCastTo::<BigUint>::cast(&c)
    }
  };
  let is_jumpr_cat_correct_pc = {
    {
      let a = ValueCastTo::<BigUint>::cast(&is_jumpr_cat_actual_taken.clone());
      let b = ValueCastTo::<BigUint>::cast(&correct_pc_cast);
      let c = (a << 32) | b;
      ValueCastTo::<BigUint>::cast(&c)
    }
  };
  let prediction_result = {
    {
      let a = ValueCastTo::<BigUint>::cast(&is_jumpr_cat_correct_pc.clone());
      let b = ValueCastTo::<BigUint>::cast(&mispredict_cast);
      let c = (a << 1) | b;
      ValueCastTo::<BigUint>::cast(&c)
    }
  };
  let mul_cycle_eq_5 = { ValueCastTo::<u8>::cast(&mul_cycle) == ValueCastTo::<u8>::cast(&1u8) };
  let mul_cycle_eq_6 = { ValueCastTo::<u8>::cast(&mul_cycle) == ValueCastTo::<u8>::cast(&2u8) };
  let mul_cycle_or_mul_cycle =
    { ValueCastTo::<bool>::cast(&mul_cycle_eq_5) | ValueCastTo::<bool>::cast(&mul_cycle_eq_6) };
  let mul_executing = {
    if mul_cycle_or_mul_cycle {
      true
    } else {
      false
    }
  };
  let is_mul_and_id_ex_2 =
    { ValueCastTo::<bool>::cast(&is_mul_inst) & ValueCastTo::<bool>::cast(&id_ex_rd_6) };
  let is_mul_and_mul_executing =
    { ValueCastTo::<bool>::cast(&is_mul_and_id_ex_2) & ValueCastTo::<bool>::cast(&mul_executing) };
  let mul_stall_needed = {
    if is_mul_and_mul_executing {
      true
    } else {
      false
    }
  };
  let div_state_neq_1 =
    { ValueCastTo::<u8>::cast(&div_state_val) != ValueCastTo::<u8>::cast(&0u8) };
  let div_state_neq_2 =
    { ValueCastTo::<u8>::cast(&div_state_val) != ValueCastTo::<u8>::cast(&35u8) };
  let div_state_and_div_state_1 =
    { ValueCastTo::<bool>::cast(&div_state_neq_1) & ValueCastTo::<bool>::cast(&div_state_neq_2) };
  let div_executing = {
    if div_state_and_div_state_1 {
      true
    } else {
      false
    }
  };
  let is_div_and_id_ex_2 =
    { ValueCastTo::<bool>::cast(&is_div_inst) & ValueCastTo::<bool>::cast(&id_ex_rd_6) };
  let is_div_and_div_executing =
    { ValueCastTo::<bool>::cast(&is_div_and_id_ex_2) & ValueCastTo::<bool>::cast(&div_executing) };
  let div_stall_needed = {
    if is_div_and_div_executing {
      true
    } else {
      false
    }
  };
  let pass_or_mux_4 = {
    if pass_or_done {
      pc_change_1
    } else {
      false
    }
  };
  let ex_mem_rd_11 = { sim.ex_mem_pc_change.payload[false as usize].clone() };
  let out_pc_change = {
    if ex_mem_rd_9 {
      pass_or_mux_4
    } else {
      ex_mem_rd_11
    }
  };
  let pass_or_mux_5 = {
    if pass_or_done {
      target_pc_2
    } else {
      0u32
    }
  };
  let ex_mem_rd_12 = { sim.ex_mem_target_pc.payload[false as usize].clone() };
  let out_target_pc = {
    if ex_mem_rd_9 {
      pass_or_mux_5
    } else {
      ex_mem_rd_12
    }
  };
  let pass_or_mux_6 = {
    if pass_or_done {
      control_in_2
    } else {
      0u64
    }
  };
  let out_control = {
    if ex_mem_rd_9 {
      pass_or_mux_6
    } else {
      mem_control
    }
  };
  let prediction_result_cast = { ValueCastTo::<BigUint>::cast(&prediction_result.clone()) };
  let pass_or_mux_7 = {
    if pass_or_done {
      prediction_result_cast.clone()
    } else {
      ValueCastTo::<BigUint>::cast(&(0 as u64))
    }
  };
  let ex_mem_rd_13 = { sim.ex_mem_prediction_result.payload[false as usize].clone() };
  let out_prediction_result = {
    if ex_mem_rd_9 {
      pass_or_mux_7.clone()
    } else {
      ex_mem_rd_13.clone()
    }
  };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1181
  print!("@line:{:<5} {:<10}: [ExecuteStageInstance]\t", line!(), cyclize(sim.stamp));
  println!("EX SIGNALS: pass_or_done={}, ex_mem_valid={}, pc_change={}, out_pc_change={}, control_in={:012x}, ex_sig_control={:012x}, out_control={:012x}", if pass_or_done { 1 } else { 0 }, if ex_mem_rd_9 { 1 } else { 0 }, if pc_change_1 { 1 } else { 0 }, if out_pc_change { 1 } else { 0 }, control_in_2, mem_control, out_control, );
  if ex_mem_rd_9 {
    let pass_or_mux_8 = {
      if pass_or_done {
        pc_change_1
      } else {
        false
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1186
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, pass_or_mux_8.clone(), "ExecuteStageInstance");
      sim.ex_mem_pc_change.write(0, write);
    };
    let pass_or_mux_9 = {
      if pass_or_done {
        target_pc_2
      } else {
        0u32
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1187
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, pass_or_mux_9.clone(), "ExecuteStageInstance");
      sim.ex_mem_target_pc.write(0, write);
    };
    let prediction_result_cast_1 = { ValueCastTo::<BigUint>::cast(&prediction_result.clone()) };
    let pass_or_mux_10 = {
      if pass_or_done {
        prediction_result_cast_1.clone()
      } else {
        ValueCastTo::<BigUint>::cast(&(0 as u64))
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1188
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(
        stamp,
        false as usize,
        pass_or_mux_10.clone().clone(),
        "ExecuteStageInstance",
      );
      sim.ex_mem_prediction_result.write(0, write);
    };
    let pass_or_mux_11 = {
      if pass_or_done {
        control_in_2
      } else {
        0u64
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1189
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, pass_or_mux_11.clone(), "ExecuteStageInstance");
      sim.ex_sig_control.write(0, write);
    };
  }
  let div_stall_cast = { ValueCastTo::<bool>::cast(&div_stall_needed) };
  let div_done_cast = { ValueCastTo::<bool>::cast(&div_done) };
  let div_busy_cast = { ValueCastTo::<bool>::cast(&div_busy) };
  let mul_stall_cast = { ValueCastTo::<bool>::cast(&mul_stall_needed) };
  let mul_done_cast = { ValueCastTo::<bool>::cast(&mul_done) };
  let mul_busy_cast = { ValueCastTo::<bool>::cast(&mul_busy) };
  let out_prediction_cast = { ValueCastTo::<BigUint>::cast(&out_prediction_result.clone()) };
  let out_control_cast = { ValueCastTo::<u64>::cast(&out_control) };
  let out_target_cast = { ValueCastTo::<u32>::cast(&out_target_pc) };
  let out_pc_cast = { ValueCastTo::<bool>::cast(&out_pc_change) };
  let div_stall_cat_div_done = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_stall_cast);
      let b = ValueCastTo::<BigUint>::cast(&div_done_cast);
      let c = (a << 1) | b;
      ValueCastTo::<u8>::cast(&c)
    }
  };
  let div_stall_cat_div_busy = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_stall_cat_div_done);
      let b = ValueCastTo::<BigUint>::cast(&div_busy_cast);
      let c = (a << 1) | b;
      ValueCastTo::<u8>::cast(&c)
    }
  };
  let div_stall_cat_mul_stall = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_stall_cat_div_busy);
      let b = ValueCastTo::<BigUint>::cast(&mul_stall_cast);
      let c = (a << 1) | b;
      ValueCastTo::<u8>::cast(&c)
    }
  };
  let div_stall_cat_mul_done = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_stall_cat_mul_stall);
      let b = ValueCastTo::<BigUint>::cast(&mul_done_cast);
      let c = (a << 1) | b;
      ValueCastTo::<u8>::cast(&c)
    }
  };
  let div_stall_cat_mul_busy = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_stall_cat_mul_done);
      let b = ValueCastTo::<BigUint>::cast(&mul_busy_cast);
      let c = (a << 1) | b;
      ValueCastTo::<u8>::cast(&c)
    }
  };
  let div_stall_cat_out_predict = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_stall_cat_mul_busy);
      let b = ValueCastTo::<BigUint>::cast(&out_prediction_cast.clone());
      let c = (a << 103) | b;
      ValueCastTo::<BigUint>::cast(&c)
    }
  };
  let div_stall_cat_out_control = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_stall_cat_out_predict.clone());
      let b = ValueCastTo::<BigUint>::cast(&out_control_cast);
      let c = (a << 48) | b;
      ValueCastTo::<BigUint>::cast(&c)
    }
  };
  let div_stall_cat_out_target = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_stall_cat_out_control.clone());
      let b = ValueCastTo::<BigUint>::cast(&out_target_cast);
      let c = (a << 32) | b;
      ValueCastTo::<BigUint>::cast(&c)
    }
  };
  let execute_signals = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_stall_cat_out_target.clone());
      let b = ValueCastTo::<BigUint>::cast(&out_pc_cast);
      let c = (a << 1) | b;
      ValueCastTo::<BigUint>::cast(&c)
    }
  };
  sim.execute_signals_value = Some(execute_signals.clone());

  true
}
