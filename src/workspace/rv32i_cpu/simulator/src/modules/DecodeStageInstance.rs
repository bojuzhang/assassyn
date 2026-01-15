use crate::simulator::Simulator;
use sim_runtime::num_bigint::{BigInt, BigUint};
use sim_runtime::*;
use std::ffi::c_void;

// Elaborating module DecodeStageInstance
pub fn DecodeStageInstance(sim: &mut Simulator) -> bool {
  let if_id_pc_in = { sim.if_id_pc.payload[false as usize].clone() };
  let instruction = { sim.if_id_instruction.payload[false as usize].clone() };
  let prediction_info_in_1 = { sim.if_id_prediction_info.payload[false as usize].clone() };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:183
  print!("@line:{:<5} {:<10}: [DecodeStageInstance]\t", line!(), cyclize(sim.stamp));
  println!("ID: PC={:08x}, Instruction={:08x}", if_id_pc_in, instruction,);
  let opcode = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("1111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let rd = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 7) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let func3 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("111", 2).unwrap();
      let res = (a >> 12) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let rs1 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 15) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let rs2 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 20) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let funct7 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("1111111", 2).unwrap();
      let res = (a >> 25) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let imm_i_bits = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("111111111111", 2).unwrap();
      let res = (a >> 20) & mask;
      ValueCastTo::<u16>::cast(&res)
    }
  };
  let sign_bit_i = {
    {
      let a = ValueCastTo::<u64>::cast(&imm_i_bits);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 11) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let sign_bit_eq = { ValueCastTo::<bool>::cast(&sign_bit_i) == ValueCastTo::<bool>::cast(&true) };
  let cat_imm_i = {
    {
      let a = ValueCastTo::<BigUint>::cast(&1048575u32);
      let b = ValueCastTo::<BigUint>::cast(&imm_i_bits);
      let c = (a << 12) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let cat_imm_cast = { ValueCastTo::<u32>::cast(&cat_imm_i) };
  let cat_imm_i_1 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&0u32);
      let b = ValueCastTo::<BigUint>::cast(&imm_i_bits);
      let c = (a << 12) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let cat_imm_cast_1 = { ValueCastTo::<u32>::cast(&cat_imm_i_1) };
  let immediate_i = {
    if sign_bit_eq {
      cat_imm_cast
    } else {
      cat_imm_cast_1
    }
  };
  let instruction_slice_7 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("1111111", 2).unwrap();
      let res = (a >> 25) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let instruction_slice_8 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 7) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let imm_s_bits = {
    {
      let a = ValueCastTo::<BigUint>::cast(&instruction_slice_7);
      let b = ValueCastTo::<BigUint>::cast(&instruction_slice_8);
      let c = (a << 5) | b;
      ValueCastTo::<u16>::cast(&c)
    }
  };
  let sign_bit_s = {
    {
      let a = ValueCastTo::<u64>::cast(&imm_s_bits);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 11) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let sign_bit_eq_1 =
    { ValueCastTo::<bool>::cast(&sign_bit_s) == ValueCastTo::<bool>::cast(&true) };
  let cat_imm_s = {
    {
      let a = ValueCastTo::<BigUint>::cast(&1048575u32);
      let b = ValueCastTo::<BigUint>::cast(&imm_s_bits);
      let c = (a << 12) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let cat_imm_cast_2 = { ValueCastTo::<u32>::cast(&cat_imm_s) };
  let cat_imm_s_1 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&0u32);
      let b = ValueCastTo::<BigUint>::cast(&imm_s_bits);
      let c = (a << 12) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let cat_imm_cast_3 = { ValueCastTo::<u32>::cast(&cat_imm_s_1) };
  let immediate_s = {
    if sign_bit_eq_1 {
      cat_imm_cast_2
    } else {
      cat_imm_cast_3
    }
  };
  let instruction_slice_9 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 31) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let instruction_slice_10 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 7) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let instruction_slice_11 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("111111", 2).unwrap();
      let res = (a >> 25) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let instruction_slice_12 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("1111", 2).unwrap();
      let res = (a >> 8) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let instruction_slice_cat_ins_1 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&instruction_slice_9);
      let b = ValueCastTo::<BigUint>::cast(&instruction_slice_10);
      let c = (a << 1) | b;
      ValueCastTo::<u8>::cast(&c)
    }
  };
  let instruction_slice_cat_ins_2 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&instruction_slice_cat_ins_1);
      let b = ValueCastTo::<BigUint>::cast(&instruction_slice_11);
      let c = (a << 6) | b;
      ValueCastTo::<u8>::cast(&c)
    }
  };
  let instruction_slice_cat_ins_3 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&instruction_slice_cat_ins_2);
      let b = ValueCastTo::<BigUint>::cast(&instruction_slice_12);
      let c = (a << 4) | b;
      ValueCastTo::<u16>::cast(&c)
    }
  };
  let imm_b_bits = {
    {
      let a = ValueCastTo::<BigUint>::cast(&instruction_slice_cat_ins_3);
      let b = ValueCastTo::<BigUint>::cast(&false);
      let c = (a << 1) | b;
      ValueCastTo::<u16>::cast(&c)
    }
  };
  let sign_bit_b = {
    {
      let a = ValueCastTo::<u64>::cast(&imm_b_bits);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 12) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let sign_bit_eq_2 =
    { ValueCastTo::<bool>::cast(&sign_bit_b) == ValueCastTo::<bool>::cast(&true) };
  let cat_imm_b = {
    {
      let a = ValueCastTo::<BigUint>::cast(&524287u32);
      let b = ValueCastTo::<BigUint>::cast(&imm_b_bits);
      let c = (a << 13) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let cat_imm_cast_4 = { ValueCastTo::<u32>::cast(&cat_imm_b) };
  let cat_imm_b_1 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&0u32);
      let b = ValueCastTo::<BigUint>::cast(&imm_b_bits);
      let c = (a << 13) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let cat_imm_cast_5 = { ValueCastTo::<u32>::cast(&cat_imm_b_1) };
  let immediate_b = {
    if sign_bit_eq_2 {
      cat_imm_cast_4
    } else {
      cat_imm_cast_5
    }
  };
  let instruction_slice_13 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("11111111111111111111", 2).unwrap();
      let res = (a >> 12) & mask;
      ValueCastTo::<u32>::cast(&res)
    }
  };
  let instruction_slice_shl =
    { ValueCastTo::<u32>::cast(&instruction_slice_13) << ValueCastTo::<u32>::cast(&12u32) };
  let immediate_u = { ValueCastTo::<u32>::cast(&instruction_slice_shl) };
  let instruction_slice_14 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 31) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let instruction_slice_15 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("11111111", 2).unwrap();
      let res = (a >> 12) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let instruction_slice_16 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 20) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let instruction_slice_17 = {
    {
      let a = ValueCastTo::<u64>::cast(&instruction);
      let mask = u64::from_str_radix("1111111111", 2).unwrap();
      let res = (a >> 21) & mask;
      ValueCastTo::<u16>::cast(&res)
    }
  };
  let instruction_slice_cat_ins_4 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&instruction_slice_14);
      let b = ValueCastTo::<BigUint>::cast(&instruction_slice_15);
      let c = (a << 8) | b;
      ValueCastTo::<u16>::cast(&c)
    }
  };
  let instruction_slice_cat_ins_5 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&instruction_slice_cat_ins_4);
      let b = ValueCastTo::<BigUint>::cast(&instruction_slice_16);
      let c = (a << 1) | b;
      ValueCastTo::<u16>::cast(&c)
    }
  };
  let instruction_slice_cat_ins_6 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&instruction_slice_cat_ins_5);
      let b = ValueCastTo::<BigUint>::cast(&instruction_slice_17);
      let c = (a << 10) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let imm_j_bits = {
    {
      let a = ValueCastTo::<BigUint>::cast(&instruction_slice_cat_ins_6);
      let b = ValueCastTo::<BigUint>::cast(&false);
      let c = (a << 1) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let sign_bit_j = {
    {
      let a = ValueCastTo::<u64>::cast(&imm_j_bits);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 20) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let sign_bit_eq_3 =
    { ValueCastTo::<bool>::cast(&sign_bit_j) == ValueCastTo::<bool>::cast(&true) };
  let cat_imm_j = {
    {
      let a = ValueCastTo::<BigUint>::cast(&2047u16);
      let b = ValueCastTo::<BigUint>::cast(&imm_j_bits);
      let c = (a << 21) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let cat_imm_cast_6 = { ValueCastTo::<u32>::cast(&cat_imm_j) };
  let cat_imm_j_1 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&0u16);
      let b = ValueCastTo::<BigUint>::cast(&imm_j_bits);
      let c = (a << 21) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let cat_imm_cast_7 = { ValueCastTo::<u32>::cast(&cat_imm_j_1) };
  let immediate_j = {
    if sign_bit_eq_3 {
      cat_imm_cast_6
    } else {
      cat_imm_cast_7
    }
  };
  let is_r_type = { ValueCastTo::<u8>::cast(&opcode) == ValueCastTo::<u8>::cast(&51u8) };
  let is_i_type = { ValueCastTo::<u8>::cast(&opcode) == ValueCastTo::<u8>::cast(&19u8) };
  let is_l_type = { ValueCastTo::<u8>::cast(&opcode) == ValueCastTo::<u8>::cast(&3u8) };
  let is_s_type = { ValueCastTo::<u8>::cast(&opcode) == ValueCastTo::<u8>::cast(&35u8) };
  let is_b_type = { ValueCastTo::<u8>::cast(&opcode) == ValueCastTo::<u8>::cast(&99u8) };
  let is_j_type = { ValueCastTo::<u8>::cast(&opcode) == ValueCastTo::<u8>::cast(&111u8) };
  let is_jr_type = { ValueCastTo::<u8>::cast(&opcode) == ValueCastTo::<u8>::cast(&103u8) };
  let is_lui_type = { ValueCastTo::<u8>::cast(&opcode) == ValueCastTo::<u8>::cast(&55u8) };
  let is_auipc_type = { ValueCastTo::<u8>::cast(&opcode) == ValueCastTo::<u8>::cast(&23u8) };
  let funct7_eq = { ValueCastTo::<u8>::cast(&funct7) == ValueCastTo::<u8>::cast(&1u8) };
  let is_m_ext = { ValueCastTo::<bool>::cast(&is_r_type) & ValueCastTo::<bool>::cast(&funct7_eq) };
  let func3_eq_mul_op = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&0u8) };
  let is_m_and_func3_eq =
    { ValueCastTo::<bool>::cast(&is_m_ext) & ValueCastTo::<bool>::cast(&func3_eq_mul_op) };
  let mul_op_2 = {
    if is_m_and_func3_eq {
      1u8
    } else {
      0u8
    }
  };
  let func3_eq = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&1u8) };
  let is_m_and_func3_eq_1 =
    { ValueCastTo::<bool>::cast(&is_m_ext) & ValueCastTo::<bool>::cast(&func3_eq) };
  let mul_op_3 = {
    if is_m_and_func3_eq_1 {
      2u8
    } else {
      mul_op_2
    }
  };
  let func3_eq_1 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&2u8) };
  let is_m_and_func3_eq_2 =
    { ValueCastTo::<bool>::cast(&is_m_ext) & ValueCastTo::<bool>::cast(&func3_eq_1) };
  let mul_op_4 = {
    if is_m_and_func3_eq_2 {
      3u8
    } else {
      mul_op_3
    }
  };
  let func3_eq_2 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&3u8) };
  let is_m_and_func3_eq_3 =
    { ValueCastTo::<bool>::cast(&is_m_ext) & ValueCastTo::<bool>::cast(&func3_eq_2) };
  let mul_op_5 = {
    if is_m_and_func3_eq_3 {
      4u8
    } else {
      mul_op_4
    }
  };
  let is_mul_inst_1 = { ValueCastTo::<u8>::cast(&mul_op_5) != ValueCastTo::<u8>::cast(&0u8) };
  let func3_eq_3 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&4u8) };
  let is_m_and_func3_eq_4 =
    { ValueCastTo::<bool>::cast(&is_m_ext) & ValueCastTo::<bool>::cast(&func3_eq_3) };
  let div_op_2 = {
    if is_m_and_func3_eq_4 {
      1u8
    } else {
      0u8
    }
  };
  let func3_eq_4 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&5u8) };
  let is_m_and_func3_eq_5 =
    { ValueCastTo::<bool>::cast(&is_m_ext) & ValueCastTo::<bool>::cast(&func3_eq_4) };
  let div_op_3 = {
    if is_m_and_func3_eq_5 {
      2u8
    } else {
      div_op_2
    }
  };
  let func3_eq_5 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&6u8) };
  let is_m_and_func3_eq_6 =
    { ValueCastTo::<bool>::cast(&is_m_ext) & ValueCastTo::<bool>::cast(&func3_eq_5) };
  let div_op_4 = {
    if is_m_and_func3_eq_6 {
      3u8
    } else {
      div_op_3
    }
  };
  let func3_eq_6 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&7u8) };
  let is_m_and_func3_eq_7 =
    { ValueCastTo::<bool>::cast(&is_m_ext) & ValueCastTo::<bool>::cast(&func3_eq_6) };
  let div_op_5 = {
    if is_m_and_func3_eq_7 {
      4u8
    } else {
      div_op_4
    }
  };
  let is_div_inst_1 = { ValueCastTo::<u8>::cast(&div_op_5) != ValueCastTo::<u8>::cast(&0u8) };
  let funct7_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&funct7);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 5) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let is_r_and_funct7_slice =
    { ValueCastTo::<bool>::cast(&is_r_type) & ValueCastTo::<bool>::cast(&funct7_slice) };
  let is_r_eq =
    { ValueCastTo::<bool>::cast(&is_r_and_funct7_slice) == ValueCastTo::<bool>::cast(&true) };
  let func3_eq_div_op = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&0u8) };
  let is_r_and_func3_eq =
    { ValueCastTo::<bool>::cast(&is_r_eq) & ValueCastTo::<bool>::cast(&func3_eq_div_op) };
  let alu_op_tmp_1 = {
    if is_r_and_func3_eq {
      1u8
    } else {
      0u8
    }
  };
  let funct7_slice_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&funct7);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 5) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let funct7_slice_eq =
    { ValueCastTo::<bool>::cast(&funct7_slice_1) == ValueCastTo::<bool>::cast(&true) };
  let func3_eq_7 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&5u8) };
  let funct7_slice_and_func3_eq =
    { ValueCastTo::<bool>::cast(&funct7_slice_eq) & ValueCastTo::<bool>::cast(&func3_eq_7) };
  let alu_op_tmp_2 = {
    if funct7_slice_and_func3_eq {
      6u8
    } else {
      alu_op_tmp_1
    }
  };
  let funct7_slice_2 = {
    {
      let a = ValueCastTo::<u64>::cast(&funct7);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 5) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let is_r_and_funct7_slice_1 =
    { ValueCastTo::<bool>::cast(&is_r_type) & ValueCastTo::<bool>::cast(&funct7_slice_2) };
  let is_r_eq_1 =
    { ValueCastTo::<bool>::cast(&is_r_and_funct7_slice_1) == ValueCastTo::<bool>::cast(&true) };
  let not_is_r = { !is_r_eq_1 };
  let func3_eq_div_op_1 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&0u8) };
  let not_is_and_func3_eq =
    { ValueCastTo::<bool>::cast(&not_is_r) & ValueCastTo::<bool>::cast(&func3_eq_div_op_1) };
  let alu_op_tmp_3 = {
    if not_is_and_func3_eq {
      0u8
    } else {
      alu_op_tmp_2
    }
  };
  let func3_eq_8 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&7u8) };
  let alu_op_tmp_4 = {
    if func3_eq_8 {
      9u8
    } else {
      alu_op_tmp_3
    }
  };
  let func3_eq_9 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&6u8) };
  let alu_op_tmp_5 = {
    if func3_eq_9 {
      8u8
    } else {
      alu_op_tmp_4
    }
  };
  let func3_eq_10 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&4u8) };
  let alu_op_tmp_6 = {
    if func3_eq_10 {
      4u8
    } else {
      alu_op_tmp_5
    }
  };
  let func3_eq_11 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&2u8) };
  let alu_op_tmp_7 = {
    if func3_eq_11 {
      3u8
    } else {
      alu_op_tmp_6
    }
  };
  let func3_eq_12 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&3u8) };
  let alu_op_tmp_8 = {
    if func3_eq_12 {
      7u8
    } else {
      alu_op_tmp_7
    }
  };
  let func3_eq_13 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&1u8) };
  let alu_op_tmp_9 = {
    if func3_eq_13 {
      2u8
    } else {
      alu_op_tmp_8
    }
  };
  let funct7_slice_3 = {
    {
      let a = ValueCastTo::<u64>::cast(&funct7);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 5) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let funct7_slice_eq_alu_a =
    { ValueCastTo::<bool>::cast(&funct7_slice_3) == ValueCastTo::<bool>::cast(&false) };
  let func3_eq_14 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&5u8) };
  let funct7_slice_and_func3_eq_1 =
    { ValueCastTo::<bool>::cast(&funct7_slice_eq_alu_a) & ValueCastTo::<bool>::cast(&func3_eq_14) };
  let alu_op_tmp_10 = {
    if funct7_slice_and_func3_eq_1 {
      5u8
    } else {
      alu_op_tmp_9
    }
  };
  let is_r_or_is_i =
    { ValueCastTo::<bool>::cast(&is_r_type) | ValueCastTo::<bool>::cast(&is_i_type) };
  let alu_op_2 = {
    if is_r_or_is_i {
      alu_op_tmp_10
    } else {
      0u8
    }
  };
  let is_r_or_is_i_1 =
    { ValueCastTo::<bool>::cast(&is_r_type) | ValueCastTo::<bool>::cast(&is_i_type) };
  let reg_write_3 = {
    if is_r_or_is_i_1 {
      true
    } else {
      false
    }
  };
  let alu_src_2 = {
    if is_r_type {
      0u8
    } else {
      0u8
    }
  };
  let alu_src_3 = {
    if is_i_type {
      1u8
    } else {
      alu_src_2
    }
  };
  let immediate_2 = {
    if is_i_type {
      immediate_i
    } else {
      0u32
    }
  };
  let mem_read_3 = {
    if is_l_type {
      true
    } else {
      false
    }
  };
  let reg_write_4 = {
    if is_l_type {
      true
    } else {
      reg_write_3
    }
  };
  let mem_to_reg_3 = {
    if is_l_type {
      true
    } else {
      false
    }
  };
  let alu_src_4 = {
    if is_l_type {
      1u8
    } else {
      alu_src_3
    }
  };
  let immediate_3 = {
    if is_l_type {
      immediate_i
    } else {
      immediate_2
    }
  };
  let func3_eq_div_op_2 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&0u8) };
  let is_l_and_func3_eq =
    { ValueCastTo::<bool>::cast(&is_l_type) & ValueCastTo::<bool>::cast(&func3_eq_div_op_2) };
  let load_type_bits_1 = {
    if is_l_and_func3_eq {
      0u8
    } else {
      2u8
    }
  };
  let func3_eq_15 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&1u8) };
  let is_l_and_func3_eq_1 =
    { ValueCastTo::<bool>::cast(&is_l_type) & ValueCastTo::<bool>::cast(&func3_eq_15) };
  let load_type_bits_2 = {
    if is_l_and_func3_eq_1 {
      1u8
    } else {
      load_type_bits_1
    }
  };
  let func3_eq_load_type = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&2u8) };
  let is_l_and_func3_eq_2 =
    { ValueCastTo::<bool>::cast(&is_l_type) & ValueCastTo::<bool>::cast(&func3_eq_load_type) };
  let load_type_bits_3 = {
    if is_l_and_func3_eq_2 {
      2u8
    } else {
      load_type_bits_2
    }
  };
  let func3_eq_16 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&4u8) };
  let is_l_and_func3_eq_3 =
    { ValueCastTo::<bool>::cast(&is_l_type) & ValueCastTo::<bool>::cast(&func3_eq_16) };
  let load_type_bits_4 = {
    if is_l_and_func3_eq_3 {
      4u8
    } else {
      load_type_bits_3
    }
  };
  let func3_eq_17 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&5u8) };
  let is_l_and_func3_eq_4 =
    { ValueCastTo::<bool>::cast(&is_l_type) & ValueCastTo::<bool>::cast(&func3_eq_17) };
  let load_type_bits_5 = {
    if is_l_and_func3_eq_4 {
      5u8
    } else {
      load_type_bits_4
    }
  };
  let mem_write_3 = {
    if is_s_type {
      true
    } else {
      false
    }
  };
  let alu_src_5 = {
    if is_s_type {
      1u8
    } else {
      alu_src_4
    }
  };
  let immediate_4 = {
    if is_s_type {
      immediate_s
    } else {
      immediate_3
    }
  };
  let func3_eq_load_type_1 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&2u8) };
  let is_s_and_func3_eq =
    { ValueCastTo::<bool>::cast(&is_s_type) & ValueCastTo::<bool>::cast(&func3_eq_load_type_1) };
  let store_type_bits_2 = {
    if is_s_and_func3_eq {
      2u8
    } else {
      0u8
    }
  };
  let func3_eq_div_op_3 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&0u8) };
  let is_s_and_func3_eq_1 =
    { ValueCastTo::<bool>::cast(&is_s_type) & ValueCastTo::<bool>::cast(&func3_eq_div_op_3) };
  let store_type_bits_3 = {
    if is_s_and_func3_eq_1 {
      0u8
    } else {
      store_type_bits_2
    }
  };
  let func3_eq_18 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&1u8) };
  let is_s_and_func3_eq_2 =
    { ValueCastTo::<bool>::cast(&is_s_type) & ValueCastTo::<bool>::cast(&func3_eq_18) };
  let store_type_bits_4 = {
    if is_s_and_func3_eq_2 {
      1u8
    } else {
      store_type_bits_3
    }
  };
  let func3_eq_branch_op = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&0u8) };
  let branch_op_tmp_1 = {
    if func3_eq_branch_op {
      1u8
    } else {
      0u8
    }
  };
  let func3_eq_19 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&1u8) };
  let branch_op_tmp_2 = {
    if func3_eq_19 {
      2u8
    } else {
      branch_op_tmp_1
    }
  };
  let func3_eq_20 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&4u8) };
  let branch_op_tmp_3 = {
    if func3_eq_20 {
      3u8
    } else {
      branch_op_tmp_2
    }
  };
  let func3_eq_21 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&5u8) };
  let branch_op_tmp_4 = {
    if func3_eq_21 {
      4u8
    } else {
      branch_op_tmp_3
    }
  };
  let func3_eq_22 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&6u8) };
  let branch_op_tmp_5 = {
    if func3_eq_22 {
      5u8
    } else {
      branch_op_tmp_4
    }
  };
  let func3_eq_23 = { ValueCastTo::<u8>::cast(&func3) == ValueCastTo::<u8>::cast(&7u8) };
  let branch_op_tmp_6 = {
    if func3_eq_23 {
      6u8
    } else {
      branch_op_tmp_5
    }
  };
  let immediate_5 = {
    if is_b_type {
      immediate_b
    } else {
      immediate_4
    }
  };
  let branch_op_2 = {
    if is_b_type {
      branch_op_tmp_6
    } else {
      0u8
    }
  };
  let is_lui_or_is_auipc =
    { ValueCastTo::<bool>::cast(&is_lui_type) | ValueCastTo::<bool>::cast(&is_auipc_type) };
  let reg_write_5 = {
    if is_lui_or_is_auipc {
      true
    } else {
      reg_write_4
    }
  };
  let alu_src_6 = {
    if is_lui_type {
      1u8
    } else {
      alu_src_5
    }
  };
  let alu_a_zero_2 = {
    if is_lui_type {
      true
    } else {
      false
    }
  };
  let is_lui_or_is_auipc_1 =
    { ValueCastTo::<bool>::cast(&is_lui_type) | ValueCastTo::<bool>::cast(&is_auipc_type) };
  let immediate_6 = {
    if is_lui_or_is_auipc_1 {
      immediate_u
    } else {
      immediate_5
    }
  };
  let alu_src_7 = {
    if is_auipc_type {
      2u8
    } else {
      alu_src_6
    }
  };
  let reg_write_6 = {
    if is_j_type {
      true
    } else {
      reg_write_5
    }
  };
  let alu_src_8 = {
    if is_j_type {
      1u8
    } else {
      alu_src_7
    }
  };
  let immediate_7 = {
    if is_j_type {
      immediate_j
    } else {
      immediate_6
    }
  };
  let jump_op_2 = {
    if is_j_type {
      true
    } else {
      false
    }
  };
  let reg_write_7 = {
    if is_jr_type {
      true
    } else {
      reg_write_6
    }
  };
  let alu_src_9 = {
    if is_jr_type {
      1u8
    } else {
      alu_src_8
    }
  };
  let immediate_8 = {
    if is_jr_type {
      immediate_i
    } else {
      immediate_7
    }
  };
  let jumpr_op_2 = {
    if is_jr_type {
      true
    } else {
      false
    }
  };
  let reg_write_8 = {
    if is_mul_inst_1 {
      true
    } else {
      reg_write_7
    }
  };
  let alu_src_10 = {
    if is_mul_inst_1 {
      0u8
    } else {
      alu_src_9
    }
  };
  let reg_write_9 = {
    if is_div_inst_1 {
      true
    } else {
      reg_write_8
    }
  };
  let alu_src_11 = {
    if is_div_inst_1 {
      0u8
    } else {
      alu_src_10
    }
  };
  let rd_eq_alu_op = { ValueCastTo::<u8>::cast(&rd) == ValueCastTo::<u8>::cast(&0u8) };
  let reg_write_10 = {
    if rd_eq_alu_op {
      false
    } else {
      reg_write_9
    }
  };
  let immediate_8_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&immediate_8);
      let mask = u64::from_str_radix("111111111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u16>::cast(&res)
    }
  };
  let div_op_cat_mul_op = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_5);
      let b = ValueCastTo::<BigUint>::cast(&mul_op_5);
      let c = (a << 3) | b;
      ValueCastTo::<u8>::cast(&c)
    }
  };
  let div_op_cat_immediate_8 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_mul_op);
      let b = ValueCastTo::<BigUint>::cast(&immediate_8_slice);
      let c = (a << 12) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let div_op_cat_rd = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_immediate_8);
      let b = ValueCastTo::<BigUint>::cast(&rd);
      let c = (a << 5) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let div_op_cat_alu_a = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_rd);
      let b = ValueCastTo::<BigUint>::cast(&alu_a_zero_2);
      let c = (a << 1) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let div_op_cat_store_type = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_alu_a);
      let b = ValueCastTo::<BigUint>::cast(&store_type_bits_4);
      let c = (a << 2) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let div_op_cat_jumpr_op = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_store_type);
      let b = ValueCastTo::<BigUint>::cast(&jumpr_op_2);
      let c = (a << 1) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let div_op_cat_jump_op = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_jumpr_op);
      let b = ValueCastTo::<BigUint>::cast(&jump_op_2);
      let c = (a << 1) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let div_op_cat_branch_op = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_jump_op);
      let b = ValueCastTo::<BigUint>::cast(&branch_op_2);
      let c = (a << 3) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let div_op_cat_load_type = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_branch_op);
      let b = ValueCastTo::<BigUint>::cast(&load_type_bits_5);
      let c = (a << 3) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let div_op_cat_branch_op_1 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_load_type);
      let b = ValueCastTo::<BigUint>::cast(&0u8);
      let c = (a << 3) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let div_op_cat_alu_src = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_branch_op_1);
      let b = ValueCastTo::<BigUint>::cast(&alu_src_11);
      let c = (a << 2) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let div_op_cat_mem_to = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_alu_src);
      let b = ValueCastTo::<BigUint>::cast(&mem_to_reg_3);
      let c = (a << 1) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let div_op_cat_reg_write = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_mem_to);
      let b = ValueCastTo::<BigUint>::cast(&reg_write_10);
      let c = (a << 1) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let div_op_cat_mem_write = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_reg_write);
      let b = ValueCastTo::<BigUint>::cast(&mem_write_3);
      let c = (a << 1) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let div_op_cat_mem_read = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_mem_write);
      let b = ValueCastTo::<BigUint>::cast(&mem_read_3);
      let c = (a << 1) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let control_signals = {
    {
      let a = ValueCastTo::<BigUint>::cast(&div_op_cat_mem_read);
      let b = ValueCastTo::<BigUint>::cast(&alu_op_2);
      let c = (a << 5) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let is_i_or_is_r =
    { ValueCastTo::<bool>::cast(&is_i_type) | ValueCastTo::<bool>::cast(&is_r_type) };
  let is_i_or_is_s =
    { ValueCastTo::<bool>::cast(&is_i_or_is_r) | ValueCastTo::<bool>::cast(&is_s_type) };
  let is_i_or_is_b =
    { ValueCastTo::<bool>::cast(&is_i_or_is_s) | ValueCastTo::<bool>::cast(&is_b_type) };
  let is_i_or_is_l =
    { ValueCastTo::<bool>::cast(&is_i_or_is_b) | ValueCastTo::<bool>::cast(&is_l_type) };
  let is_i_or_is_jr =
    { ValueCastTo::<bool>::cast(&is_i_or_is_l) | ValueCastTo::<bool>::cast(&is_jr_type) };
  let is_i_or_is_mul =
    { ValueCastTo::<bool>::cast(&is_i_or_is_jr) | ValueCastTo::<bool>::cast(&is_mul_inst_1) };
  let need_rs1 =
    { ValueCastTo::<bool>::cast(&is_i_or_is_mul) | ValueCastTo::<bool>::cast(&is_div_inst_1) };
  let is_r_or_is_s =
    { ValueCastTo::<bool>::cast(&is_r_type) | ValueCastTo::<bool>::cast(&is_s_type) };
  let is_r_or_is_b =
    { ValueCastTo::<bool>::cast(&is_r_or_is_s) | ValueCastTo::<bool>::cast(&is_b_type) };
  let is_r_or_is_mul =
    { ValueCastTo::<bool>::cast(&is_r_or_is_b) | ValueCastTo::<bool>::cast(&is_mul_inst_1) };
  let need_rs2 =
    { ValueCastTo::<bool>::cast(&is_r_or_is_mul) | ValueCastTo::<bool>::cast(&is_div_inst_1) };
  let id_ex_rd_7 = { sim.id_ex_valid.payload[false as usize].clone() };
  if id_ex_rd_7 {
    let if_id_rd_3 = { sim.if_id_valid.payload[false as usize].clone() };
    let if_id_mux = {
      if if_id_rd_3 {
        if_id_pc_in
      } else {
        0u32
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:411
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, if_id_mux.clone(), "DecodeStageInstance");
      sim.id_ex_pc.write(0, write);
    };
    let if_id_mux_1 = {
      if if_id_rd_3 {
        need_rs1
      } else {
        false
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:412
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, if_id_mux_1.clone(), "DecodeStageInstance");
      sim.id_ex_need_rs1.write(0, write);
    };
    let if_id_mux_2 = {
      if if_id_rd_3 {
        need_rs2
      } else {
        false
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:413
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, if_id_mux_2.clone(), "DecodeStageInstance");
      sim.id_ex_need_rs2.write(0, write);
    };
    let if_id_mux_3 = {
      if if_id_rd_3 {
        prediction_info_in_1
      } else {
        0u64
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:415
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, if_id_mux_3.clone(), "DecodeStageInstance");
      sim.id_ex_prediction_info.write(0, write);
    };
  }
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:431
  ();
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:431
  {
    let stamp = sim.stamp - sim.stamp % 100 + 100;
    sim.ExecuteStageInstance_event.push_back(stamp)
  };
  let if_id_rd_4 = { sim.if_id_valid.payload[false as usize].clone() };
  let control_signals_cast = { ValueCastTo::<u64>::cast(&control_signals) };
  let if_id_mux_4 = {
    if if_id_rd_4 {
      control_signals_cast
    } else {
      0u64
    }
  };
  let id_ex_rd_8 = { sim.id_ex_control.payload[false as usize].clone() };
  let out_control_1 = {
    if id_ex_rd_7 {
      if_id_mux_4
    } else {
      id_ex_rd_8
    }
  };
  let out_mul_op = {
    {
      let a = ValueCastTo::<u64>::cast(&out_control_1);
      let mask = u64::from_str_radix("111", 2).unwrap();
      let res = (a >> 42) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let if_id_mux_5 = {
    if if_id_rd_4 {
      prediction_info_in_1
    } else {
      0u64
    }
  };
  let id_ex_rd_9 = { sim.id_ex_prediction_info.payload[false as usize].clone() };
  let id_ex_mux_1 = {
    if id_ex_rd_7 {
      if_id_mux_5
    } else {
      id_ex_rd_9
    }
  };
  let need_rs2_cast = { ValueCastTo::<bool>::cast(&need_rs2) };
  let if_id_mux_6 = {
    if if_id_rd_4 {
      need_rs2_cast
    } else {
      false
    }
  };
  let id_ex_rd_10 = { sim.id_ex_need_rs2.payload[false as usize].clone() };
  let id_ex_cast = { ValueCastTo::<bool>::cast(&id_ex_rd_10) };
  let id_ex_mux_2 = {
    if id_ex_rd_7 {
      if_id_mux_6
    } else {
      id_ex_cast
    }
  };
  let need_rs1_cast = { ValueCastTo::<bool>::cast(&need_rs1) };
  let if_id_mux_7 = {
    if if_id_rd_4 {
      need_rs1_cast
    } else {
      false
    }
  };
  let id_ex_rd_11 = { sim.id_ex_need_rs1.payload[false as usize].clone() };
  let id_ex_cast_1 = { ValueCastTo::<bool>::cast(&id_ex_rd_11) };
  let id_ex_mux_3 = {
    if id_ex_rd_7 {
      if_id_mux_7
    } else {
      id_ex_cast_1
    }
  };
  let if_id_mux_8 = {
    if if_id_rd_4 {
      immediate_8
    } else {
      0u32
    }
  };
  let id_ex_rd_12 = { sim.id_ex_immediate.payload[false as usize].clone() };
  let id_ex_mux_4 = {
    if id_ex_rd_7 {
      if_id_mux_8
    } else {
      id_ex_rd_12
    }
  };
  let rs2_cast = { ValueCastTo::<u8>::cast(&rs2) };
  let if_id_mux_9 = {
    if if_id_rd_4 {
      rs2_cast
    } else {
      0u8
    }
  };
  let id_ex_rd_13 = { sim.id_ex_rs2_idx.payload[false as usize].clone() };
  let id_ex_mux_5 = {
    if id_ex_rd_7 {
      if_id_mux_9
    } else {
      id_ex_rd_13
    }
  };
  let rs1_cast = { ValueCastTo::<u8>::cast(&rs1) };
  let if_id_mux_10 = {
    if if_id_rd_4 {
      rs1_cast
    } else {
      0u8
    }
  };
  let id_ex_rd_14 = { sim.id_ex_rs1_idx.payload[false as usize].clone() };
  let id_ex_mux_6 = {
    if id_ex_rd_7 {
      if_id_mux_10
    } else {
      id_ex_rd_14
    }
  };
  let id_ex_cat_id_ex = {
    {
      let a = ValueCastTo::<BigUint>::cast(&id_ex_mux_1);
      let b = ValueCastTo::<BigUint>::cast(&id_ex_mux_2);
      let c = (a << 1) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let id_ex_cat_id_ex_1 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&id_ex_cat_id_ex);
      let b = ValueCastTo::<BigUint>::cast(&id_ex_mux_3);
      let c = (a << 1) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let id_ex_cat_id_ex_2 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&id_ex_cat_id_ex_1);
      let b = ValueCastTo::<BigUint>::cast(&id_ex_mux_4);
      let c = (a << 32) | b;
      ValueCastTo::<BigUint>::cast(&c)
    }
  };
  let id_ex_cat_id_ex_3 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&id_ex_cat_id_ex_2.clone());
      let b = ValueCastTo::<BigUint>::cast(&id_ex_mux_5);
      let c = (a << 5) | b;
      ValueCastTo::<BigUint>::cast(&c)
    }
  };
  let id_ex_cat_id_ex_4 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&id_ex_cat_id_ex_3.clone());
      let b = ValueCastTo::<BigUint>::cast(&id_ex_mux_6);
      let c = (a << 5) | b;
      ValueCastTo::<BigUint>::cast(&c)
    }
  };
  let decode_signals = {
    {
      let a = ValueCastTo::<BigUint>::cast(&id_ex_cat_id_ex_4.clone());
      let b = ValueCastTo::<BigUint>::cast(&out_control_1);
      let c = (a << 48) | b;
      ValueCastTo::<BigUint>::cast(&c)
    }
  };
  sim.decode_signals_value = Some(decode_signals.clone());

  true
}
