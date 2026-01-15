use crate::simulator::Simulator;
use sim_runtime::num_bigint::{BigInt, BigUint};
use sim_runtime::*;
use std::ffi::c_void;

// Elaborating module HazardUnit
pub fn HazardUnit(sim: &mut Simulator) -> bool {
  let execute_signals_valid = { sim.execute_signals_value.is_some() };
  let execute_signals_1 = {
    if execute_signals_valid {
      {
        if let Some(x) = &sim.execute_signals_value {
          x
        } else {
          panic!("Value execute_signals invalid!");
        }
      }
      .clone()
    } else {
      ValueCastTo::<BigUint>::cast(&(0 as u64))
    }
  };
  let decode_signals_valid = { sim.decode_signals_value.is_some() };
  let decode_signals_1 = {
    if decode_signals_valid {
      {
        if let Some(x) = &sim.decode_signals_value {
          x
        } else {
          panic!("Value decode_signals invalid!");
        }
      }
      .clone()
    } else {
      ValueCastTo::<BigUint>::cast(&(0 as u64))
    }
  };
  let fetch_signals_valid = { sim.fetch_signals_value.is_some() };
  let fetch_signals_1 = {
    if fetch_signals_valid {
      {
        if let Some(x) = &sim.fetch_signals_value {
          x
        } else {
          panic!("Value fetch_signals invalid!");
        }
      }
      .clone()
    } else {
      0u32
    }
  };
  let memory_signals_valid = { sim.memory_signals_value.is_some() };
  let memory_signals_1 = {
    if memory_signals_valid {
      {
        if let Some(x) = &sim.memory_signals_value {
          x
        } else {
          panic!("Value memory_signals invalid!");
        }
      }
      .clone()
    } else {
      0u64
    }
  };
  let writeback_signals_valid = { sim.writeback_signals_value.is_some() };
  let writeback_signals_1 = {
    if writeback_signals_valid {
      {
        if let Some(x) = &sim.writeback_signals_value {
          x
        } else {
          panic!("Value writeback_signals invalid!");
        }
      }
      .clone()
    } else {
      0u64
    }
  };
  let execute_signals_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&execute_signals_1.clone());
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let pc_change_2 = { ValueCastTo::<bool>::cast(&execute_signals_slice) };
  let execute_signals_slice_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&execute_signals_1.clone());
      let mask = u64::from_str_radix("11111111111111111111111111111111", 2).unwrap();
      let res = (a >> 1) & mask;
      ValueCastTo::<u32>::cast(&res)
    }
  };
  let target_pc_3 = { ValueCastTo::<u32>::cast(&execute_signals_slice_1) };
  let execute_signals_slice_2 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&execute_signals_1.clone());
      let mask = BigUint::parse_bytes("1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111".as_bytes(), 2).unwrap();
      let res = (a >> 81) & mask;
      ValueCastTo::<BigUint>::cast(&res)
    }
  };
  let prediction_result_1 = { ValueCastTo::<BigUint>::cast(&execute_signals_slice_2.clone()) };
  let execute_signals_slice_3 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&execute_signals_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 184) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let mul_busy_sig = { ValueCastTo::<bool>::cast(&execute_signals_slice_3) };
  let execute_signals_slice_4 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&execute_signals_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 185) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let mul_done_sig = { ValueCastTo::<bool>::cast(&execute_signals_slice_4) };
  let execute_signals_slice_5 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&execute_signals_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 186) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let mul_stall_sig = { ValueCastTo::<bool>::cast(&execute_signals_slice_5) };
  let execute_signals_slice_6 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&execute_signals_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 187) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let div_busy_sig = { ValueCastTo::<bool>::cast(&execute_signals_slice_6) };
  let execute_signals_slice_7 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&execute_signals_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 188) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let div_done_sig = { ValueCastTo::<bool>::cast(&execute_signals_slice_7) };
  let execute_signals_slice_8 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&execute_signals_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 189) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let div_stall_sig = { ValueCastTo::<bool>::cast(&execute_signals_slice_8) };
  let prediction_result_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&prediction_result_1.clone());
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let mispredict_1 = { ValueCastTo::<bool>::cast(&prediction_result_slice) };
  let prediction_result_slice_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&prediction_result_1.clone());
      let mask = u64::from_str_radix("11111111111111111111111111111111", 2).unwrap();
      let res = (a >> 1) & mask;
      ValueCastTo::<u32>::cast(&res)
    }
  };
  let correct_pc_1 = { ValueCastTo::<u32>::cast(&prediction_result_slice_1) };
  let prediction_result_slice_2 = {
    {
      let a = ValueCastTo::<u64>::cast(&prediction_result_1.clone());
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 33) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let actual_taken_1 = { ValueCastTo::<bool>::cast(&prediction_result_slice_2) };
  let prediction_result_slice_3 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&prediction_result_1.clone());
      let mask = BigUint::parse_bytes("11111111111111111111111111111111".as_bytes(), 2).unwrap();
      let res = (a >> 34) & mask;
      ValueCastTo::<u32>::cast(&res)
    }
  };
  let actual_target_pc_1 = { ValueCastTo::<u32>::cast(&prediction_result_slice_3) };
  let prediction_result_slice_4 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&prediction_result_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 66) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let pred_btb_hit = { ValueCastTo::<bool>::cast(&prediction_result_slice_4) };
  let prediction_result_slice_5 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&prediction_result_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 67) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let pred_predict_taken = { ValueCastTo::<bool>::cast(&prediction_result_slice_5) };
  let prediction_result_slice_6 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&prediction_result_1.clone());
      let mask = BigUint::parse_bytes("11111111111111111111111111111111".as_bytes(), 2).unwrap();
      let res = (a >> 68) & mask;
      ValueCastTo::<u32>::cast(&res)
    }
  };
  let branch_pc = { ValueCastTo::<u32>::cast(&prediction_result_slice_6) };
  let prediction_result_slice_7 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&prediction_result_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 100) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let is_branch_ex = { ValueCastTo::<bool>::cast(&prediction_result_slice_7) };
  let prediction_result_slice_8 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&prediction_result_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 101) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let is_jump_ex = { ValueCastTo::<bool>::cast(&prediction_result_slice_8) };
  let prediction_result_slice_9 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&prediction_result_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 102) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let is_jumpr_ex = { ValueCastTo::<bool>::cast(&prediction_result_slice_9) };
  let instruction_3 = { ValueCastTo::<u32>::cast(&fetch_signals_1) };
  let decode_signals_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&decode_signals_1.clone());
      let mask =
        u64::from_str_radix("111111111111111111111111111111111111111111111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u64>::cast(&res)
    }
  };
  let control_in_3 = { ValueCastTo::<u64>::cast(&decode_signals_slice) };
  let decode_signals_slice_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&decode_signals_1.clone());
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 48) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let rs1_1 = { ValueCastTo::<u8>::cast(&decode_signals_slice_1) };
  let decode_signals_slice_2 = {
    {
      let a = ValueCastTo::<u64>::cast(&decode_signals_1.clone());
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 53) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let rs2_1 = { ValueCastTo::<u8>::cast(&decode_signals_slice_2) };
  let decode_signals_slice_3 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&decode_signals_1.clone());
      let mask = BigUint::parse_bytes("11111111111111111111111111111111".as_bytes(), 2).unwrap();
      let res = (a >> 58) & mask;
      ValueCastTo::<u32>::cast(&res)
    }
  };
  let immediate_9 = { ValueCastTo::<u32>::cast(&decode_signals_slice_3) };
  let decode_signals_slice_4 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&decode_signals_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 90) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let needs_rs1 = { ValueCastTo::<bool>::cast(&decode_signals_slice_4) };
  let decode_signals_slice_5 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&decode_signals_1.clone());
      let mask = BigUint::parse_bytes("1".as_bytes(), 2).unwrap();
      let res = (a >> 91) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let needs_rs2 = { ValueCastTo::<bool>::cast(&decode_signals_slice_5) };
  let decode_signals_slice_6 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&decode_signals_1.clone());
      let mask = BigUint::parse_bytes("1111111111111111111111111111111111".as_bytes(), 2).unwrap();
      let res = (a >> 92) & mask;
      ValueCastTo::<u64>::cast(&res)
    }
  };
  let prediction_info_id = { ValueCastTo::<u64>::cast(&decode_signals_slice_6) };
  let execute_signals_slice_9 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&execute_signals_1.clone());
      let mask =
        BigUint::parse_bytes("111111111111111111111111111111111111111111111111".as_bytes(), 2)
          .unwrap();
      let res = (a >> 33) & mask;
      ValueCastTo::<u64>::cast(&res)
    }
  };
  let memory_control = { ValueCastTo::<u64>::cast(&execute_signals_slice_9) };
  let rd_mem = {
    {
      let a = ValueCastTo::<u64>::cast(&memory_control);
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 25) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let reg_write_mem = {
    {
      let a = ValueCastTo::<u64>::cast(&memory_control);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 7) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let mem_read_mem = {
    {
      let a = ValueCastTo::<u64>::cast(&memory_control);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 5) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let id_ex_rd_15 = { sim.id_ex_valid.payload[false as usize].clone() };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1512
  print!("@line:{:<5} {:<10}: [HazardUnit]\t", line!(), cyclize(sim.stamp));
  println!("HAZARD memory_control: id_ex_valid={}, memory_control={:012x}, rd_mem={}, reg_write_mem={}, mem_read_mem={}", if id_ex_rd_15 { 1 } else { 0 }, memory_control, rd_mem, if reg_write_mem { 1 } else { 0 }, if mem_read_mem { 1 } else { 0 }, );
  let memory_signals_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&memory_signals_1);
      let mask =
        u64::from_str_radix("111111111111111111111111111111111111111111111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u64>::cast(&res)
    }
  };
  let mem_sig_control = { ValueCastTo::<u64>::cast(&memory_signals_slice) };
  let memory_signals_slice_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&memory_signals_1);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 48) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let sb_sh_stall = { ValueCastTo::<bool>::cast(&memory_signals_slice_1) };
  let ex_mem_rd_14 = { sim.ex_mem_valid.payload[false as usize].clone() };
  let wb_control_1 = {
    if ex_mem_rd_14 {
      mem_sig_control
    } else {
      0u64
    }
  };
  let rd_wb = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_control_1);
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 25) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let reg_write_wb = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_control_1);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 7) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let mem_read_wb = {
    {
      let a = ValueCastTo::<u64>::cast(&wb_control_1);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 5) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let mem_read_and_reg_write =
    { ValueCastTo::<bool>::cast(&mem_read_mem) & ValueCastTo::<bool>::cast(&reg_write_mem) };
  let rd_mem_neq_alu_op = { ValueCastTo::<u8>::cast(&rd_mem) != ValueCastTo::<u8>::cast(&0u8) };
  let mem_read_and_rd_mem = {
    ValueCastTo::<bool>::cast(&mem_read_and_reg_write)
      & ValueCastTo::<bool>::cast(&rd_mem_neq_alu_op)
  };
  let rs1_1_eq_rd_mem = { ValueCastTo::<u8>::cast(&rs1_1) == ValueCastTo::<u8>::cast(&rd_mem) };
  let needs_rs1_and_rs1_1 =
    { ValueCastTo::<bool>::cast(&needs_rs1) & ValueCastTo::<bool>::cast(&rs1_1_eq_rd_mem) };
  let rs2_1_eq_rd_mem = { ValueCastTo::<u8>::cast(&rs2_1) == ValueCastTo::<u8>::cast(&rd_mem) };
  let needs_rs2_and_rs2_1 =
    { ValueCastTo::<bool>::cast(&needs_rs2) & ValueCastTo::<bool>::cast(&rs2_1_eq_rd_mem) };
  let needs_rs1_or_needs_rs2 = {
    ValueCastTo::<bool>::cast(&needs_rs1_and_rs1_1)
      | ValueCastTo::<bool>::cast(&needs_rs2_and_rs2_1)
  };
  let load_use_hazard_mem = {
    ValueCastTo::<bool>::cast(&mem_read_and_rd_mem)
      & ValueCastTo::<bool>::cast(&needs_rs1_or_needs_rs2)
  };
  let rs1_1_eq_rd_mem_1 = { ValueCastTo::<u8>::cast(&rs1_1) == ValueCastTo::<u8>::cast(&rd_mem) };
  let rs2_1_eq_rd_mem_1 = { ValueCastTo::<u8>::cast(&rs2_1) == ValueCastTo::<u8>::cast(&rd_mem) };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1528
  print!("@line:{:<5} {:<10}: [HazardUnit]\t", line!(), cyclize(sim.stamp));
  println!("LOAD_USE_HAZARD_MEM: mem_read_mem={}, reg_write_mem={}, rd_mem={}, rs1={}, rs2={}, needs_rs1={}, needs_rs2={}, rs1==rd_mem={}, rs2==rd_mem={}, result={}", if mem_read_mem { 1 } else { 0 }, if reg_write_mem { 1 } else { 0 }, rd_mem, rs1_1, rs2_1, if needs_rs1 { 1 } else { 0 }, if needs_rs2 { 1 } else { 0 }, if rs1_1_eq_rd_mem_1 { 1 } else { 0 }, if rs2_1_eq_rd_mem_1 { 1 } else { 0 }, if load_use_hazard_mem { 1 } else { 0 }, );
  let mem_read_and_reg_write_1 =
    { ValueCastTo::<bool>::cast(&mem_read_wb) & ValueCastTo::<bool>::cast(&reg_write_wb) };
  let rd_wb_neq_alu_op = { ValueCastTo::<u8>::cast(&rd_wb) != ValueCastTo::<u8>::cast(&0u8) };
  let mem_read_and_rd_wb = {
    ValueCastTo::<bool>::cast(&mem_read_and_reg_write_1)
      & ValueCastTo::<bool>::cast(&rd_wb_neq_alu_op)
  };
  let rs1_1_eq_rd_wb = { ValueCastTo::<u8>::cast(&rs1_1) == ValueCastTo::<u8>::cast(&rd_wb) };
  let needs_rs1_and_rs1_1_1 =
    { ValueCastTo::<bool>::cast(&needs_rs1) & ValueCastTo::<bool>::cast(&rs1_1_eq_rd_wb) };
  let rs2_1_eq_rd_wb = { ValueCastTo::<u8>::cast(&rs2_1) == ValueCastTo::<u8>::cast(&rd_wb) };
  let needs_rs2_and_rs2_1_1 =
    { ValueCastTo::<bool>::cast(&needs_rs2) & ValueCastTo::<bool>::cast(&rs2_1_eq_rd_wb) };
  let needs_rs1_or_needs_rs2_1 = {
    ValueCastTo::<bool>::cast(&needs_rs1_and_rs1_1_1)
      | ValueCastTo::<bool>::cast(&needs_rs2_and_rs2_1_1)
  };
  let load_use_hazard_wb = {
    ValueCastTo::<bool>::cast(&mem_read_and_rd_wb)
      & ValueCastTo::<bool>::cast(&needs_rs1_or_needs_rs2_1)
  };
  let ex_control = { sim.id_ex_control.payload[false as usize].clone() };
  let ex_rd = {
    {
      let a = ValueCastTo::<u64>::cast(&ex_control);
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 25) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let ex_mul_op = {
    {
      let a = ValueCastTo::<u64>::cast(&ex_control);
      let mask = u64::from_str_radix("111", 2).unwrap();
      let res = (a >> 42) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let ex_div_op = {
    {
      let a = ValueCastTo::<u64>::cast(&ex_control);
      let mask = u64::from_str_radix("111", 2).unwrap();
      let res = (a >> 45) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let is_ex_mul = { ValueCastTo::<u8>::cast(&ex_mul_op) != ValueCastTo::<u8>::cast(&0u8) };
  let is_ex_div = { ValueCastTo::<u8>::cast(&ex_div_op) != ValueCastTo::<u8>::cast(&0u8) };
  let mul_cycle_1 = { sim.mul_cycle_counter.payload[false as usize].clone() };
  let mul_cycle_neq_store_type =
    { ValueCastTo::<u8>::cast(&mul_cycle_1) != ValueCastTo::<u8>::cast(&0u8) };
  let mul_executing_1 = {
    if mul_cycle_neq_store_type {
      true
    } else {
      false
    }
  };
  let ex_rd_neq_alu_op = { ValueCastTo::<u8>::cast(&ex_rd) != ValueCastTo::<u8>::cast(&0u8) };
  let is_ex_and_ex_rd =
    { ValueCastTo::<bool>::cast(&is_ex_mul) & ValueCastTo::<bool>::cast(&ex_rd_neq_alu_op) };
  let rs1_1_eq_ex_rd = { ValueCastTo::<u8>::cast(&rs1_1) == ValueCastTo::<u8>::cast(&ex_rd) };
  let needs_rs1_and_rs1_1_2 =
    { ValueCastTo::<bool>::cast(&needs_rs1) & ValueCastTo::<bool>::cast(&rs1_1_eq_ex_rd) };
  let rs2_1_eq_ex_rd = { ValueCastTo::<u8>::cast(&rs2_1) == ValueCastTo::<u8>::cast(&ex_rd) };
  let needs_rs2_and_rs2_1_2 =
    { ValueCastTo::<bool>::cast(&needs_rs2) & ValueCastTo::<bool>::cast(&rs2_1_eq_ex_rd) };
  let needs_rs1_or_needs_rs2_2 = {
    ValueCastTo::<bool>::cast(&needs_rs1_and_rs1_1_2)
      | ValueCastTo::<bool>::cast(&needs_rs2_and_rs2_1_2)
  };
  let mul_result_hazard = {
    ValueCastTo::<bool>::cast(&is_ex_and_ex_rd)
      & ValueCastTo::<bool>::cast(&needs_rs1_or_needs_rs2_2)
  };
  let div_cycle = { sim.div_iter_count.payload[false as usize].clone() };
  let div_state_val_1 = { sim.div_state.payload[false as usize].clone() };
  let div_state_neq_3 =
    { ValueCastTo::<u8>::cast(&div_state_val_1) != ValueCastTo::<u8>::cast(&0u8) };
  let div_executing_1 = {
    if div_state_neq_3 {
      true
    } else {
      false
    }
  };
  let ex_rd_neq_alu_op_1 = { ValueCastTo::<u8>::cast(&ex_rd) != ValueCastTo::<u8>::cast(&0u8) };
  let is_ex_and_ex_rd_1 =
    { ValueCastTo::<bool>::cast(&is_ex_div) & ValueCastTo::<bool>::cast(&ex_rd_neq_alu_op_1) };
  let rs1_1_eq_ex_rd_1 = { ValueCastTo::<u8>::cast(&rs1_1) == ValueCastTo::<u8>::cast(&ex_rd) };
  let needs_rs1_and_rs1_1_3 =
    { ValueCastTo::<bool>::cast(&needs_rs1) & ValueCastTo::<bool>::cast(&rs1_1_eq_ex_rd_1) };
  let rs2_1_eq_ex_rd_1 = { ValueCastTo::<u8>::cast(&rs2_1) == ValueCastTo::<u8>::cast(&ex_rd) };
  let needs_rs2_and_rs2_1_3 =
    { ValueCastTo::<bool>::cast(&needs_rs2) & ValueCastTo::<bool>::cast(&rs2_1_eq_ex_rd_1) };
  let needs_rs1_or_needs_rs2_3 = {
    ValueCastTo::<bool>::cast(&needs_rs1_and_rs1_1_3)
      | ValueCastTo::<bool>::cast(&needs_rs2_and_rs2_1_3)
  };
  let div_result_hazard = {
    ValueCastTo::<bool>::cast(&is_ex_and_ex_rd_1)
      & ValueCastTo::<bool>::cast(&needs_rs1_or_needs_rs2_3)
  };
  let mispredict_1_or_is_jump =
    { ValueCastTo::<bool>::cast(&mispredict_1) | ValueCastTo::<bool>::cast(&is_jump_ex) };
  let mispredict_1_or_is_jumpr = {
    ValueCastTo::<bool>::cast(&mispredict_1_or_is_jump) | ValueCastTo::<bool>::cast(&is_jumpr_ex)
  };
  let need_flush_1 = {
    if mispredict_1_or_is_jumpr {
      true
    } else {
      false
    }
  };
  let load_use_or_mul_executing = {
    ValueCastTo::<bool>::cast(&load_use_hazard_mem) | ValueCastTo::<bool>::cast(&mul_executing_1)
  };
  let load_use_or_mul_result = {
    ValueCastTo::<bool>::cast(&load_use_or_mul_executing)
      | ValueCastTo::<bool>::cast(&mul_result_hazard)
  };
  let load_use_or_div_executing = {
    ValueCastTo::<bool>::cast(&load_use_or_mul_result) | ValueCastTo::<bool>::cast(&div_executing_1)
  };
  let load_use_or_div_result = {
    ValueCastTo::<bool>::cast(&load_use_or_div_executing)
      | ValueCastTo::<bool>::cast(&div_result_hazard)
  };
  let load_use_or_sb_sh = {
    ValueCastTo::<bool>::cast(&load_use_or_div_result) | ValueCastTo::<bool>::cast(&sb_sh_stall)
  };
  let not_need_flush = { !need_flush_1 };
  let data_hazard =
    { ValueCastTo::<bool>::cast(&load_use_or_sb_sh) & ValueCastTo::<bool>::cast(&not_need_flush) };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1581
  print!("@line:{:<5} {:<10}: [HazardUnit]\t", line!(), cyclize(sim.stamp));
  println!(
    "HAZARD: data_hazard={}, load_use_hazard_mem={}, sb_sh_stall={}, need_flush={}",
    if data_hazard { 1 } else { 0 },
    if load_use_hazard_mem { 1 } else { 0 },
    if sb_sh_stall { 1 } else { 0 },
    if need_flush_1 { 1 } else { 0 },
  );
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1583
  print!("@line:{:<5} {:<10}: [HazardUnit]\t", line!(), cyclize(sim.stamp));
  println!("HAZARD DETAIL: rd_mem={}, rs1={}, rs2={}, needs_rs1={}, needs_rs2={}, mem_read_mem={}, reg_write_mem={}", rd_mem, rs1_1, rs2_1, if needs_rs1 { 1 } else { 0 }, if needs_rs2 { 1 } else { 0 }, if mem_read_mem { 1 } else { 0 }, if reg_write_mem { 1 } else { 0 }, );
  let not_data_hazard = { !data_hazard };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1591
  {
    let stamp = sim.stamp - sim.stamp % 100 + 50;
    let write = ArrayWrite::new(stamp, false as usize, not_data_hazard.clone(), "HazardUnit");
    sim.id_ex_valid.write(0, write);
  };
  let not_data_hazard_1 = { !data_hazard };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1592
  {
    let stamp = sim.stamp - sim.stamp % 100 + 50;
    let write = ArrayWrite::new(stamp, false as usize, not_data_hazard_1.clone(), "HazardUnit");
    sim.if_id_valid.write(1, write);
  };
  let not_sb_sh = { !sb_sh_stall };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1594
  {
    let stamp = sim.stamp - sim.stamp % 100 + 50;
    let write = ArrayWrite::new(stamp, false as usize, not_sb_sh.clone(), "HazardUnit");
    sim.ex_mem_valid.write(0, write);
  };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1595
  {
    let stamp = sim.stamp - sim.stamp % 100 + 50;
    let write = ArrayWrite::new(stamp, false as usize, true.clone(), "HazardUnit");
    sim.mem_wb_valid.write(0, write);
  };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1596
  {
    let stamp = sim.stamp - sim.stamp % 100 + 50;
    let write = ArrayWrite::new(stamp, false as usize, data_hazard.clone(), "HazardUnit");
    sim.stall.write(0, write);
  };
  let branch_pc_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&branch_pc);
      let mask = u64::from_str_radix("111111", 2).unwrap();
      let res = (a >> 2) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let btb_update_index = { ValueCastTo::<u8>::cast(&branch_pc_slice) };
  let current_bht = { sim.bht.payload[btb_update_index as usize].clone() };
  let current_bht_eq = { ValueCastTo::<u8>::cast(&current_bht) == ValueCastTo::<u8>::cast(&3u8) };
  let current_bht_add = { ValueCastTo::<u8>::cast(&current_bht) + ValueCastTo::<u8>::cast(&1u8) };
  let new_bht_taken = {
    if current_bht_eq {
      3u8
    } else {
      current_bht_add
    }
  };
  let current_bht_eq_store_type =
    { ValueCastTo::<u8>::cast(&current_bht) == ValueCastTo::<u8>::cast(&0u8) };
  let current_bht_sub = { ValueCastTo::<u8>::cast(&current_bht) - ValueCastTo::<u8>::cast(&1u8) };
  let new_bht_not_taken = {
    if current_bht_eq_store_type {
      0u8
    } else {
      current_bht_sub
    }
  };
  let new_bht = {
    if actual_taken_1 {
      new_bht_taken
    } else {
      new_bht_not_taken
    }
  };
  if is_branch_ex {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1614
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, btb_update_index as usize, actual_target_pc_1.clone(), "HazardUnit");
      sim.btb.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1615
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, btb_update_index as usize, true.clone(), "HazardUnit");
      sim.btb_valid.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1616
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, btb_update_index as usize, new_bht.clone(), "HazardUnit");
      sim.bht.write(0, write);
    };
  }
  let if_id_rd_7 = { sim.if_id_prediction_info.payload[false as usize].clone() };
  let if_id_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&if_id_rd_7);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let current_btb_hit = { ValueCastTo::<bool>::cast(&if_id_slice) };
  let if_id_slice_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&if_id_rd_7);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 1) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let current_predict_taken = { ValueCastTo::<bool>::cast(&if_id_slice_1) };
  let if_id_slice_2 = {
    {
      let a = ValueCastTo::<u64>::cast(&if_id_rd_7);
      let mask = u64::from_str_radix("11111111111111111111111111111111", 2).unwrap();
      let res = (a >> 2) & mask;
      ValueCastTo::<u32>::cast(&res)
    }
  };
  let current_predicted_pc = { ValueCastTo::<u32>::cast(&if_id_slice_2) };
  let current_btb_and_current_p = {
    ValueCastTo::<bool>::cast(&current_btb_hit) & ValueCastTo::<bool>::cast(&current_predict_taken)
  };
  let pc_rd_1 = { sim.pc.payload[false as usize].clone() };
  let pc_rd_add = { ValueCastTo::<u32>::cast(&pc_rd_1) + ValueCastTo::<u32>::cast(&4u32) };
  let normal_next_pc = {
    if current_btb_and_current_p {
      current_predicted_pc
    } else {
      pc_rd_add
    }
  };
  let is_jump_mux_3 = {
    if is_jump_ex {
      actual_target_pc_1
    } else {
      correct_pc_1
    }
  };
  let flush_pc = {
    if is_jumpr_ex {
      target_pc_3
    } else {
      is_jump_mux_3
    }
  };
  let is_jump_mux_4 = {
    if is_jump_ex {
      actual_target_pc_1
    } else {
      correct_pc_1
    }
  };
  let flush_pc_1 = {
    if is_jumpr_ex {
      target_pc_3
    } else {
      is_jump_mux_4
    }
  };
  let data_hazard_mux = {
    if data_hazard {
      pc_rd_1
    } else {
      normal_next_pc
    }
  };
  let need_flush_mux = {
    if need_flush_1 {
      flush_pc_1
    } else {
      data_hazard_mux
    }
  };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1646
  {
    let stamp = sim.stamp - sim.stamp % 100 + 50;
    let write = ArrayWrite::new(stamp, false as usize, need_flush_mux.clone(), "HazardUnit");
    sim.pc.write(0, write);
  };
  let if_id_rd_8 = { sim.if_id_valid.payload[false as usize].clone() };
  if if_id_rd_8 {
    let need_flush_mux_1 = {
      if need_flush_1 {
        19u32
      } else {
        instruction_3
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1652
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, need_flush_mux_1.clone(), "HazardUnit");
      sim.if_id_instruction.write(0, write);
    };
    let need_flush_mux_2 = {
      if need_flush_1 {
        0u64
      } else {
        prediction_info_id
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1653
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, need_flush_mux_2.clone(), "HazardUnit");
      sim.if_id_prediction_info.write(1, write);
    };
  }
  if id_ex_rd_15 {
    let need_flush_mux_3 = {
      if need_flush_1 {
        0u64
      } else {
        control_in_3
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1655
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, need_flush_mux_3.clone(), "HazardUnit");
      sim.id_ex_control.write(0, write);
    };
    let need_flush_mux_4 = {
      if need_flush_1 {
        0u32
      } else {
        immediate_9
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1656
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, need_flush_mux_4.clone(), "HazardUnit");
      sim.id_ex_immediate.write(0, write);
    };
    let need_flush_mux_5 = {
      if need_flush_1 {
        0u8
      } else {
        rs1_1
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1657
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, need_flush_mux_5.clone(), "HazardUnit");
      sim.id_ex_rs1_idx.write(0, write);
    };
    let need_flush_mux_6 = {
      if need_flush_1 {
        0u8
      } else {
        rs2_1
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1658
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, need_flush_mux_6.clone(), "HazardUnit");
      sim.id_ex_rs2_idx.write(0, write);
    };
    let need_flush_mux_7 = {
      if need_flush_1 {
        0u64
      } else {
        prediction_info_id
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1659
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, need_flush_mux_7.clone(), "HazardUnit");
      sim.id_ex_prediction_info.write(1, write);
    };
  }

  true
}
