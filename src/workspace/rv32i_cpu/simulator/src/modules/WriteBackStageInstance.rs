use crate::simulator::Simulator;
use sim_runtime::num_bigint::{BigInt, BigUint};
use sim_runtime::*;
use std::ffi::c_void;

// Elaborating module WriteBackStageInstance
pub fn WriteBackStageInstance(sim: &mut Simulator) -> bool {
  let wbs_lw_data = { sim.SRAM_rdata.payload[false as usize].clone() };
  let ex_result_in = { sim.mem_wb_ex_result.payload[false as usize].clone() };
  let control_in = { sim.mem_wb_control.payload[false as usize].clone() };
  let addr_in = { sim.mem_wb_addr.payload[false as usize].clone() };
  let reg_write = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 7) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let mem_to_reg = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 8) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let wb_rd = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in);
      let mask = u64::from_str_radix("11111", 2).unwrap();
      let res = (a >> 25) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let load_type = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in);
      let mask = u64::from_str_radix("111", 2).unwrap();
      let res = (a >> 14) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let byte_offset = {
    {
      let a = ValueCastTo::<u64>::cast(&addr_in);
      let mask = u64::from_str_radix("11", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let wbs_byte0 = {
    {
      let a = ValueCastTo::<u64>::cast(&wbs_lw_data);
      let mask = u64::from_str_radix("11111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let wbs_byte1 = {
    {
      let a = ValueCastTo::<u64>::cast(&wbs_lw_data);
      let mask = u64::from_str_radix("11111111", 2).unwrap();
      let res = (a >> 8) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let wbs_byte2 = {
    {
      let a = ValueCastTo::<u64>::cast(&wbs_lw_data);
      let mask = u64::from_str_radix("11111111", 2).unwrap();
      let res = (a >> 16) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let wbs_byte3 = {
    {
      let a = ValueCastTo::<u64>::cast(&wbs_lw_data);
      let mask = u64::from_str_radix("11111111", 2).unwrap();
      let res = (a >> 24) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let byte_offset_eq = { ValueCastTo::<u8>::cast(&byte_offset) == ValueCastTo::<u8>::cast(&0u8) };
  let byte_offset_eq_1 = { ValueCastTo::<u8>::cast(&byte_offset) == ValueCastTo::<u8>::cast(&1u8) };
  let byte_offset_eq_2 = { ValueCastTo::<u8>::cast(&byte_offset) == ValueCastTo::<u8>::cast(&2u8) };
  let byte_offset_mux = {
    if byte_offset_eq_2 {
      wbs_byte2
    } else {
      wbs_byte3
    }
  };
  let byte_offset_mux_1 = {
    if byte_offset_eq_1 {
      wbs_byte1
    } else {
      byte_offset_mux
    }
  };
  let wbs_selected_byte = {
    if byte_offset_eq {
      wbs_byte0
    } else {
      byte_offset_mux_1
    }
  };
  let wbs_half0 = {
    {
      let a = ValueCastTo::<u64>::cast(&wbs_lw_data);
      let mask = u64::from_str_radix("1111111111111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u16>::cast(&res)
    }
  };
  let wbs_half1 = {
    {
      let a = ValueCastTo::<u64>::cast(&wbs_lw_data);
      let mask = u64::from_str_radix("1111111111111111", 2).unwrap();
      let res = (a >> 16) & mask;
      ValueCastTo::<u16>::cast(&res)
    }
  };
  let byte_offset_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&byte_offset);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 1) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let byte_offset_eq_3 =
    { ValueCastTo::<bool>::cast(&byte_offset_slice) == ValueCastTo::<bool>::cast(&false) };
  let wbs_selected_half = {
    if byte_offset_eq_3 {
      wbs_half0
    } else {
      wbs_half1
    }
  };
  let wbs_byte_sign = {
    {
      let a = ValueCastTo::<u64>::cast(&wbs_selected_byte);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 7) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let wbs_byte_mux = {
    if wbs_byte_sign {
      16777215u32
    } else {
      0u32
    }
  };
  let wbs_byte_cat_wbs_selected = {
    {
      let a = ValueCastTo::<BigUint>::cast(&wbs_byte_mux);
      let b = ValueCastTo::<BigUint>::cast(&wbs_selected_byte);
      let c = (a << 8) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let wbs_lb_data = { ValueCastTo::<u32>::cast(&wbs_byte_cat_wbs_selected) };
  let cat_wbs_selected = {
    {
      let a = ValueCastTo::<BigUint>::cast(&0u32);
      let b = ValueCastTo::<BigUint>::cast(&wbs_selected_byte);
      let c = (a << 8) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let wbs_lbu_data = { ValueCastTo::<u32>::cast(&cat_wbs_selected) };
  let wbs_half_sign = {
    {
      let a = ValueCastTo::<u64>::cast(&wbs_selected_half);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 15) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let wbs_half_mux = {
    if wbs_half_sign {
      65535u16
    } else {
      0u16
    }
  };
  let wbs_half_cat_wbs_selected = {
    {
      let a = ValueCastTo::<BigUint>::cast(&wbs_half_mux);
      let b = ValueCastTo::<BigUint>::cast(&wbs_selected_half);
      let c = (a << 16) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let wbs_lh_data = { ValueCastTo::<u32>::cast(&wbs_half_cat_wbs_selected) };
  let cat_wbs_selected_1 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&0u16);
      let b = ValueCastTo::<BigUint>::cast(&wbs_selected_half);
      let c = (a << 16) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let wbs_lhu_data = { ValueCastTo::<u32>::cast(&cat_wbs_selected_1) };
  let load_type_eq = { ValueCastTo::<u8>::cast(&load_type) == ValueCastTo::<u8>::cast(&0u8) };
  let load_type_eq_1 = { ValueCastTo::<u8>::cast(&load_type) == ValueCastTo::<u8>::cast(&1u8) };
  let load_type_eq_2 = { ValueCastTo::<u8>::cast(&load_type) == ValueCastTo::<u8>::cast(&2u8) };
  let load_type_eq_3 = { ValueCastTo::<u8>::cast(&load_type) == ValueCastTo::<u8>::cast(&4u8) };
  let load_type_mux = {
    if load_type_eq_3 {
      wbs_lbu_data
    } else {
      wbs_lhu_data
    }
  };
  let load_type_mux_1 = {
    if load_type_eq_2 {
      wbs_lw_data
    } else {
      load_type_mux
    }
  };
  let load_type_mux_2 = {
    if load_type_eq_1 {
      wbs_lh_data
    } else {
      load_type_mux_1
    }
  };
  let processed_mem_data = {
    if load_type_eq {
      wbs_lb_data
    } else {
      load_type_mux_2
    }
  };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1424
  print!("@line:{:<5} {:<10}: [WriteBackStageInstance]\t", line!(), cyclize(sim.stamp));
  println!(
    "WB LOAD PROCESS: mem_data_in={:08x}, load_type={:03b}, processed_mem_data={:08x}",
    wbs_lw_data, load_type, processed_mem_data,
  );
  let wb_data = {
    if mem_to_reg {
      processed_mem_data
    } else {
      ex_result_in
    }
  };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1430
  print!("@line:{:<5} {:<10}: [WriteBackStageInstance]\t", line!(), cyclize(sim.stamp));
  println!("WB STAGE: ex_result_in={:08x}, mem_to_reg={}, wb_data={:08x}, wb_rd={}, reg_write={}, load_type={:03b}", ex_result_in, if mem_to_reg { 1 } else { 0 }, wb_data, wb_rd, if reg_write { 1 } else { 0 }, load_type, );
  let mem_wb_rd_3 = { sim.mem_wb_valid.payload[false as usize].clone() };
  if mem_wb_rd_3 {
    if reg_write {
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1436
      print!("@line:{:<5} {:<10}: [WriteBackStageInstance]\t", line!(), cyclize(sim.stamp));
      println!("WB WRITE: reg[{}] = {:08x}", wb_rd, wb_data,);
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1437
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write =
          ArrayWrite::new(stamp, wb_rd as usize, wb_data.clone(), "WriteBackStageInstance");
        sim.reg_file.write(0, write);
      };
    }
  }
  let writeback_signals = { ValueCastTo::<u64>::cast(&control_in) };
  sim.writeback_signals_value = Some(writeback_signals.clone());

  true
}
