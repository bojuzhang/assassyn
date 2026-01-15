use crate::simulator::Simulator;
use sim_runtime::num_bigint::{BigInt, BigUint};
use sim_runtime::*;
use std::ffi::c_void;

// Elaborating module FetchStageInstance
pub fn FetchStageInstance(sim: &mut Simulator) -> bool {
  let current_pc = { sim.pc.payload[false as usize].clone() };
  let word_addr_1 = { ValueCastTo::<u32>::cast(&current_pc) >> ValueCastTo::<u32>::cast(&2u32) };
  let instruction_2 = { sim.instruction_memory.payload[word_addr_1 as usize].clone() };
  let current_pc_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&current_pc);
      let mask = u64::from_str_radix("111111", 2).unwrap();
      let res = (a >> 2) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let btb_index = { ValueCastTo::<u8>::cast(&current_pc_slice) };
  let btb_entry = { sim.btb.payload[btb_index as usize].clone() };
  let bht_entry = { sim.bht.payload[btb_index as usize].clone() };
  let btb_hit_1 = { sim.btb_valid.payload[btb_index as usize].clone() };
  let bht_entry_ge = { ValueCastTo::<u8>::cast(&bht_entry) >= ValueCastTo::<u8>::cast(&2u8) };
  let predict_taken_1 = {
    if bht_entry_ge {
      true
    } else {
      false
    }
  };
  let btb_hit_and_predict_taken =
    { ValueCastTo::<bool>::cast(&btb_hit_1) & ValueCastTo::<bool>::cast(&predict_taken_1) };
  let current_pc_add = { ValueCastTo::<u32>::cast(&current_pc) + ValueCastTo::<u32>::cast(&4u32) };
  let predicted_pc_1 = {
    if btb_hit_and_predict_taken {
      btb_entry
    } else {
      current_pc_add
    }
  };
  let predicted_pc_cat_predict = {
    {
      let a = ValueCastTo::<BigUint>::cast(&predicted_pc_1);
      let b = ValueCastTo::<BigUint>::cast(&predict_taken_1);
      let c = (a << 1) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let predicted_pc_cat_btb_hit = {
    {
      let a = ValueCastTo::<BigUint>::cast(&predicted_pc_cat_predict);
      let b = ValueCastTo::<BigUint>::cast(&btb_hit_1);
      let c = (a << 1) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let prediction_info = { ValueCastTo::<u64>::cast(&predicted_pc_cat_btb_hit) };
  let if_id_rd_5 = { sim.if_id_valid.payload[false as usize].clone() };
  if if_id_rd_5 {
    let stall_rd = { sim.stall.payload[false as usize].clone() };
    let stall_rd_mux = {
      if stall_rd {
        0u32
      } else {
        current_pc
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:157
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, stall_rd_mux.clone(), "FetchStageInstance");
      sim.if_id_pc.write(0, write);
    };
    let stall_rd_mux_1 = {
      if stall_rd {
        false
      } else {
        true
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:158
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, stall_rd_mux_1.clone(), "FetchStageInstance");
      sim.if_id_valid.write(0, write);
    };
    let stall_rd_mux_2 = {
      if stall_rd {
        0u64
      } else {
        prediction_info
      }
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:159
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write =
        ArrayWrite::new(stamp, false as usize, stall_rd_mux_2.clone(), "FetchStageInstance");
      sim.if_id_prediction_info.write(0, write);
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:160
    print!("@line:{:<5} {:<10}: [FetchStageInstance]\t", line!(), cyclize(sim.stamp));
    println!("IF: PC={:08x}, Instruction={:08x}", current_pc, instruction_2,);
  }
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:162
  ();
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:162
  {
    let stamp = sim.stamp - sim.stamp % 100 + 100;
    sim.DecodeStageInstance_event.push_back(stamp)
  };
  let stall_rd_1 = { sim.stall.payload[false as usize].clone() };
  let stall_rd_mux_3 = {
    if stall_rd_1 {
      0u32
    } else {
      instruction_2
    }
  };
  let if_id_rd_6 = { sim.if_id_instruction.payload[false as usize].clone() };
  let if_id_mux_11 = {
    if if_id_rd_5 {
      stall_rd_mux_3
    } else {
      if_id_rd_6
    }
  };
  let fetch_signals = { ValueCastTo::<u32>::cast(&if_id_mux_11) };
  sim.fetch_signals_value = Some(fetch_signals.clone());

  true
}
