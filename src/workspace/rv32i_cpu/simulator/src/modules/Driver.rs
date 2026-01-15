use crate::simulator::Simulator;
use sim_runtime::num_bigint::{BigInt, BigUint};
use sim_runtime::*;
use std::ffi::c_void;

// Elaborating module Driver
pub fn Driver(sim: &mut Simulator) -> bool {
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1669
  ();
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1669
  {
    let stamp = sim.stamp - sim.stamp % 100 + 100;
    sim.FetchStageInstance_event.push_back(stamp)
  };

  true
}
