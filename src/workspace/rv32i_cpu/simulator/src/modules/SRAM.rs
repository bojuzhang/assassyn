use crate::simulator::Simulator;
use sim_runtime::num_bigint::{BigInt, BigUint};
use sim_runtime::*;
use std::ffi::c_void;

// Elaborating module SRAM
pub fn SRAM(sim: &mut Simulator) -> bool {
  let sram_we_and_sram_re = {
    ValueCastTo::<bool>::cast(
      &{
        if let Some(x) = &sim.sram_we_value {
          x
        } else {
          panic!("Value sram_we invalid!");
        }
      }
      .clone(),
    ) & ValueCastTo::<bool>::cast(
      &{
        if let Some(x) = &sim.sram_re_value {
          x
        } else {
          panic!("Value sram_re invalid!");
        }
      }
      .clone(),
    )
  };
  let not_sram_we = { !sram_we_and_sram_re };
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1324
  assert!(not_sram_we);
  if {
    if let Some(x) = &sim.sram_we_value {
      x
    } else {
      panic!("Value sram_we invalid!");
    }
  }
  .clone()
  {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1324
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(
        stamp,
        {
          if let Some(x) = &sim.sram_addr_value {
            x
          } else {
            panic!("Value sram_addr invalid!");
          }
        }
        .clone() as usize,
        {
          if let Some(x) = &sim.sram_wdata_value {
            x
          } else {
            panic!("Value sram_wdata invalid!");
          }
        }
        .clone()
        .clone(),
        "SRAM",
      );
      sim.SRAM_val.write(0, write);
    };
  }
  if {
    if let Some(x) = &sim.sram_re_value {
      x
    } else {
      panic!("Value sram_re invalid!");
    }
  }
  .clone()
  {
    let SRAM_val_rd = {
      sim.SRAM_val.payload[{
        if let Some(x) = &sim.sram_addr_value {
          x
        } else {
          panic!("Value sram_addr invalid!");
        }
      }
      .clone() as usize]
        .clone()
    };
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1324
    {
      let stamp = sim.stamp - sim.stamp % 100 + 50;
      let write = ArrayWrite::new(stamp, false as usize, SRAM_val_rd.clone(), "SRAM");
      sim.SRAM_rdata.write(0, write);
    };
  }

  true
}
