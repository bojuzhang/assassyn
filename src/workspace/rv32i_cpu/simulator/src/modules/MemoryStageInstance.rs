use crate::simulator::Simulator;
use sim_runtime::num_bigint::{BigInt, BigUint};
use sim_runtime::*;
use std::ffi::c_void;

// Elaborating module MemoryStageInstance
pub fn MemoryStageInstance(sim: &mut Simulator) -> bool {
  let pc_in = { sim.ex_mem_pc.payload[false as usize].clone() };
  let addr_in_1 = { sim.ex_mem_result.payload[false as usize].clone() };
  let data_in = { sim.ex_mem_data.payload[false as usize].clone() };
  let control_in_1 = { sim.ex_mem_control.payload[false as usize].clone() };
  let mem_read = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_1);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 5) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let mem_write = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_1);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 6) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let store_type = {
    {
      let a = ValueCastTo::<u64>::cast(&control_in_1);
      let mask = u64::from_str_radix("11", 2).unwrap();
      let res = (a >> 22) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let word_addr = { ValueCastTo::<u32>::cast(&addr_in_1) >> ValueCastTo::<u32>::cast(&2u32) };
  let byte_offset_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&addr_in_1);
      let mask = u64::from_str_radix("11", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let ex_mem_rd_4 = { sim.ex_mem_valid.payload[false as usize].clone() };
  let mem_read_and_ex_mem =
    { ValueCastTo::<bool>::cast(&mem_read) & ValueCastTo::<bool>::cast(&ex_mem_rd_4) };
  if mem_read_and_ex_mem {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1228
    print!("@line:{:<5} {:<10}: [MemoryStageInstance]\t", line!(), cyclize(sim.stamp));
    println!(
      "MEM READ: addr={:08x}, word_addr={:08x}, byte_offset={}",
      addr_in_1, word_addr, byte_offset_1,
    );
  }
  let mem_write_and_ex_mem =
    { ValueCastTo::<bool>::cast(&mem_write) & ValueCastTo::<bool>::cast(&ex_mem_rd_4) };
  if mem_write_and_ex_mem {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1230
    print!("@line:{:<5} {:<10}: [MemoryStageInstance]\t", line!(), cyclize(sim.stamp));
    println!(
      "MEM WRITE: addr={:08x}, word_addr={:08x}, byte_offset={}, data={:08x}",
      addr_in_1, word_addr, byte_offset_1, data_in,
    );
  }
  let is_sb = { ValueCastTo::<u8>::cast(&store_type) == ValueCastTo::<u8>::cast(&0u8) };
  let is_sh = { ValueCastTo::<u8>::cast(&store_type) == ValueCastTo::<u8>::cast(&1u8) };
  let is_sw = { ValueCastTo::<u8>::cast(&store_type) == ValueCastTo::<u8>::cast(&2u8) };
  let is_sb_or_is_sh = { ValueCastTo::<bool>::cast(&is_sb) | ValueCastTo::<bool>::cast(&is_sh) };
  let needs_rmw =
    { ValueCastTo::<bool>::cast(&mem_write) & ValueCastTo::<bool>::cast(&is_sb_or_is_sh) };
  let data_byte = {
    {
      let a = ValueCastTo::<u64>::cast(&data_in);
      let mask = u64::from_str_radix("11111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let data_half = {
    {
      let a = ValueCastTo::<u64>::cast(&data_in);
      let mask = u64::from_str_radix("1111111111111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u16>::cast(&res)
    }
  };
  let current_state = { sim.sb_sh_state.payload[false as usize].clone() };
  let is_idle = { ValueCastTo::<u8>::cast(&current_state) == ValueCastTo::<u8>::cast(&0u8) };
  let is_write_phase = { ValueCastTo::<u8>::cast(&current_state) == ValueCastTo::<u8>::cast(&2u8) };
  let saved_addr = { sim.sb_sh_addr.payload[false as usize].clone() };
  let saved_data = { sim.sb_sh_data.payload[false as usize].clone() };
  let saved_type = { sim.sb_sh_type.payload[false as usize].clone() };
  let saved_word_addr =
    { ValueCastTo::<u32>::cast(&saved_addr) >> ValueCastTo::<u32>::cast(&2u32) };
  let saved_byte_offset = {
    {
      let a = ValueCastTo::<u64>::cast(&saved_addr);
      let mask = u64::from_str_radix("11", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let saved_data_byte = {
    {
      let a = ValueCastTo::<u64>::cast(&saved_data);
      let mask = u64::from_str_radix("11111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u8>::cast(&res)
    }
  };
  let saved_data_half = {
    {
      let a = ValueCastTo::<u64>::cast(&saved_data);
      let mask = u64::from_str_radix("1111111111111111", 2).unwrap();
      let res = (a >> 0) & mask;
      ValueCastTo::<u16>::cast(&res)
    }
  };
  let saved_byte_eq =
    { ValueCastTo::<u8>::cast(&saved_byte_offset) == ValueCastTo::<u8>::cast(&0u8) };
  let cat_saved_data = {
    {
      let a = ValueCastTo::<BigUint>::cast(&0u32);
      let b = ValueCastTo::<BigUint>::cast(&saved_data_byte);
      let c = (a << 8) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let saved_byte_eq_1 =
    { ValueCastTo::<u8>::cast(&saved_byte_offset) == ValueCastTo::<u8>::cast(&1u8) };
  let cat_saved_data_1 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&0u16);
      let b = ValueCastTo::<BigUint>::cast(&saved_data_byte);
      let c = (a << 8) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let cat_saved_cat = {
    {
      let a = ValueCastTo::<BigUint>::cast(&cat_saved_data_1);
      let b = ValueCastTo::<BigUint>::cast(&0u8);
      let c = (a << 8) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let saved_byte_eq_2 =
    { ValueCastTo::<u8>::cast(&saved_byte_offset) == ValueCastTo::<u8>::cast(&2u8) };
  let cat_saved_data_2 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&0u8);
      let b = ValueCastTo::<BigUint>::cast(&saved_data_byte);
      let c = (a << 8) | b;
      ValueCastTo::<u16>::cast(&c)
    }
  };
  let cat_saved_cat_1 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&cat_saved_data_2);
      let b = ValueCastTo::<BigUint>::cast(&0u16);
      let c = (a << 16) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let saved_data_cat = {
    {
      let a = ValueCastTo::<BigUint>::cast(&saved_data_byte);
      let b = ValueCastTo::<BigUint>::cast(&0u32);
      let c = (a << 24) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let saved_byte_mux = {
    if saved_byte_eq_2 {
      cat_saved_cat_1
    } else {
      saved_data_cat
    }
  };
  let saved_byte_mux_1 = {
    if saved_byte_eq_1 {
      cat_saved_cat
    } else {
      saved_byte_mux
    }
  };
  let saved_byte_mux_2 = {
    if saved_byte_eq {
      cat_saved_data
    } else {
      saved_byte_mux_1
    }
  };
  let sb_data_saved = { ValueCastTo::<u32>::cast(&saved_byte_mux_2) };
  let saved_byte_eq_3 =
    { ValueCastTo::<u8>::cast(&saved_byte_offset) == ValueCastTo::<u8>::cast(&0u8) };
  let saved_byte_eq_4 =
    { ValueCastTo::<u8>::cast(&saved_byte_offset) == ValueCastTo::<u8>::cast(&1u8) };
  let saved_byte_eq_5 =
    { ValueCastTo::<u8>::cast(&saved_byte_offset) == ValueCastTo::<u8>::cast(&2u8) };
  let saved_byte_mux_3 = {
    if saved_byte_eq_5 {
      4278255615u32
    } else {
      16777215u32
    }
  };
  let saved_byte_mux_4 = {
    if saved_byte_eq_4 {
      4294902015u32
    } else {
      saved_byte_mux_3
    }
  };
  let sb_mask_saved = {
    if saved_byte_eq_3 {
      4294967040u32
    } else {
      saved_byte_mux_4
    }
  };
  let saved_byte_slice = {
    {
      let a = ValueCastTo::<u64>::cast(&saved_byte_offset);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 1) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let saved_byte_eq_6 =
    { ValueCastTo::<bool>::cast(&saved_byte_slice) == ValueCastTo::<bool>::cast(&false) };
  let cat_saved_data_3 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&0u16);
      let b = ValueCastTo::<BigUint>::cast(&saved_data_half);
      let c = (a << 16) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let saved_data_cat_1 = {
    {
      let a = ValueCastTo::<BigUint>::cast(&saved_data_half);
      let b = ValueCastTo::<BigUint>::cast(&0u16);
      let c = (a << 16) | b;
      ValueCastTo::<u32>::cast(&c)
    }
  };
  let saved_byte_mux_6 = {
    if saved_byte_eq_6 {
      cat_saved_data_3
    } else {
      saved_data_cat_1
    }
  };
  let sh_data_saved = { ValueCastTo::<u32>::cast(&saved_byte_mux_6) };
  let saved_byte_slice_1 = {
    {
      let a = ValueCastTo::<u64>::cast(&saved_byte_offset);
      let mask = u64::from_str_radix("1", 2).unwrap();
      let res = (a >> 1) & mask;
      ValueCastTo::<bool>::cast(&res)
    }
  };
  let saved_byte_eq_7 =
    { ValueCastTo::<bool>::cast(&saved_byte_slice_1) == ValueCastTo::<bool>::cast(&false) };
  let sh_mask_saved = {
    if saved_byte_eq_7 {
      4294901760u32
    } else {
      65535u32
    }
  };
  let is_idle_and_ex_mem =
    { ValueCastTo::<bool>::cast(&is_idle) & ValueCastTo::<bool>::cast(&ex_mem_rd_4) };
  let sb_sh_active =
    { ValueCastTo::<bool>::cast(&is_idle_and_ex_mem) & ValueCastTo::<bool>::cast(&needs_rmw) };
  let original_data = { sim.SRAM_rdata.payload[false as usize].clone() };
  let is_sb_saved = { ValueCastTo::<u8>::cast(&saved_type) == ValueCastTo::<u8>::cast(&0u8) };
  let original_data_and_sb_mask =
    { ValueCastTo::<u32>::cast(&original_data) & ValueCastTo::<u32>::cast(&sb_mask_saved) };
  let merged_sb = {
    ValueCastTo::<u32>::cast(&original_data_and_sb_mask) | ValueCastTo::<u32>::cast(&sb_data_saved)
  };
  let original_data_and_sh_mask =
    { ValueCastTo::<u32>::cast(&original_data) & ValueCastTo::<u32>::cast(&sh_mask_saved) };
  let merged_sh = {
    ValueCastTo::<u32>::cast(&original_data_and_sh_mask) | ValueCastTo::<u32>::cast(&sh_data_saved)
  };
  let rmw_write_data = {
    if is_sb_saved {
      merged_sb
    } else {
      merged_sh
    }
  };
  let mem_wb_rd_4 = { sim.mem_wb_valid.payload[false as usize].clone() };
  let do_rmw_write =
    { ValueCastTo::<bool>::cast(&is_write_phase) & ValueCastTo::<bool>::cast(&mem_wb_rd_4) };
  let is_idle_and_ex_mem_1 =
    { ValueCastTo::<bool>::cast(&is_idle) & ValueCastTo::<bool>::cast(&ex_mem_rd_4) };
  let is_idle_and_needs_rmw_1 =
    { ValueCastTo::<bool>::cast(&is_idle_and_ex_mem_1) & ValueCastTo::<bool>::cast(&needs_rmw) };
  let do_rmw_read = {
    ValueCastTo::<bool>::cast(&is_idle_and_needs_rmw_1) & ValueCastTo::<bool>::cast(&mem_wb_rd_4)
  };
  let is_idle_and_ex_mem_2 =
    { ValueCastTo::<bool>::cast(&is_idle) & ValueCastTo::<bool>::cast(&ex_mem_rd_4) };
  let is_idle_and_mem_write =
    { ValueCastTo::<bool>::cast(&is_idle_and_ex_mem_2) & ValueCastTo::<bool>::cast(&mem_write) };
  let is_idle_and_is_sw =
    { ValueCastTo::<bool>::cast(&is_idle_and_mem_write) & ValueCastTo::<bool>::cast(&is_sw) };
  let do_sw_write =
    { ValueCastTo::<bool>::cast(&is_idle_and_is_sw) & ValueCastTo::<bool>::cast(&mem_wb_rd_4) };
  let is_idle_and_ex_mem_3 =
    { ValueCastTo::<bool>::cast(&is_idle) & ValueCastTo::<bool>::cast(&ex_mem_rd_4) };
  let is_idle_and_mem_read =
    { ValueCastTo::<bool>::cast(&is_idle_and_ex_mem_3) & ValueCastTo::<bool>::cast(&mem_read) };
  let not_mem_write = { !mem_write };
  let is_idle_and_not_mem = {
    ValueCastTo::<bool>::cast(&is_idle_and_mem_read) & ValueCastTo::<bool>::cast(&not_mem_write)
  };
  let do_load_read =
    { ValueCastTo::<bool>::cast(&is_idle_and_not_mem) & ValueCastTo::<bool>::cast(&mem_wb_rd_4) };
  let sram_we =
    { ValueCastTo::<bool>::cast(&do_rmw_write) | ValueCastTo::<bool>::cast(&do_sw_write) };
  sim.sram_we_value = Some(sram_we.clone());
  let sram_re =
    { ValueCastTo::<bool>::cast(&do_rmw_read) | ValueCastTo::<bool>::cast(&do_load_read) };
  sim.sram_re_value = Some(sram_re.clone());
  let sram_addr = {
    if do_rmw_write {
      saved_word_addr
    } else {
      word_addr
    }
  };
  sim.sram_addr_value = Some(sram_addr.clone());
  let rmw_write_data_uint = { ValueCastTo::<u32>::cast(&rmw_write_data) };
  let sram_wdata = {
    if do_rmw_write {
      rmw_write_data_uint
    } else {
      data_in
    }
  };
  sim.sram_wdata_value = Some(sram_wdata.clone());
  if do_rmw_write {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1328
    print!("@line:{:<5} {:<10}: [MemoryStageInstance]\t", line!(), cyclize(sim.stamp));
    println!(
      "MEM RMW WRITE PHASE: original_data={:08x}, saved_data={:08x}",
      original_data, saved_data,
    );
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1329
    print!("@line:{:<5} {:<10}: [MemoryStageInstance]\t", line!(), cyclize(sim.stamp));
    println!("MEM RMW WRITE: addr={:08x}, wdata={:08x}", saved_word_addr, rmw_write_data,);
  }
  if do_rmw_read {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1331
    print!("@line:{:<5} {:<10}: [MemoryStageInstance]\t", line!(), cyclize(sim.stamp));
    println!("MEM RMW READ: word_addr={:08x}", word_addr,);
  }
  if do_sw_write {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1333
    print!("@line:{:<5} {:<10}: [MemoryStageInstance]\t", line!(), cyclize(sim.stamp));
    println!("MEM WRITE SW: word_addr={:08x}, wdata={:08x}", word_addr, data_in,);
  }
  if do_load_read {
    // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1335
    print!("@line:{:<5} {:<10}: [MemoryStageInstance]\t", line!(), cyclize(sim.stamp));
    println!("MEM READ SRAM: word_addr={:08x}", word_addr,);
  }
  if mem_wb_rd_4 {
    if is_write_phase {
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1341
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write = ArrayWrite::new(stamp, false as usize, 0u8.clone(), "MemoryStageInstance");
        sim.sb_sh_state.write(0, write);
      };
    }
    let ex_mem_rd_5 = { sim.ex_mem_valid.payload[false as usize].clone() };
    let is_idle_and_ex_mem_4 =
      { ValueCastTo::<bool>::cast(&is_idle) & ValueCastTo::<bool>::cast(&ex_mem_rd_5) };
    if is_idle_and_ex_mem_4 {
      if needs_rmw {
        // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1347
        {
          let stamp = sim.stamp - sim.stamp % 100 + 50;
          let write =
            ArrayWrite::new(stamp, false as usize, addr_in_1.clone(), "MemoryStageInstance");
          sim.sb_sh_addr.write(0, write);
        };
        // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1348
        {
          let stamp = sim.stamp - sim.stamp % 100 + 50;
          let write =
            ArrayWrite::new(stamp, false as usize, data_in.clone(), "MemoryStageInstance");
          sim.sb_sh_data.write(0, write);
        };
        // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1349
        {
          let stamp = sim.stamp - sim.stamp % 100 + 50;
          let write =
            ArrayWrite::new(stamp, false as usize, store_type.clone(), "MemoryStageInstance");
          sim.sb_sh_type.write(0, write);
        };
        // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1351
        {
          let stamp = sim.stamp - sim.stamp % 100 + 50;
          let write = ArrayWrite::new(stamp, false as usize, 2u8.clone(), "MemoryStageInstance");
          sim.sb_sh_state.write(0, write);
        };
      }
      let SRAM_rdata_rd_2 = { sim.SRAM_rdata.payload[false as usize].clone() };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1353
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write =
          ArrayWrite::new(stamp, false as usize, SRAM_rdata_rd_2.clone(), "MemoryStageInstance");
        sim.mem_wb_mem_data.write(0, write);
      };
      let mem_wb_rd_5 = { sim.mem_wb_mem_data.payload[false as usize].clone() };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1354
      print!("@line:{:<5} {:<10}: [MemoryStageInstance]\t", line!(), cyclize(sim.stamp));
      println!("MEM UPDATE: mem_wb_mem_data={:08x}", mem_wb_rd_5,);
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1355
      print!("@line:{:<5} {:<10}: [MemoryStageInstance]\t", line!(), cyclize(sim.stamp));
      println!("MEM SRAM DOUT: data_sram.dout={:08x}", SRAM_rdata_rd_2,);
    }
    let not_ex_mem = { !ex_mem_rd_5 };
    let is_idle_and_not_ex =
      { ValueCastTo::<bool>::cast(&is_idle) & ValueCastTo::<bool>::cast(&not_ex_mem) };
    if is_idle_and_not_ex {
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1358
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write = ArrayWrite::new(stamp, false as usize, 0u32.clone(), "MemoryStageInstance");
        sim.mem_wb_mem_data.write(0, write);
      };
    }
    let not_needs_rmw = { !needs_rmw };
    let is_idle_and_not_needs =
      { ValueCastTo::<bool>::cast(&is_idle) & ValueCastTo::<bool>::cast(&not_needs_rmw) };
    let is_idle_or_is_write = {
      ValueCastTo::<bool>::cast(&is_idle_and_not_needs) | ValueCastTo::<bool>::cast(&is_write_phase)
    };
    if is_idle_or_is_write {
      let ex_mem_rd_6 = { sim.ex_mem_valid.payload[false as usize].clone() };
      let ex_mem_mux = {
        if ex_mem_rd_6 {
          control_in_1
        } else {
          0u64
        }
      };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1362
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write =
          ArrayWrite::new(stamp, false as usize, ex_mem_mux.clone(), "MemoryStageInstance");
        sim.mem_wb_control.write(0, write);
      };
      let ex_mem_rd_7 = { sim.ex_mem_result.payload[false as usize].clone() };
      let ex_mem_mux_1 = {
        if ex_mem_rd_6 {
          ex_mem_rd_7
        } else {
          0u32
        }
      };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1363
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write =
          ArrayWrite::new(stamp, false as usize, ex_mem_mux_1.clone(), "MemoryStageInstance");
        sim.mem_wb_ex_result.write(0, write);
      };
      let ex_mem_mux_2 = {
        if ex_mem_rd_6 {
          addr_in_1
        } else {
          0u32
        }
      };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1364
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write =
          ArrayWrite::new(stamp, false as usize, ex_mem_mux_2.clone(), "MemoryStageInstance");
        sim.mem_wb_addr.write(0, write);
      };
    }
    let is_idle_and_needs_rmw_2 =
      { ValueCastTo::<bool>::cast(&is_idle) & ValueCastTo::<bool>::cast(&needs_rmw) };
    if is_idle_and_needs_rmw_2 {
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1368
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write = ArrayWrite::new(stamp, false as usize, 0u64.clone(), "MemoryStageInstance");
        sim.mem_wb_control.write(0, write);
      };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1369
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write = ArrayWrite::new(stamp, false as usize, 0u32.clone(), "MemoryStageInstance");
        sim.mem_wb_ex_result.write(0, write);
      };
      // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1370
      {
        let stamp = sim.stamp - sim.stamp % 100 + 50;
        let write = ArrayWrite::new(stamp, false as usize, 0u32.clone(), "MemoryStageInstance");
        sim.mem_wb_addr.write(0, write);
      };
    }
  }
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1372
  ();
  // @/mnt/d/Tomato_Fish/assassyn/src/rv32i_cpu.py:1372
  {
    let stamp = sim.stamp - sim.stamp % 100 + 100;
    sim.WriteBackStageInstance_event.push_back(stamp)
  };
  let sb_sh_cat_control_in = {
    {
      let a = ValueCastTo::<BigUint>::cast(&sb_sh_active);
      let b = ValueCastTo::<BigUint>::cast(&control_in_1);
      let c = (a << 48) | b;
      ValueCastTo::<u64>::cast(&c)
    }
  };
  let memory_signals = { ValueCastTo::<u64>::cast(&sb_sh_cat_control_in) };
  sim.memory_signals_value = Some(memory_signals.clone());

  true
}
