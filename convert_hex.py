#!/usr/bin/env python3
"""
将 Verilog HEX 格式转换为普通 HEX 格式

Verilog HEX 格式：
- 使用 @ 符号标记地址
- 数据以空格分隔的字节形式表示

普通 HEX 格式：
- 每一行一个 32 位的十六进制数
- 格式为 0xXXXXXXXX
- 每一行严格对应其真实位置（按字节地址/4）
- 中间有跳跃的情况用 0x00000000 补齐
"""

import sys
import argparse


def parse_verilog_hex(filename):
    """
    解析 Verilog HEX 格式文件
    
    返回: dict {address: value}，其中 address 是字节地址，value 是 32 位字
    """
    data = {}
    current_addr = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是地址行
            if line.startswith('@'):
                current_addr = int(line[1:], 16)
                continue
            
            # 解析数据字节
            bytes_list = line.split()
            
            # 将字节转换为 32 位字（小端序）
            for i in range(0, len(bytes_list), 4):
                word_bytes = bytes_list[i:i+4]
                if len(word_bytes) < 4:
                    # 填充不足的字节
                    word_bytes += ['00'] * (4 - len(word_bytes))
                
                # 小端序：低字节在前
                word_value = 0
                for j in range(4):
                    byte_value = int(word_bytes[j], 16)
                    word_value |= (byte_value << (j * 8))
                
                # 计算字地址（字节地址 / 4）
                word_addr = current_addr // 4
                data[word_addr] = word_value
                
                # 更新地址
                current_addr += 4
    
    return data


def convert_to_simple_hex(data, output_filename=None):
    """
    将解析的数据转换为普通 HEX 格式
    
    Args:
        data: dict {word_address: word_value}
        output_filename: 输出文件名（可选）
    
    Returns:
        list of hex strings
    """
    if not data:
        return []
    
    # 找到最大地址
    max_addr = max(data.keys())
    
    # 生成完整的地址范围
    result = []
    for addr in range(max_addr + 1):
        if addr in data:
            value = data[addr]
        else:
            value = 0x00000000
        
        result.append(f"0x{value:08x}")
    
    # 写入文件（如果指定了输出文件）
    if output_filename:
        with open(output_filename, 'w') as f:
            for hex_str in result:
                f.write(hex_str + '\n')
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='将 Verilog HEX 格式转换为普通 HEX 格式'
    )
    parser.add_argument(
        'input_file',
        help='输入文件（Verilog HEX 格式）'
    )
    parser.add_argument(
        '-o', '--output',
        help='输出文件（普通 HEX 格式）'
    )
    
    args = parser.parse_args()
    
    # 解析输入文件
    print(f"正在解析输入文件: {args.input_file}")
    data = parse_verilog_hex(args.input_file)
    print(f"解析完成，共 {len(data)} 个字")
    
    # 转换为普通 HEX 格式
    result = convert_to_simple_hex(data, args.output)
    print(f"转换完成，共 {len(result)} 行")
    
    if args.output:
        print(f"已写入输出文件: {args.output}")
    else:
        # 打印前 20 行和后 20 行
        print("\n前 20 行:")
        for i, line in enumerate(result[:20]):
            print(f"{i:4d} | {line}")
        
        if len(result) > 40:
            print("\n...")
            print(f"\n后 20 行:")
            for i, line in enumerate(result[-20:], start=len(result)-20):
                print(f"{i:4d} | {line}")


if __name__ == '__main__':
    main()
