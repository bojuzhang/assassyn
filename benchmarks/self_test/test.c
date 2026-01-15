// Test Load instructions: LB, LH, LW, LBU, LHU
// 测试各种Load指令

int main() {
    // 创建测试数据: 0x12345678 存储在内存中
    // 小端序: 地址+0=0x78, +1=0x56, +2=0x34, +3=0x12
    int data = 0x12345678;
    
    // 获取数据的地址
    unsigned char *ptr = (unsigned char *)&data;
    
    int result = 0;
    
    // 测试 LBU (Load Byte Unsigned) - 读取最低字节 0x78
    unsigned char byte0 = ptr[0];  // 应该是 0x78 = 120
    if (byte0 == 0x78) {
        result += 1;  // +1
    }
    
    // 测试 LBU - 读取第二个字节 0x56
    unsigned char byte1 = ptr[1];  // 应该是 0x56 = 86
    if (byte1 == 0x56) {
        result += 2;  // +2
    }
    
    // 测试 LHU (Load Halfword Unsigned) - 读取低半字 0x5678
    unsigned short *hptr = (unsigned short *)&data;
    unsigned short half0 = hptr[0];  // 应该是 0x5678 = 22136
    if (half0 == 0x5678) {
        result += 4;  // +4
    }
    
    // 测试 LW (Load Word) - 读取完整字
    int word = data;  // 应该是 0x12345678
    if (word == 0x12345678) {
        result += 8;  // +8
    }
    
    // 如果所有测试通过, result = 1 + 2 + 4 + 8 = 15
    return result;
}
