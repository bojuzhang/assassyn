// test_mulh_bug/test.c
// 测试 MULH/MULHSU 指令的符号扩展bug
// 
// Bug描述: 当前实现使用零扩展而非符号扩展
// a_64 = a.bitcast(UInt(64))  // 错误：零扩展
// 应该是: a_64 = concat(a_high, a)  // 正确：符号扩展
//
// 预期行为 vs 错误行为:
// (-1) * (-1) = 1
//   正确: 0x0000000000000001, 高32位 = 0x00000000
//   错误: 0xFFFFFFFF * 0xFFFFFFFF = 0xFFFFFFFE00000001, 高32位 = 0xFFFFFFFE

// 使用64位乘法获取高32位（编译器会生成MULH指令）
int mulh(int a, int b) {
    long long result = (long long)a * (long long)b;
    return (int)(result >> 32);
}

int main() {
    int a, b;
    int result_mulh;
    
    // 关键测试: MULH(-1, -1)
    // (-1) * (-1) = 1
    // 正确结果: 高32位 = 0
    // 错误结果: 高32位 = 0xFFFFFFFE (因为零扩展导致)
    a = -1;  // 0xFFFFFFFF
    b = -1;  // 0xFFFFFFFF
    
    // MULH: 高32位 (有符号*有符号)
    result_mulh = mulh(a, b);
    
    // 结果会在 a0 中返回
    // 正确结果: 0
    // 错误结果: 0xFFFFFFFE (证明bug存在)
    
    return result_mulh;
}
