// test_div_overflow/test.c
// 测试除法器 2D 溢出 bug
//
// Bug 描述: 当 D >= 0x40000000 时，two_D = D << 1 会溢出 Int(32)
// 导致 two_D 被错误解释为负数，商数字选择逻辑出错
//
// 例如: D = 0x40000000
//       two_D = 0x80000000 (应该是正数 2147483648，但被解释为 -2147483648)

// 无符号除法函数
unsigned int my_divu(unsigned int a, unsigned int b) {
    return a / b;
}

int main() {
    unsigned int a, b, result;

    a = 0x80000001;
    b = 0x40000000;
    result = my_divu(a, b);
    // 正确结果: 2
    
    // 最后一个测试的结果放在 a0 返回
    // 如果实现正确，返回 2
    // 如果有 bug，返回其他值
    
    return result;
}
