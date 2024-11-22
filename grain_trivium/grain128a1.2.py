class Grain128a:
    """Grain-128a流密码的核心实现"""

    def __init__(self):
        # 基本参数
        self.LFSR_SIZE = 128
        self.NFSR_SIZE = 128
        self.KEY_SIZE = 128
        self.IV_SIZE = 96

        # 状态寄存器
        self.lfsr = [0] * self.LFSR_SIZE  # 线性反馈移位寄存器
        self.nfsr = [0] * self.NFSR_SIZE  # 非线性反馈移位寄存器

        # 认证模式相关
        self.auth_mode = False
        self.accumulator = 0
    #     2.LFSR实现
    def lfsr_feedback(self):
        """LFSR反馈函数实现"""
        # f(x) = x^128 + x^96 + x^81 + x^70 + x^38 + x^7 + 1
        return (self.lfsr[0] ^ self.lfsr[7] ^ self.lfsr[38] ^
                self.lfsr[70] ^ self.lfsr[81] ^ self.lfsr[96])

    def update_lfsr(self):
        """更新LFSR状态"""
        feedback = self.lfsr_feedback()
        # 移位操作
        for i in range(0, self.LFSR_SIZE - 1):
            self.lfsr[i] = self.lfsr[i + 1]
        self.lfsr[self.LFSR_SIZE - 1] = feedback
    # 3 NFSR实现
    def nfsr_feedback(self):
        """NFSR反馈函数实现"""
        # g(x)函数实现
        b = self.nfsr  # 简化表示

        return (b[0] ^ b[26] ^ b[56] ^ b[91] ^ b[96] ^
                b[3] & b[67] ^ b[11] & b[13] ^ b[17] & b[18] ^
                b[27] & b[59] ^ b[40] & b[48] ^ b[61] & b[65] ^
                b[68] & b[84] ^ b[22] & b[24] & b[25] ^
                b[70] & b[78] & b[82] ^ b[88] & b[92] & b[93] & b[95])

    def update_nfsr(self, lfsr_out):
        """更新NFSR状态"""
        feedback = self.nfsr_feedback() ^ lfsr_out
        # 移位操作
        for i in range(0, self.NFSR_SIZE - 1):
            self.nfsr[i] = self.nfsr[i + 1]
        self.nfsr[self.NFSR_SIZE - 1] = feedback
    # 4 输出函数实现
    def h_function(self):
        """h(x)函数实现"""
        # 选择的比特位
        x = [self.nfsr[12], self.lfsr[8], self.lfsr[13],
             self.lfsr[20], self.nfsr[95], self.lfsr[42],
             self.lfsr[60], self.lfsr[79], self.nfsr[94]]

        return (x[0] & x[1] ^ x[2] & x[3] ^ x[4] & x[5] ^
                x[6] & x[7] ^ x[0] & x[4] & x[8])

    def output_function(self):
        """输出函数实现"""
        # 选择的LFSR位
        s = [self.lfsr[93], self.lfsr[15], self.lfsr[46],
             self.lfsr[50], self.lfsr[64], self.lfsr[71],
             self.lfsr[78]]

        # 选择的NFSR位
        b = [self.nfsr[12], self.nfsr[95]]

        # 计算输出
        return (s[0] ^ s[1] ^ s[2] ^ s[3] ^ s[4] ^ s[5] ^
                s[6] ^ b[0] ^ b[1] ^ self.h_function())
    # 5初始化实现
    def initialize(self, key, iv):
        """初始化过程实现"""
        # 验证输入
        if len(key) != self.KEY_SIZE // 8:
            raise ValueError("Invalid key size")
        if len(iv) != self.IV_SIZE // 8:
            raise ValueError("Invalid IV size")

        # 转换为比特列表
        key_bits = self._to_bits(key)
        iv_bits = self._to_bits(iv)

        # 加载NFSR（密钥）
        self.nfsr = key_bits

        # 加载LFSR（IV和填充）
        self.lfsr = iv_bits + [1] * (self.LFSR_SIZE - self.IV_SIZE)

        # 初始化轮数
        for _ in range(256):
            output = self.output_function()
            lfsr_out = self.lfsr_feedback()
            nfsr_out = self.nfsr_feedback()

            # 反馈输出到状态更新
            self.update_lfsr()
            self.update_nfsr(lfsr_out ^ output)

        self.initialized = True
    # 6认证模式实现
    def auth_update(self, message_bit, keystream_bit):
        """更新认证累加器"""
        if self.auth_mode:
            self.auth_bits.append(message_bit)
            self.keystream_bits.append(keystream_bit)

            if len(self.auth_bits) == 32:
                # 转换为32位整数
                message_block = self._bits_to_int(self.auth_bits)
                keystream_block = self._bits_to_int(self.keystream_bits)

                # 更新累加器
                self.accumulator = (self.accumulator +
                                    message_block +
                                    keystream_block) % (2 ** 32)

                # 清空缓存
                self.auth_bits = []
                self.keystream_bits = []

    def generate_tag(self):
        """生成认证标签"""
        if not self.auth_mode:
            return None

        # 继续运行64个时钟周期
        for _ in range(64):
            self.generate_keystream(1)

        return self.accumulator.to_bytes(4, 'big')
    # 7辅助函数
    def _to_bits(self, data):
        """字节转比特列表"""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return bits

    def _to_bytes(self, bits):
        """比特列表转字节"""
        bytes_list = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | bits[i + j]
            bytes_list.append(byte)
        return bytes(bytes_list)

    def _bits_to_int(self, bits):
        """比特列表转整数"""
        result = 0
        for bit in bits:
            result = (result << 1) | bit
        return result
# 7使用测试
def demonstrate_usage():
    """演示Grain-128a的使用流程"""
    # 创建实例
    cipher = Grain128a()

    # 示例密钥和IV
    key = bytes([1] * 16)  # 128位密钥
    iv = bytes([2] * 12)  # 96位IV

    # 初始化
    cipher.initialize(key, iv)

    # 加密示例
    plaintext = b"Hello, World!"
    ciphertext = cipher.encrypt(plaintext)

    # 解密示例
    decrypted = cipher.decrypt(ciphertext)

    # 认证模式示例
    cipher.auth_mode = True
    cipher.initialize(key, iv)

    # 加密并生成标签
    ciphertext, tag = cipher.encrypt(plaintext)

    # 验证并解密
    decrypted = cipher.decrypt(ciphertext, tag)
