class Grain128a:
    def __init__(self):
        self.NFSR_SIZE = 128
        self.LFSR_SIZE = 128
        self.IV_SIZE = 96
        self.KEY_SIZE = 128

        # 寄存器状态
        self.NFSR = [0] * self.NFSR_SIZE
        self.LFSR = [0] * self.LFSR_SIZE

        # 认证模式标志
        self.auth_mode = False

    def init(self, key, iv, auth_mode=False):
        """
        初始化Grain-128a

        Args:
            key: 128位密钥
            iv: 96位初始化向量
            auth_mode: 是否启用认证模式
        """
        if len(key) != self.KEY_SIZE or len(iv) != self.IV_SIZE:
            raise ValueError("Invalid key or IV length")

        self.auth_mode = auth_mode

        # 初始化NFSR（使用密钥）
        self.NFSR = list(key)

        # 初始化LFSR（使用IV和填充）
        self.LFSR = list(iv) + [1] * (self.LFSR_SIZE - self.IV_SIZE)
        if auth_mode:
            self.LFSR[0] = 1  # IV0 设置为1表示认证模式

        # 初始化阶段
        for _ in range(256):
            self._clock()

    def _g(self, x):
        """非线性反馈函数g"""
        # 实现g(x)函数
        # 这里简化了实际的复杂布尔函数
        result = x[0] ^ x[3] ^ x[11] ^ x[13] ^ x[17] ^ x[18] ^ x[26] ^ x[27] ^ \
                 x[40] ^ x[48] ^ x[56] ^ x[59] ^ x[61] ^ x[65] ^ x[67] ^ x[68] ^ \
                 x[84] ^ x[91] ^ x[96] ^ x[3]
        return result

    def _f(self, x):
        """线性反馈函数f"""
        # 实现f(x)函数
        # 这里简化了实际的线性反馈函数
        result = x[0] ^ x[7] ^ x[38] ^ x[70] ^ x[81] ^ x[96]
        return result

    def _h(self):
        """预输出函数h"""
        # 实现h(x)函数
        # 从LFSR和NFSR中选择特定位进行组合
        x = [self.NFSR[12], self.LFSR[8], self.LFSR[13], self.LFSR[20],
             self.NFSR[95], self.LFSR[42], self.LFSR[60], self.LFSR[79]]

        result = x[0] ^ x[1] ^ x[2] ^ x[3] ^ (x[4] & x[5]) ^ (x[6] & x[7])
        return result

    def _clock(self):
        """时钟寄存器前进一步"""
        # 计算新的反馈值
        new_nfsr_bit = self._g(self.NFSR) ^ self.LFSR[-1]
        new_lfsr_bit = self._f(self.LFSR)

        # 移位寄存器
        self.NFSR = [new_nfsr_bit] + self.NFSR[:-1]
        self.LFSR = [new_lfsr_bit] + self.LFSR[:-1]

    def generate_keystream(self, length):
        """
        生成密钥流

        Args:
            length: 需要生成的密钥流长度

        Returns:
            生成的密钥流
        """
        keystream = []

        for _ in range(length):
            # 生成预输出位
            z = self._h() ^ self.NFSR[-1] ^ self.LFSR[-2]

            if self.auth_mode:
                # 认证模式下每两个预输出位只使用一个
                self._clock()
                self._clock()
                keystream.append(z)
            else:
                # 非认证模式下直接使用预输出位
                self._clock()
                keystream.append(z)

        return keystream

    def encrypt(self, plaintext, key, iv, auth_mode=False):
        """
        加密函数

        Args:
            plaintext: 明文比特列表
            key: 128位密钥
            iv: 96位IV
            auth_mode: 是否使用认证模式

        Returns:
            密文比特列表
        """
        self.init(key, iv, auth_mode)
        keystream = self.generate_keystream(len(plaintext))
        return [p ^ k for p, k in zip(plaintext, keystream)]

    def decrypt(self, ciphertext, key, iv, auth_mode=False):
        """
        解密函数

        Args:
            ciphertext: 密文比特列表
            key: 128位密钥
            iv: 96位IV
            auth_mode: 是否使用认证模式

        Returns:
            明文比特列表
        """
        # 解密过程与加密相同（流密码的特性）
        return self.encrypt(ciphertext, key, iv, auth_mode)


# 使用示例
def test_grain128a():
    # 创建测试数据
    key = [1] * 128  # 128位密钥
    iv = [0] * 96  # 96位IV
    plaintext = [1, 0, 1, 0, 1, 1, 0, 0]  # 测试明文

    # 创建Grain-128a实例
    cipher = Grain128a()

    # 加密
    ciphertext = cipher.encrypt(plaintext, key, iv)
    print("密文:", ciphertext)

    # 解密
    decrypted = cipher.decrypt(ciphertext, key, iv)
    print("解密后:", decrypted)

    # 验证
    print("验证成功:" if plaintext == decrypted else "验证失败")


if __name__ == "__main__":
    test_grain128a()