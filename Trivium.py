class Trivium:
    def __init__(self):
        self.state = [0] * 288  # 288位内部状态

    def initialize(self, key, iv):
        """初始化Trivium的内部状态

        Args:
            key: 80位密钥(以比特列表形式)
            iv: 80位初始向量(以比特列表形式)
        """
        if len(key) != 80 or len(iv) != 80:
            raise ValueError("密钥和IV必须为80位")

        # 加载密钥(前80位)
        self.state[:80] = key

        # 加载IV(94-173位)
        self.state[93:173] = iv

        # 设置最后三位为1
        self.state[285] = self.state[286] = self.state[287] = 1

        # 进行4*288轮初始化
        for _ in range(4 * 288):
            self._update_state()

    def _update_state(self):
        """更新内部状态"""
        t1 = self.state[65] ^ self.state[92]
        t2 = self.state[161] ^ self.state[176]
        t3 = self.state[242] ^ self.state[287]

        t1 = t1 ^ (self.state[90] & self.state[91]) ^ self.state[170]
        t2 = t2 ^ (self.state[174] & self.state[175]) ^ self.state[263]
        t3 = t3 ^ (self.state[285] & self.state[286]) ^ self.state[68]

        # 移位操作
        self.state = [t3] + self.state[:92] + \
                     [t1] + self.state[93:176] + \
                     [t2] + self.state[177:287]

    def generate_keystream(self, length):
        """生成指定长度的密钥流

        Args:
            length: 需要生成的密钥流长度

        Returns:
            生成的密钥流(比特列表)
        """
        keystream = []

        for _ in range(length):
            # 计算输出位
            t1 = self.state[65] ^ self.state[92]
            t2 = self.state[161] ^ self.state[176]
            t3 = self.state[242] ^ self.state[287]

            # 生成密钥流位
            keystream_bit = t1 ^ t2 ^ t3
            keystream.append(keystream_bit)

            # 更新状态
            self._update_state()

        return keystream


def text_to_bits(text):
    """将文本转换为比特列表"""
    result = []
    for c in text.encode('utf-8'):
        bits = bin(c)[2:].zfill(8)
        result.extend([int(b) for b in bits])
    return result


def bits_to_text(bits):
    """将比特列表转换为文本"""
    # 确保比特数能被8整除
    if len(bits) % 8 != 0:
        raise ValueError("比特数必须是8的倍数")

    bytes_list = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]
        byte_val = int(''.join(str(b) for b in byte), 2)
        bytes_list.append(byte_val)

    return bytes(bytes_list).decode('utf-8')


class TriviumCipher:
    def __init__(self, key, iv):
        """
        初始化Trivium加密器

        Args:
            key: 80位密钥(比特列表)
            iv: 80位初始向量(比特列表)
        """
        self.key = key
        self.iv = iv

    def encrypt(self, plaintext):
        """
        加密文本

        Args:
            plaintext: 明文字符串

        Returns:
            密文比特列表
        """
        # 将明文转换为比特
        plaintext_bits = text_to_bits(plaintext)

        # 初始化Trivium
        cipher = Trivium()
        cipher.initialize(self.key, self.iv)

        # 生成相应长度的密钥流
        keystream = cipher.generate_keystream(len(plaintext_bits))

        # 加密：明文与密钥流异或
        ciphertext_bits = [p ^ k for p, k in zip(plaintext_bits, keystream)]

        return ciphertext_bits

    def decrypt(self, ciphertext_bits):
        """
        解密比特流

        Args:
            ciphertext_bits: 密文比特列表

        Returns:
            解密后的明文字符串
        """
        # 初始化Trivium
        cipher = Trivium()
        cipher.initialize(self.key, self.iv)

        # 生成相应长度的密钥流
        keystream = cipher.generate_keystream(len(ciphertext_bits))

        # 解密：密文与密钥流异或
        plaintext_bits = [c ^ k for c, k in zip(ciphertext_bits, keystream)]

        # 转换回文本
        return bits_to_text(plaintext_bits)


def test_trivium_encryption():
    # 测试示例
    # 生成示例密钥和IV（在实际应用中应该使用密码学安全的随机数生成器）
    key = [1] * 80  # 80位全1密钥
    iv = [0] * 80  # 80位全0初始向量

    # 创建加密器实例
    cipher = TriviumCipher(key, iv)

    # 测试文本
    original_text = "Hello, Trivium!"
    print(f"原始文本: {original_text}")

    # 加密
    encrypted_bits = cipher.encrypt(original_text)
    print(f"加密后(比特): {encrypted_bits[:32]}...")  # 只显示前32位
    encrypted_bits1=encrypted_bits
    encrypted_bits1.reverse()
    # 解密
    decrypted_text = cipher.decrypt(encrypted_bits1)
    print(f"解密后: {decrypted_text}")

    # 验证
    if original_text == decrypted_text:
        print("加密解密测试成功！")
    else:
        print("错误，密钥流不对")


if __name__ == "__main__":
    test_trivium_encryption()
