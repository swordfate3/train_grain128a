import numpy as np


class Trivium:
    def __init__(self):
        self.state = np.zeros(288, dtype=np.uint8)

    def initialize(self, key, iv):
        # 初始化状态
        self.state[:80] = key
        self.state[93:173] = iv
        self.state[285:288] = np.array([1, 1, 1])

        # 空转1152轮
        for _ in range(1152):
            self._update()

    def _update(self):
        t1 = self.state[65] ^ self.state[92]
        t2 = self.state[161] ^ self.state[176]
        t3 = self.state[242] ^ self.state[287]

        z = t1 ^ t2 ^ t3

        t1 = t1 ^ (self.state[90] & self.state[91]) ^ self.state[170]
        t2 = t2 ^ (self.state[174] & self.state[175]) ^ self.state[263]
        t3 = t3 ^ (self.state[285] & self.state[286]) ^ self.state[68]

        self.state = np.roll(self.state, -1)
        self.state[92] = t1
        self.state[176] = t2
        self.state[287] = t3

        return z

    def generate_keystream(self, length):
        return np.array([self._update() for _ in range(length)])


def text_to_bits(text):
    """将文本转换为二进制比特流"""
    bits = []
    for c in text.encode('utf-8'):
        for i in range(8):
            bits.append((c >> (7 - i)) & 1)
    return np.array(bits, dtype=np.uint8)


def bits_to_text(bits):
    """将二进制比特流转换回文本"""
    # 确保比特数是8的倍数
    bits = bits[:len(bits) - len(bits) % 8]
    bytes_data = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        bytes_data.append(byte)
    return bytes(bytes_data).decode('utf-8')


def encrypt(plaintext, key, iv):
    """加密函数"""
    # 将明文转换为比特流
    plainbits = text_to_bits(plaintext)

    # 初始化Trivium
    cipher = Trivium()
    cipher.initialize(key, iv)

    # 生成密钥流并进行异或运算
    keystream = cipher.generate_keystream(len(plainbits))
    cipherbits = plainbits ^ keystream

    return cipherbits


def decrypt(cipherbits, key, iv):
    """解密函数"""
    # 初始化Trivium
    cipher = Trivium()
    cipher.initialize(key, iv)

    # 生成相同的密钥流并进行异或运算
    keystream = cipher.generate_keystream(len(cipherbits))
    plainbits = cipherbits ^ keystream
    # plainbits=encrypt(cipherbits, key, iv)
    # 将比特流转换回文本
    return bits_to_text(plainbits)

def texttrivium():
    # 测试加密解密过程
    # 生成随机密钥和IV
    key = np.random.randint(0, 2, 80, dtype=np.uint8)
    iv = np.random.randint(0, 2, 80, dtype=np.uint8)

    # 测试文本
    original_text = "Hello, Trivium!"
    print("原始文本:", original_text)

    # 加密
    ciphertext = encrypt(original_text, key, iv)
    print("\n加密后的比特流:", ciphertext)

    # 解密
    decrypted_text = decrypt(ciphertext, key, iv)
    print("\n解密后的文本:", decrypted_text)

    # 验证
    print("\n加密解密是否成功:", original_text == decrypted_text)

if __name__ == '__main__':
    texttrivium()