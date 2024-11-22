import time

import numpy as np

from train_grain128a import auth_mode


class Grain128a:
    """
    Grain-128a流密码的NumPy实现

    Grain-128a是一个轻量级流密码，主要包含：
    - 128位LFSR（线性反馈移位寄存器）
    - 128位NFSR（非线性反馈移位寄存器）
    - 认证累加器（用于认证模式）
    """

    def __init__(self,auth_mode):
        """
        初始化Grain-128a的内部状态

        使用numpy数组存储：
        - LFSR: 线性反馈移位寄存器，128位
        - NFSR: 非线性反馈移位寄存器，128位
        - auth_acc: 认证累加器，32位
        - auth_sr: 认证移位寄存器，64位
        """
        # 使用np.int8类型可以节省内存，因为我们只需要存储0和1
        self.auth_mode = auth_mode
        self.LFSR = np.zeros(128, dtype=np.int8)  # 线性反馈移位寄存器
        self.NFSR = np.zeros(128, dtype=np.int8)  # 非线性反馈移位寄存器
        self.auth_acc = np.zeros(32, dtype=np.int8)  # 认证累加器
        self.auth_sr = np.zeros(64, dtype=np.int8)  # 认证移位寄存器
    def init(self, key, iv):
        """
        初始化Grain-128a的状态

        参数:
        key: 128位密钥（numpy数组）
        iv: 96位初始化向量（numpy数组）
        auth_mode: 是否启用认证模式

        初始化过程:
        1. 将密钥加载到NFSR
        2. 将IV加载到LFSR的前96位
        3. LFSR的剩余位设置为1
        4. 如果是认证模式，设置特殊标志
        5. 进行320轮初始化时钟
        """
        # 复制密钥到NFSR
        self.NFSR = np.copy(key)

        # 加载IV并设置LFSR的初始状态
        self.LFSR[:96] = iv  # 前96位是IV
        self.LFSR[96:] = 1  # 剩余位设为1

        # 认证模式的特殊设置
        if self.auth_mode:
            self.LFSR[0] = 1

        # 进行256轮初始化时钟
        for _ in range(256):
            self._clock()

    def _clock(self):
        """
        时钟函数 - 更新LFSR和NFSR的状态

        该函数执行一次时钟周期：
        1. 计算LFSR的新反馈位
        2. 计算NFSR的新反馈位
        3. 对两个寄存器进行移位操作
        """
        # LFSR反馈多项式: f(x) = 1 + x^32 + x^47 + x^58 + x^90 + x^121 + x^128
        #  对应的更新函数: s[i+128] = s[i] + s[i+7] + s[i+38] + s[i+70] + s[i+81] + s[i+96]
        # bitwise_xor.reduce处理多个异或
        new_lfsr = np.bitwise_xor.reduce([
            self.LFSR[0],  # x^0
            self.LFSR[7],  # x^7
            self.LFSR[38],  # x^38
            self.LFSR[70],  # x^70
            self.LFSR[81],  # x^81
            self.LFSR[96]  # x^96
        ])

        # NFSR非线性反馈函数g(x)
        # 包含线性项和非线性项（与运算）
        g = np.bitwise_xor.reduce([
            self.NFSR[0],  # 线性项
            self.NFSR[26],
            self.NFSR[56],
            self.NFSR[91],
            self.NFSR[96],
            # 非线性项（二次项）
            self.NFSR[3] & self.NFSR[67],
            self.NFSR[11] & self.NFSR[13],
            self.NFSR[17] & self.NFSR[18],
            self.NFSR[27] & self.NFSR[59],
            self.NFSR[40] & self.NFSR[48],
            self.NFSR[61] & self.NFSR[65],
            self.NFSR[68] & self.NFSR[84],
            self.NFSR[88] & self.NFSR[92]&self.NFSR[93]&self.NFSR[95],
            self.NFSR[22]&self.NFSR[24]&self.NFSR[25],
            self.NFSR[70] & self.NFSR[78]&self.NFSR[82]

        ])

        # NFSR的新状态还要与LFSR的输出异或
        new_nfsr = np.bitwise_xor(g, self.LFSR[0])

        # 使用numpy的roll函数进行高效的移位操作
        self.LFSR = np.roll(self.LFSR, -1)  # 向左移位
        self.NFSR = np.roll(self.NFSR, -1)
        self.LFSR[-1] = new_lfsr  # 设置新的反馈位
        self.NFSR[-1] = new_nfsr

    def _h(self):
        """
        预输出函数h(x)

        这是一个非线性布尔函数，用于增加密码的复杂度
        使用LFSR和NFSR的特定位作为输入
        """
        # 选择特定位置的值
        b=self.NFSR
        s=self.LFSR
        # x = self.LFSR[[12, 95, 87, 33]]  # LFSR的选定位
        # y = self.NFSR[[12, 95, 87, 33]]  # NFSR的选定位
        #
        # # 计算非线性组合
        # return np.bitwise_xor.reduce([
        #     x[0] & x[1],  # LFSR位的与运算
        #     x[2] & x[3],
        #     y[0] & y[1],  # NFSR位的与运算
        #     y[2] & y[3],
        #     self.LFSR[12] & self.NFSR[95],  # 交叉项
        #     self.LFSR[95] & self.NFSR[12]
        # ])
        x = [0] * 9
        x[0] = b[12]
        x[1] = s[8]
        x[2] = s[13]
        x[3] = s[20]
        x[4] = b[95]
        x[5] = s[42]
        x[6] = s[60]
        x[7] = s[79]
        x[8] = s[94]
        return np.bitwise_xor.reduce([
            x[0] & x[1],  # LFSR位的与运算
            x[2] & x[3],
            x[4] & x[5],  # NFSR位的与运算
            x[6] & x[7],
            x[0]&x[4] & x[8],
        ])

    def _get_output(self):
        """
        预输出函数:
        y[i] = h(x) + s[i+93] + Σ b[i+j]

        其中 j ∈ A = {2, 15, 36, 45, 64, 73, 89}
        """
        h = self._h()

        # 线性加入的比特
        sum_bits = (self.LFSR[93] ^  # s[i+93]
                    self.NFSR[2] ^  # b[i+2]
                    self.NFSR[15] ^  # b[i+15]
                    self.NFSR[36] ^  # b[i+36]
                    self.NFSR[45] ^  # b[i+45]
                    self.NFSR[64] ^  # b[i+64]
                    self.NFSR[73] ^  # b[i+73]
                    self.NFSR[89])  # b[i+89]

        return h ^ sum_bits
    def get_keystream(self, length):
        """
        生成指定长度的密钥流

        参数:
        length: 需要生成的密钥流长度
        auth_mode: 是否在认证模式下运行

        返回:
        numpy数组形式的密钥流
        """
        keystream = np.zeros(length, dtype=np.int8)

        for i in range(length):
            # 计算输出位：结合多个NFSR位、一个LFSR位和h函数的输出
            # output = np.bitwise_xor.reduce([
            #     self.LFSR[93],
            #     self.NFSR[2],
            #     self.NFSR[15],
            #     self.NFSR[36],
            #     self.NFSR[45],
            #     self.NFSR[64],
            #     self.NFSR[73],
            #     self.NFSR[89],
            #     self._h()
            # ])
            output = self._get_output()

            # 认证模式的特殊处理
            if self.auth_mode:
                if i % 2 == 0:
                    keystream[i] = output
                else:
                    self._update_auth(output)
            else:
                keystream[i] = output

            self._clock()

        return keystream

    def _update_auth(self, auth_bit):
        """
        更新认证状态

        参数:
        auth_bit: 新的认证位

        用于认证模式，更新认证累加器和移位寄存器
        """
        # 更新认证移位寄存器
        self.auth_sr = np.roll(self.auth_sr, -1)
        self.auth_sr[-1] = auth_bit

        # 当移位寄存器满时更新累加器
        if len(self.auth_sr) == 64:
            # 使用前32位和后32位进行与运算，结果与累加器异或
            self.auth_acc ^= self.auth_sr[:32] & self.auth_sr[32:]
    def encrypt(self, plaintext, key, iv):
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
        self.init(key, iv)
        keystream = self.get_keystream(len(plaintext))
        return [p ^ k for p, k in zip(plaintext, keystream)]

    def decrypt(self, ciphertext, key, iv):
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
        return self.encrypt(ciphertext, key, iv)

#     攻击方式1：代数攻击
    def algebraic_attack_analysis(self):
        """
        代数攻击分析模拟
        1. NFSR的代数度为2
        2. h(x)函数的代数度为3
        3. 输出函数y的代数度为3
        """

        class AlgebraicAnalysis:
            def __init__(self):
                self.nfsr_degree = 2  # NFSR反馈函数的代数度
                self.h_degree = 3  # h(x)函数的代数度
                self.y_degree = 3  # 输出函数y的代数度

            def estimate_equations(self, rounds):
                """估算方程系统的复杂度"""
                variables = 256  # 初始状态变量数(128 LFSR + 128 NFSR)
                equations = rounds  # 每轮产生一个方程
                degree = self.y_degree  # 方程的最大代数度

                # 计算求解复杂度
                complexity = 2 ** min(variables, equations * degree)
                return complexity

        return AlgebraicAnalysis()
    # 2相关攻击
    def correlation_attack_analysis(self):
        """
        相关攻击分析模拟
        1. 分析LFSR和输出之间的相关性
        2. 评估非线性布尔函数的相关性免疫性
        """

        class CorrelationAnalysis:
            def analyze_correlation(self):
                # LFSR和输出位之间的相关性分析
                linear_terms = 8  # 线性项数量
                nonlinear_terms = self._count_nonlinear_terms()

                # 计算相关性系数
                correlation = 2 ** (-nonlinear_terms / 2)
                return correlation

            def _count_nonlinear_terms(self):
                # h(x)函数中的非线性项
                h_nonlinear = 4  # 二次项数量
                h_cubic = 1  # 三次项数量

                # NFSR反馈中的非线性项
                nfsr_nonlinear = 7  # 二次项数量
                nfsr_cubic = 3  # 三次项数量

                return h_nonlinear + h_cubic + nfsr_nonlinear + nfsr_cubic

        return CorrelationAnalysis()
    # 3差分攻击
    def differential_attack_analysis(self):
        """
        差分攻击分析模拟
        1. 评估状态差分传播特性
        2. 分析输出函数的差分特性
        """

        class DifferentialAnalysis:
            def analyze_differential_probability(self):
                # 分析状态更新函数的差分特性
                nfsr_active_bits = self._count_active_bits_nfsr()
                lfsr_active_bits = self._count_active_bits_lfsr()

                # 计算差分传播概率
                prob = 2 ** -(nfsr_active_bits + lfsr_active_bits)
                return prob

            def _count_active_bits_nfsr(self):
                # NFSR中活跃比特数
                return 128  # 完整状态大小

            def _count_active_bits_lfsr(self):
                # LFSR中活跃比特数
                return 128  # 完整状态大小

        return DifferentialAnalysis()
    # 4猜测和确定攻击
    def guess_determine_attack_analysis(self):
        """
        猜测和确定攻击分析模拟
        1. 评估状态恢复攻击的复杂度
        2. 分析密钥恢复攻击的可行性
        """

        class GuessAndDetermineAnalysis:
            def estimate_attack_complexity(self):
                # 状态变量总数
                total_state_bits = 256  # 128(LFSR) + 128(NFSR)

                # 需要猜测的比特数
                guess_bits = 128  # 通常需要猜测一半的状态

                # 验证复杂度
                verification_complexity = 2 ** guess_bits

                return {
                    'total_state': total_state_bits,
                    'guess_bits': guess_bits,
                    'complexity': verification_complexity
                }

        return GuessAndDetermineAnalysis()
#     5时间-记忆-数据权衡攻击
    def tmd_tradeoff_analysis(self):
        """
        时间-记忆-数据权衡攻击分析模拟
        分析TMD权衡攻击的复杂度
        """

        class TMDAnalysis:
            def analyze_tmd_complexity(self):
                state_size = 256  # 总状态大小

                # 计算TMD权衡攻击的复杂度参数
                time_complexity = 2 ** (state_size / 2)
                memory_complexity = 2 ** (state_size / 2)
                data_complexity = 2 ** (state_size / 2)

                return {
                    'time': time_complexity,
                    'memory': memory_complexity,
                    'data': data_complexity,
                    'total_complexity': 2 ** state_size
                }

        return TMDAnalysis()


# def test_grain128a():
#     """
#     测试Grain-128a的功能并生成可视化结果
#     """
#     # 创建密码实例
#     cipher = Grain128a(auth_mode=True)
#
#     # 生成随机测试数据
#     key = np.random.randint(0, 2, 128, dtype=np.int8)
#     iv = np.random.randint(0, 2, 96, dtype=np.int8)
#     plaintext = [1, 0, 1, 0, 1, 1, 0, 0]  # 测试明文
#     # 初始化密码
#     cipher.init(key, iv)
#
#     # 生成测试密钥流
#     keystream = cipher.get_keystream(128)
#
#     # 打印部分结果
#     print("Key (first 32 bits):", key[:32])
#     print("IV (first 32 bits):", iv[:32])
#     print("Keystream (first 32 bits):", keystream[:32])
#
#     return key, iv, keystream

# 使用示例
def test_grain128a1():
    # 创建测试数据
    key = np.random.randint(0, 2, 128, dtype=np.int8)
    iv = np.random.randint(0, 2, 96, dtype=np.int8)
    plaintext = [1, 0, 1, 0, 1, 1, 0, 0]  # 测试明文

    # 创建Grain-128a实例
    cipher = Grain128a(auth_mode=False)

    # 加密
    ciphertext = cipher.encrypt(plaintext, key, iv)
    print("密文:", ciphertext)

    # 解密
    decrypted = cipher.decrypt(ciphertext, key, iv)
    print("解密后:", decrypted)

    # 验证
    print("验证成功:" if plaintext == decrypted else "验证失败")

# 运行测试并创建可视化
# key, iv, keystream = test_grain128a()
if __name__ == '__main__':
    time1=time.time()
    test_grain128a1()
    time2=time.time()
    print("时间差", time2-time1)
# 创建可视化图表
# import matplotlib.pyplot as plt
#
# # 创建三个子图来显示密钥、IV和密钥流
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
#
# # 绘制密钥分布
# ax1.plot(key[:32], 'r-o', label='Key')
# ax1.set_title('Key Distribution (First 32 bits)')
# ax1.grid(True)
# ax1.legend()
# ax1.set_ylim(-0.2, 1.2)
#
# # 绘制IV分布
# ax2.plot(iv[:32], 'g-o', label='IV')
# ax2.set_title('IV Distribution (First 32 bits)')
# ax2.grid(True)
# ax2.legend()
# ax2.set_ylim(-0.2, 1.2)
#
# # 绘制密钥流分布
# ax3.plot(keystream[:32], 'b-o', label='Keystream')
# ax3.set_title('Keystream Distribution (First 32 bits)')
# ax3.grid(True)
# ax3.legend()
# ax3.set_ylim(-0.2, 1.2)
#
# plt.tight_layout()
# plt.show()
#
# # 计算并显示统计信息
# print("\nStatistical Analysis:")
# print(f"Keystream ones ratio: {np.mean(keystream):.3f}")
# print(f"Key ones ratio: {np.mean(key):.3f}")
# print(f"IV ones ratio: {np.mean(iv):.3f}")
#
# # 计算并绘制自相关性
# autocorr = np.correlate(keystream - np.mean(keystream),
#                         keystream - np.mean(keystream),
#                         mode='full') / (len(keystream) * np.var(keystream))
# autocorr = autocorr[len(autocorr) // 2:]
#
# plt.figure(figsize=(10, 4))
# plt.plot(autocorr[:32], 'b-o')
# plt.title('Keystream Autocorrelation (First 32 lags)')
# plt.xlabel('Lag')
# plt.ylabel('Autocorrelation')
# plt.grid(True)
# plt.show()