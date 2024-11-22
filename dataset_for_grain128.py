from os import urandom
import numpy as np

# Constants
STREAM_BYTES = 32
MSG_BYTES = 0
init_rounds = 0
auth_mode = 0

# Grain state structure
class GrainState:
    def __init__(self):
        self.lfsr = np.zeros(128, dtype=np.uint8)
        self.nfsr = np.zeros(128, dtype=np.uint8)
        self.auth_acc = np.zeros(32, dtype=np.uint8)
        self.auth_sr = np.zeros(32, dtype=np.uint8)

# Grain data structure
class GrainData:
    def __init__(self, msg):
        self.keystream = np.zeros(STREAM_BYTES, dtype=np.uint8)
        self.macstream = np.zeros(STREAM_BYTES, dtype=np.uint8)
        self.message = np.zeros(STREAM_BYTES * 8, dtype=np.uint8)
        self.init_data(msg)

    def init_data(self, msg):
        for i in range(len(msg)):
            self.message[i] = msg[i]
        self.message[len(msg)] = 1  # always pad data with a 1

def init_grain(grain, key, iv):
    global auth_mode
    # Initialize LFSR with IV
    for i in range(12):
        for j in range(8):
            grain.lfsr[8 * i + j] = (iv[i] >> (7 - j)) & 1
    grain.lfsr[96:127] = 1
    grain.lfsr[127] = 0
    if grain.lfsr[0] == 1:
        auth_mode = 1

    # Initialize NFSR with key
    for i in range(16):
        for j in range(8):
            grain.nfsr[8 * i + j] = (key[i] >> (7 - j)) & 1

    # Initialize authentication accumulator and shift register
    grain.auth_acc.fill(0)
    grain.auth_sr.fill(0)

def next_lfsr_fb(grain):
    return grain.lfsr[96] ^ grain.lfsr[81] ^ grain.lfsr[70] ^ grain.lfsr[38] ^ grain.lfsr[7] ^ grain.lfsr[0]

def next_nfsr_fb(grain):
    return grain.nfsr[96] ^ grain.nfsr[91] ^ grain.nfsr[56] ^ grain.nfsr[26] ^ grain.nfsr[0] ^ (
                grain.nfsr[84] & grain.nfsr[68]) ^ \
        (grain.nfsr[67] & grain.nfsr[3]) ^ (grain.nfsr[65] & grain.nfsr[61]) ^ (grain.nfsr[59] & grain.nfsr[27]) ^ \
        (grain.nfsr[48] & grain.nfsr[40]) ^ (grain.nfsr[18] & grain.nfsr[17]) ^ (grain.nfsr[13] & grain.nfsr[11]) ^ \
        (grain.nfsr[82] & grain.nfsr[78] & grain.nfsr[70]) ^ (grain.nfsr[25] & grain.nfsr[24] & grain.nfsr[22]) ^ \
        (grain.nfsr[95] & grain.nfsr[93] & grain.nfsr[92] & grain.nfsr[88])

# def output_function(self, lfsr, nfsr):
def next_h(grain):
    return (grain.nfsr[12] & grain.lfsr[8]) ^ (grain.lfsr[13] & grain.lfsr[20]) ^ (
                grain.nfsr[95] & grain.lfsr[42]) ^ \
        (grain.lfsr[60] & grain.lfsr[79]) ^ (grain.nfsr[12] & grain.nfsr[95] & grain.lfsr[94])

def shift(fsr, fb):
    out = fsr[0]
    fsr[:-1] = fsr[1:]
    fsr[-1] = fb
    return out

def auth_shift(sr, fb):
    sr[:-1] = sr[1:]
    sr[-1] = fb

def accumulate(grain):
    grain.auth_acc ^= grain.auth_sr

def next_z(grain):
    lfsr_fb = next_lfsr_fb(grain)
    nfsr_fb = next_nfsr_fb(grain)
    h_out = next_h(grain)

    A = [2, 15, 36, 45, 64, 73, 89]
    nfsr_tmp = 0
    for a in A:
        nfsr_tmp ^= grain.nfsr[a]

    y = h_out ^ grain.lfsr[93] ^ nfsr_tmp

    if init_rounds:
        lfsr_out = shift(grain.lfsr, lfsr_fb ^ y)
        shift(grain.nfsr, nfsr_fb ^ lfsr_out ^ y)
    else:
        lfsr_out = shift(grain.lfsr, lfsr_fb)
        shift(grain.nfsr, nfsr_fb ^ lfsr_out)

    return y


def print_stream(stream, byte_size):
    for i in range(byte_size):
        yi = 0
        for j in range(8):
            yi = (yi << 1) ^ stream[i * 8 + j]
        print(f"{yi:02x}", end=" ")
    print()

def generate_keystream(grain, data):
    if auth_mode:
        # Load authentication accumulator and shift register
        for i in range(32):
            grain.auth_acc[i] = next_z(grain)
        for i in range(32):
            grain.auth_sr[i] = next_z(grain)

        print("Accumulator: ", end="")
        print_stream(grain.auth_acc, 4)

        print("Shift register: ", end="")
        print_stream(grain.auth_sr, 4)

        ks = np.zeros(STREAM_BYTES * 8, dtype=np.uint8)
        ms = np.zeros(STREAM_BYTES * 8, dtype=np.uint8)
        pre = np.zeros(2 * STREAM_BYTES * 8, dtype=np.uint8)

        ks_cnt, ms_cnt, pre_cnt = 0, 0, 0

        for i in range(STREAM_BYTES):
            for j in range(16):
                z_next = next_z(grain)
                if j % 2 == 0:
                    ks[ks_cnt] = z_next
                    ks_cnt += 1
                else:
                    ms[ms_cnt] = z_next
                    if data.message[ms_cnt] == 1:
                        accumulate(grain)
                    auth_shift(grain.auth_sr, z_next)
                    ms_cnt += 1
                pre[pre_cnt] = z_next
                pre_cnt += 1

        print("Pre-output: ", end="")
        print_stream(pre, 2 * STREAM_BYTES)

        print("Keystream: ", end="")
        print_stream(ks, STREAM_BYTES)

        print("Macstream: ", end="")
        print_stream(ms, STREAM_BYTES)

        print("Accumulator: ", end="")
        print_stream(grain.auth_acc, 4)

    else:
        ks = np.zeros(STREAM_BYTES * 8, dtype=np.uint8)
        for i in range(STREAM_BYTES * 8):
            ks[i] = next_z(grain)
    return ks

def convert_to_binary(arr, n_words, word_size) -> np.ndarray:
    sample_len = 2 * n_words * word_size
    print("", )
    n_samples = len(arr[0])
    x = np.zeros((sample_len, n_samples), dtype=np.uint8)
    for i in range(sample_len):
        x[i] = arr[i] & 1
    x = x.transpose()
    return x

def preprocess_samples(ks0, ks1, cipher) -> np.ndarray:
    grain = convert_to_binary(np.concatenate((ks0, ks1), axis=0), 32, 4)
    return grain

def make_train_data(
        n_samples, cipher, diff, rounds, y=None
):
    # # 设置 NumPy 打印选项，确保打印出所有元素,每行元素不换行
    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # generate labels
    if y is None:
        y = np.frombuffer(urandom(n_samples), dtype=np.uint8) & 1
    elif y == 0 or y == 1:
        y = np.array([y for _ in range(n_samples)], dtype=np.uint8)
    # draw keys and plaintexts , 目前 keys 和 iv0 都是 n_sample 列 （正常情况）
    keys = draw_keys(n_samples)
    iv0 = draw_nonces(n_samples)

    # 下面的代码将 keys 和 iv0 都转为 n_sample 行了（为了按行遍历 iv0 ，将其与密钥 keys 按行共同存入 grain 内部状态中）
    keys = keys.transpose()
    iv0 = iv0.transpose()
    grain0 = []
    grain1 = []
    # 初始化内部状态：遍历 key 和 nonce 数组，生成 grain 数组，一个是 grain_iv0 二维数组，一个是 grain_iv1 二维数组
    for key_row, iv0_row in zip(keys, iv0):
        grain = GrainState()  # 每次都创建一个新的 GrainState
        init_grain(grain, key_row, iv0_row)  # 将 key_row 和 iv_row 传入 init_grain 函数
        global init_rounds
        init_rounds = 1
        for i in range(rounds):
            # print_state(grain)
            next_z(grain)
            # print_state(grain)
        init_rounds = 0  # 生成密钥流阶段了
        # 假设 init_grain 函数返回一个结果，存入 results 列表
        grain0.append(grain)
    # keys = keys.transpose()

    # 下面的代码又将 keys 和 iv0 都转为 n_sample 列了（正常，为了拿 iv0 和 diff 异或，生成 iv1）
    iv0 = iv0.transpose()
    iv1 = iv0 ^ np.array(diff, dtype=np.uint8)[:, np.newaxis]

    # 下面的代码又将 iv1 转为 n_sample 行了（为了按行遍历 iv1 ，将其与密钥 keys 按行共同存入 grain 内部状态中）
    # iv1 = iv1.transpose()

    num_rand_samples = np.sum(y == 0)
    print(num_rand_samples)
    iv1[:, y == 0] = draw_random_nonces(num_rand_samples)
    iv1 = iv1.transpose()
    # 初始化内部状态：遍历 key 和 nonce 数组，生成 grain 数组，一个是 grain_iv0 二维数组，一个是 grain_iv1 二维数组
    for key_row, iv1_row in zip(keys, iv1):
        grain = GrainState()  # 每次都创建一个新的 GrainState
        init_grain(grain, key_row, iv1_row)  # 将 key_row 和 iv_row 传入 init_grain 函数
        # global init_rounds
        init_rounds = 1
        for i in range(rounds):
            #print_state(grain)
            next_z(grain)
            # print_state(grain)
        init_rounds = 0 #生成密钥流阶段了
        # 假设 init_grain 函数返回一个结果，存入 results 列表
        grain1.append(grain)

    # 下面的代码又将 iv1 转为 n_sample 列了（正常，为了给 iv1 生成相应的标签）
    # iv1 = iv1.transpose()
    # # replace plaintexts in pt1 with random ones if label is 0
    # num_rand_samples = np.sum(y == 0)
    # iv1[:, y == 0] = draw_random_nonces(num_rand_samples)

    # 生成密钥流 grain_iv0 grain_iv1
    # 遍历 grain0 和 grain1 矩阵，逐个生成 ks0 和 ks1 矩阵
    # 假设 ks0 是一个二维数组，大小为 (n_samples, keystream_length)
    # msg = np.zeros(50, dtype=np.uint8)
    # data = GrainData(msg)
    # ks0 = np.zeros((n_samples, 256), dtype=np.uint8)
    # ks1 = np.zeros((n_samples, 256), dtype=np.uint8)
    # # print("grain0\n")
    # for i, grain in enumerate(grain0):
    #     # print(f"grain0[{i}]:\n", grain.lfsr, grain.nfsr)  # 按行打印 grain0
    #     ks = generate_keystream(grain, data)  # 获取 keystream
    #     ks0[i, :] = ks  # 将 keystream 存入 ks0 对应位置
    # # print("grain0 ks:", ks)

    # # print("grain1\n")
    # for i, grain in enumerate(grain1):
    #     # print(f"grain1[{i}]:\n", grain.lfsr, grain.nfsr)  # 按行打印 grain0
    #     ks = generate_keystream(grain, data)  # 获取 keystream
    #     ks1[i, :] = ks  # 将 keystream 存入 ks1 对应位置
    # # print("grain1 ks:", ks)

    # ks0_arr = np.zeros((n_samples, 256), dtype=np.uint8)
    ks0_arr = np.zeros((n_samples, 128), dtype=np.uint8)
    # ks1_arr = np.zeros((n_samples, 256), dtype=np.uint8)
    ks1_arr = np.zeros((n_samples, 128), dtype=np.uint8)
    for i, grain in enumerate(grain0):
        # 将lfsr和nfsr拼接成一维数组
        # lfsr_nfsr_concat = np.concatenate((grain.lfsr, grain.nfsr))
        # 将拼接后的数组存储在ks0的对应行
        # ks0_arr[i, :] = lfsr_nfsr_concat  # 转换为二维数组的第i行
        ks0_arr[i, :] = grain.nfsr  # 转换为二维数组的第i行
    # print("ks0_arr:\n", ks0_arr)
    for i, grain in enumerate(grain1):
        # lfsr_nfsr_concat = np.concatenate((grain.lfsr, grain.nfsr))
        # ks1_arr[i, :] = lfsr_nfsr_concat  # 转换为二维数组的第i行
        ks1_arr[i, :] = grain.nfsr
    # print("ks1_arr:\n", ks1_arr)

    # 下面的代码又将 ks0_arr 和 ks1_arr 都转为 n_sample 列了（正常，为了拿 ks0_arr 和 ks1_arr 合并，生成 训练数据）
    ks0_arr = ks0_arr.transpose()
    ks1_arr = ks1_arr.transpose()
    x = preprocess_samples(ks0_arr, ks1_arr, cipher)
    return x, y

def draw_keys(n_samples):
    return np.reshape(
        np.frombuffer(urandom(16 * n_samples), dtype=np.uint8),
        (16, n_samples)
    )

def draw_nonces(n_samples):
    nonces = np.reshape(
        np.frombuffer(urandom(12 * n_samples), dtype=np.uint8),
        (12, n_samples)
    )
    # 创建一个可写的副本
    nonces = np.copy(nonces)
    # 设置 nonces 为可写
    nonces.setflags(write=True)
    # 将每个 nonce 的第一个字节的最高位（第一个比特）设置为 0
    nonces[0] &= 0x7F # 0x7F 是二进制 0111 1111，按位与运算会将最高位变为 0
    return nonces

def draw_random_nonces(n_samples):
    random_nonces = np.reshape(np.frombuffer(urandom(12 * n_samples), dtype=np.uint8), (12, n_samples))
    # random_nonces = random_nonces.transpose()
    # print("Here shape of random_nonces:", random_nonces.shape)
    return random_nonces
    # return draw_nonces(n_samples)

import time
import sys
if __name__ == "__main__":
    grain128a = GrainState()
    n_train_samples = 1000
    # n_val_samples = 1
    in_diff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0x1, 0, 0]
    rounds = int(sys.argv[1])
    fx = open('data.txt','a')
    fy = open('label.txt','a')
    t1=time.time()
    x,y = make_train_data(n_train_samples,None,in_diff,rounds)
    np.savetxt(fx,x,fmt="%d")
    np.savetxt(fy,y,fmt="%d")
    t2=time.time()
    print(t2-t1)
    fx.close()
    fy.close()