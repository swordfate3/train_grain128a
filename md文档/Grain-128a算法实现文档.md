# Grain-128a 算法实现文档

## 目录
- [1. 算法概述](#1-算法概述)
- [2. 数据结构](#2-数据结构)
- [3. 核心功能实现](#3-核心功能实现)
- [4. 认证机制](#4-认证机制)
- [5. 训练数据生成](#5-训练数据生成)
- [6. 使用示例](#6-使用示例)

## 1. 算法概述

Grain-128a 是一个硬件导向的流密码，具有以下特点：
- 128位密钥
- 96位初始化向量（IV）
- 认证功能
- 基于LFSR和NFSR的设计

### 1.1 主要组件
- LFSR（线性反馈移位寄存器）：128位
- NFSR（非线性反馈移位寄存器）：128位
- 认证累加器：32位
- 认证移位寄存器：32位

## 2. 数据结构

### 2.1 GrainState 类
```python
class GrainState:
    def __init__(self):
        self.lfsr = np.zeros(128, dtype=np.uint8)  # 线性反馈移位寄存器
        self.nfsr = np.zeros(128, dtype=np.uint8)  # 非线性反馈移位寄存器
        self.auth_acc = np.zeros(32, dtype=np.uint8)  # 认证累加器
        self.auth_sr = np.zeros(32, dtype=np.uint8)   # 认证移位寄存器
```

### 2.2 GrainData 类
```python
class GrainData:
    def __init__(self, msg):
        self.keystream = np.zeros(STREAM_BYTES, dtype=np.uint8)
        self.macstream = np.zeros(STREAM_BYTES, dtype=np.uint8)
        self.message = np.zeros(STREAM_BYTES * 8, dtype=np.uint8)
        self.init_data(msg)
```

## 3. 核心功能实现

### 3.1 反馈函数

#### LFSR反馈函数
```python
def next_lfsr_fb(grain):
    return (grain.lfsr[96] ^ grain.lfsr[81] ^ grain.lfsr[70] ^
            grain.lfsr[38] ^ grain.lfsr[7] ^ grain.lfsr[0])
```

#### NFSR反馈函数
```python
def next_nfsr_fb(grain):
    return (grain.nfsr[96] ^ grain.lfsr[0] ^ grain.nfsr[0] ^
            (grain.nfsr[26] & grain.nfsr[56]) ^
            (grain.nfsr[91] & grain.nfsr[96]) ^
            (grain.nfsr[3] & grain.nfsr[67]) ^
            # ... 更多非线性项
            (grain.nfsr[70] & grain.nfsr[78] & grain.nfsr[82]))
```

### 3.2 输出函数
```python
def next_h(grain):
    return (grain.nfsr[12] & grain.lfsr[8]) ^ 
           (grain.lfsr[13] & grain.lfsr[20]) ^ 
           (grain.nfsr[95] & grain.lfsr[42]) ^ 
           (grain.lfsr[60] & grain.lfsr[79]) ^ 
           (grain.nfsr[12] & grain.nfsr[95] & grain.lfsr[94])
```

## 4. 认证机制

### 4.1 认证累加器更新
```python
def auth_accumulate(grain):
    """更新认证累加器"""
    for i in range(MAC_SIZE):
        grain.auth_acc[i] ^= grain.auth_sr[i]
```

### 4.2 认证移位寄存器操作
```python
def auth_shift(grain, fb_bit):
    """认证移位寄存器移位"""
    for i in range(MAC_SIZE-1, 0, -1):
        grain.auth_sr[i] = grain.auth_sr[i-1]
    grain.auth_sr[0] = fb_bit
```

## 5. 训练数据生成

### 5.1 主要生成函数
```python
def make_train_data(n_samples, cipher, diff, rounds, y=None):
    # 生成标签
    if y is None:
        y = np.frombuffer(urandom(n_samples), dtype=np.uint8) & 1
    
    # 生成密钥和IV
    keys = draw_keys(n_samples)
    iv0 = draw_nonces(n_samples)
    
    # 处理差分
    iv1 = iv0 ^ np.array(diff, dtype=np.uint8)[:, np.newaxis]
```

### 5.2 辅助函数
```python
def draw_keys(n_samples):
    return np.reshape(
        np.frombuffer(urandom(16 * n_samples), dtype=np.uint8),
        (16, n_samples)
    )

def draw_nonces(n_samples):
    return np.reshape(
        np.frombuffer(urandom(12 * n_samples), dtype=np.uint8),
        (12, n_samples)
    )
```

## 6. 使用示例

### 6.1 基本使用
```python
# 创建Grain实例
grain = GrainState()

# 初始化
key = urandom(16)  # 128位密钥
iv = urandom(12)   # 96位IV
init_grain(grain, key, iv)

# 生成密钥流
msg = np.zeros(50, dtype=np.uint8)
data = GrainData(msg)
keystream = generate_keystream(grain, data)
```

### 6.2 生成训练数据
```python
# 生成训练数据
n_train_samples = 1000
diff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0x1, 0, 0]  # 差分向量
rounds = 256
x, y = make_train_data(n_train_samples, None, diff, rounds)
```

## 注意事项

1. **安全考虑**
   - 确保密钥和IV的随机性
   - 正确实现认证机制
   - 保护状态更新的完整性

2. **性能优化**
   - 使用NumPy向量化操作
   - 避免不必要的循环
   - 合理管理内存使用

3. **调试建议**
   - 使用小规模测试数据验证功能
   - 检查状态更新的正确性
   - 验证认证机制的有效性