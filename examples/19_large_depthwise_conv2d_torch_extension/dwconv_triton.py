import torch
import triton
import triton.language as tl


# 针对特定卷积核尺寸和硬件优化的版本
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 4, 'BLOCK_C': 32}),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 8, 'BLOCK_C': 32}),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 4, 'BLOCK_C': 32}),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8, 'BLOCK_C': 32}),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 8, 'BLOCK_C': 64}),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_C': 32}),
    ],
    key=['batch', 'height', 'width', 'channels', 'kernel_h', 'kernel_w'],
)
@triton.jit
def _depthwise_conv_kernel_optimized(
    input_ptr, weight_ptr, output_ptr,
    batch, height, width, channels,
    out_height, out_width,
    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
    KH_BLOCK=4, KW_BLOCK=4
):
    # 维度分解
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_batch_c = tl.program_id(2)
    
    # 分解批次和通道维度
    pid_batch = pid_batch_c // tl.cdiv(channels, BLOCK_C)
    pid_c = pid_batch_c % tl.cdiv(channels, BLOCK_C)
    
    # 边界检查
    if pid_batch >= batch:
        return
    if pid_c * BLOCK_C >= channels:
        return
    
    # 初始化位置参数
    c_start = pid_c * BLOCK_C
    c_end = tl.minimum(c_start + BLOCK_C, channels)
    c_mask = tl.arange(0, BLOCK_C) < (c_end - c_start)
    
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    
    batch_offset = pid_batch * height * width * channels
    
    # 重构后的循环结构（支持Triton JIT）
    for h_offset in range(BLOCK_H):
        h = h_start + h_offset
        # 改写为条件判断代替break
        if h < out_height:
            for w_offset in range(BLOCK_W):
                w = w_start + w_offset
                if w < out_width:  # 有效位置判断
                    # 输入起始位置计算
                    in_h = h * stride_h - padding_h
                    in_w = w * stride_w - padding_w
                    
                    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
                    
                    # 卷积核循环
                    for kh in range(kernel_h):
                        ih = in_h + kh
                        # 显式边界检查
                        if ih >= 0 and ih < height:
                            for kw in range(kernel_w):
                                iw = in_w + kw
                                if iw >= 0 and iw < width:
                                    # 输入数据访问
                                    input_offset = (
                                        batch_offset + 
                                        ih * width * channels + 
                                        iw * channels + 
                                        c_start + tl.arange(0, BLOCK_C)
                                    )
                                    x = tl.load(input_ptr + input_offset, mask=c_mask, other=0.0)
                                    
                                    # 权重访问（通道优先布局）
                                    # weight_offset = (
                                    #     (c_start + tl.arange(0, BLOCK_C)) * kernel_h * kernel_w + 
                                    #     kh * kernel_w + 
                                    #     kw
                                    # )
                                    weight_offset = (
                                        kh * kernel_w * channels +  # 当前kernel行偏移
                                        kw * channels +             # 当前kernel列偏移
                                        c_start + tl.arange(0, BLOCK_C)  # 通道偏移
                                    )
                                    w_val = tl.load(weight_ptr + weight_offset, mask=c_mask, other=0.0)
                                    
                                    acc += x * w_val
                    
                    # 结果存储
                    output_offset = (
                        batch_offset + 
                        h * out_width * channels + 
                        w * channels + 
                        c_start + tl.arange(0, BLOCK_C)
                    )
                    tl.store(output_ptr + output_offset, acc, mask=c_mask)


class OptimizedDepthwiseConv2d(torch.nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0, dilation=1, dtype=None):
        super(OptimizedDepthwiseConv2d, self).__init__()
        
        # 处理卷积核参数
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
            
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # 设置默认数据类型，支持fp16和bf16
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
            assert dtype in [torch.float16, torch.bfloat16], "仅支持fp16和bf16数据类型"
        
        # 初始化权重
        self.weight = torch.nn.Parameter(
            torch.empty(kernel_size[0], kernel_size[1], channels, dtype=self.dtype)
        )
        torch.nn.init.kaiming_uniform_(self.weight)
        
    def forward(self, x):
        # 检查输入格式，必须是BHWC
        assert x.dim() == 4 and x.shape[3] == self.channels, "输入必须是BHWC格式，且通道数匹配"
        
        # 确保输入和权重使用相同数据类型
        x = x.to(self.dtype)
        
        # 获取输入维度
        batch, height, width, channels = x.shape
        
        # 计算输出维度
        out_height = (height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        # 创建输出tensor
        output = torch.empty((batch, out_height, out_width, channels), device=x.device, dtype=self.dtype)
        
        # 计算网格维度 - 根据计算量动态调整
        grid = (
            triton.cdiv(out_height, 8),  # 默认值，会被autotune覆盖
            triton.cdiv(out_width, 8),   # 默认值，会被autotune覆盖
            batch * triton.cdiv(channels, 32)  # 默认值，会被autotune覆盖
        )
        
        # 为大卷积核选择合适的块大小
        kh_block = 4
        kw_block = 4
        
        if min(self.kernel_size) <= 3:
            # 小卷积核不需要分块
            kh_block = self.kernel_size[0]
            kw_block = self.kernel_size[1]
        elif max(self.kernel_size) >= 13:
            # 大卷积核使用更小的块
            kh_block = 4
            kw_block = 4
        
        # 启动triton kernel并进行自动调优
        _depthwise_conv_kernel_optimized[grid](
            x, self.weight, output,
            batch, height, width, channels,
            out_height, out_width,
            self.kernel_size[0], self.kernel_size[1],
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            KH_BLOCK=kh_block,
            KW_BLOCK=kw_block
        )
        
        return output


# 性能测试
def benchmark(batch_size=8, height=224, width=224, channels=64, kernel_size=13, dtype=torch.float16):
    import time
    
    # 创建输入
    x = torch.randn(batch_size, height, width, channels, dtype=dtype).cuda()
    
    # 创建我们的实现
    conv_triton = OptimizedDepthwiseConv2d(
        channels=channels,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        dtype=dtype
    ).cuda()
    
    # 创建PyTorch的实现（需要转换为BCHW格式）
    x_torch = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
    conv_torch = torch.nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        groups=channels,
        bias=False
    ).cuda().to(dtype)

    conv_torch.weight.data = conv_triton.weight.permute(2,0,1).unsqueeze(1)
    
    # 预热
    for _ in range(10):
        _ = conv_triton(x)
        _ = conv_torch(x_torch)
    
    torch.cuda.synchronize()
    
    # 测试我们的实现
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        _ = conv_triton(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / iterations
    
    # 测试PyTorch的实现
    start = time.time()
    for _ in range(iterations):
        _ = conv_torch(x_torch)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / iterations
    
    # 结果转换为毫秒
    triton_ms = triton_time * 1000
    torch_ms = torch_time * 1000
    
    print(f"批大小={batch_size}, 高={height}, 宽={width}, 通道数={channels}, 卷积核={kernel_size}x{kernel_size}")
    print(f"Triton: {triton_ms:.3f}ms, PyTorch: {torch_ms:.3f}ms")
    print(f"加速比: {torch_ms/triton_ms:.2f}x")
    
    # 验证结果正确性
    out_triton = conv_triton(x)
    out_torch = conv_torch(x_torch).permute(0, 2, 3, 1)  # BCHW -> BHWC
    
    max_diff = torch.max(torch.abs(out_triton - out_torch))
    print(f"最大绝对误差: {max_diff.item()}")
    
    return triton_ms, torch_ms

# 简单的使用演示
def example_usage():
    # 创建BHWC格式的输入张量
    batch_size = 2
    height, width = 64, 64
    channels = 512
    
    # 创建输入，BHWC格式 (2, 64, 64, 512)
    x = torch.randn(batch_size, height, width, channels, dtype=torch.float16).cuda()
    
    # 常见的大卷积核尺寸
    kernel_size = 13  # 介于11x11到17x17之间
    
    # 创建depth-wise卷积层
    conv = OptimizedDepthwiseConv2d(
        channels=channels,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,  # same padding
        dtype=torch.float16  # 可以更改为torch.bfloat16
    ).cuda()
    
    # 前向传播
    output = conv(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"数据类型: {output.dtype}")
    
    return output

if __name__ == "__main__":
    # 使用案例测试
    batch_size = 2
    height = 64
    width = 64
    channels = 512
    benchmark(batch_size, height, width, channels)