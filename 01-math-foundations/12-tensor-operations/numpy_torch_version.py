import numpy as np
import torch


def demo_shape_and_stride():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    print(f"NumPy  shape={a.shape}  strides(bytes)={a.strides}")
    # NumPy strides are in bytes; PyTorch strides are in elements
    print(f"PyTorch shape={tuple(t.shape)}  strides(elements)={t.stride()}")


def demo_view_vs_reshape():
    t = torch.arange(6).reshape(2, 3)
    tr = t.transpose(0, 1)

    print(f"t contiguous:  {t.is_contiguous()}")
    print(f"tr contiguous: {tr.is_contiguous()}")

    try:
        tr.view(6)
    except RuntimeError as e:
        print(f"view on non-contiguous fails: {str(e)[:60]}...")

    print(f"reshape works (copies when needed): {tr.reshape(6)}")
    print(f"contiguous().view works: {tr.contiguous().view(6)}")


def demo_unsqueeze_and_broadcast():
    x = torch.randn(4, 3)
    bias = torch.tensor([0.1, 0.2, 0.3])
    print(f"(4,3) + (3,) -> {tuple((x + bias).shape)}")

    col = torch.tensor([1, 2, 3]).unsqueeze(1)
    row = torch.tensor([10, 20, 30, 40]).unsqueeze(0)
    print(f"(3,1) * (1,4) -> {tuple((col * row).shape)}")


def demo_einsum():
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    out = torch.einsum("ik,kj->ij", A, B)
    print(f"einsum matmul matches @: {torch.allclose(out, A @ B)}")

    Q = torch.randn(2, 4, 8, 16)
    K = torch.randn(2, 4, 8, 16)
    scores = torch.einsum("bhtd,bhsd->bhts", Q, K)
    print(f"attention scores shape: {tuple(scores.shape)}")


def demo_numpy_torch_bridge():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = torch.from_numpy(a)
    # from_numpy shares memory with the source array
    a[0, 0] = 100
    print(f"shared memory: t[0,0]={t[0, 0].item()}")
    print(f"back to numpy: {t.numpy().shape}")


if __name__ == "__main__":
    demo_shape_and_stride()
    demo_view_vs_reshape()
    demo_unsqueeze_and_broadcast()
    demo_einsum()
    demo_numpy_torch_bridge()
