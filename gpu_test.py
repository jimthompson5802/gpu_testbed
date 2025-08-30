import torch
import sys


def test_gpu_availability():
    """Test if GPU is available and print detailed information."""
    print("PyTorch GPU Test")
    print("Hello World")
    print("=" * 50)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("✅ GPU is available!")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print()

        # Get number of GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        print()

        # Print information for each GPU
        for i in range(num_gpus):
            print(f"GPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(
                f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB"
            )
            print(
                f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"  # noqa E501
            )
            print(
                f"  Multi-processor count: {torch.cuda.get_device_properties(i).multi_processor_count}"
            )

            # Memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  Memory allocated: {memory_allocated:.2f} GB")
                print(f"  Memory reserved: {memory_reserved:.2f} GB")
            print()

        # Test GPU computation
        print("Testing GPU computation...")
        device = torch.device("cuda")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print("✅ GPU computation test successful!")
        print(f"Result tensor shape: {z.shape}")
        print(f"Result tensor device: {z.device}")

    else:
        print("❌ GPU is not available")
        print("Reasons GPU might not be available:")
        print("- No NVIDIA GPU installed")
        print("- CUDA drivers not installed")
        print("- PyTorch not compiled with CUDA support")
        print()
        print("Testing CPU computation...")
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        z = torch.matmul(x, y)
        print("✅ CPU computation works fine")
        print(f"PyTorch version: {torch.__version__}")


if __name__ == "__main__":
    test_gpu_availability()
