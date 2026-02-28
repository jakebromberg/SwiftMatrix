# SwiftMatrix

A Swift library for multi-dimensional array (tensor) operations.

## Building

Requires Swift 6.2+.

```bash
swift build
swift test
```

## Types

### Tensor

`Tensor<Element>` is a rank-N multi-dimensional array backed by flat `[Element]` storage with shape/strides. Row-major (C-order) layout by default.

```swift
// Create from shape + repeating value
let zeros = Tensor(shape: [2, 3, 4], repeating: 0.0)

// Create from flat array + shape
let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])

// Create from nested arrays (rank 2)
let m = Tensor([[1, 2, 3], [4, 5, 6]])

// Multi-dimensional subscript
t[1, 2]  // 6

// Conforms to RandomAccessCollection (linear index)
Array(t)  // [1, 2, 3, 4, 5, 6]
```

### Zero-copy views

```swift
t.transposed()            // swap last two axes
t.permuted(axes: [1, 0])  // arbitrary axis permutation
t.reshaped(to: [6])       // reshape (contiguous tensors only)
t.slice(axis: 0, range: 0..<1)  // slice along an axis
```

### Arithmetic

Element-wise `+`, `-`, `*`, `/`, negation, scalar variants, and compound assignment (`+=`, `-=`, `*=`, `/=`). Broadcasting supported:

```swift
let a = Tensor(shape: [3, 1], elements: [1, 2, 3])
let b = Tensor(shape: [1, 4], elements: [10, 20, 30, 40])
let c = a + b  // shape [3, 4]
```

### Lazy evaluation

```swift
// Fuse a + b * c into one pass with zero intermediate allocations
let result = Tensor(evaluating: a.lazy + b.lazy * c.lazy)
```

### Reductions

```swift
t.sum()          // sum of all elements
t.sum(axis: 0)   // sum along axis
t.mean()         // arithmetic mean
t.mean(axis: 1)  // mean along axis

Tensor.dot(a, b)     // dot product (rank-1)
Tensor.matmul(a, b)  // matrix multiply (rank-2)
```

### Sparse Types

`COOTensor<Element>` stores sparse tensors of any rank in Coordinate (COO) format. `CSRMatrix<Element>` stores sparse rank-2 matrices in Compressed Sparse Row format.

```swift
// COO: general-rank sparse tensor
let coo = COOTensor(
    shape: [3, 4],
    indices: [[0, 1, 2], [1, 2, 0]],
    values: [10, 20, 30]
)
coo.nnz       // 3
coo.density   // 0.25

// CSR: rank-2 sparse matrix
let csr = CSRMatrix(
    rows: 3, columns: 3,
    rowPointers: [0, 2, 3, 4],
    columnIndices: [0, 2, 2, 0],
    values: [1, 2, 3, 4]
)

// Convert between formats
let dense = coo.toTensor()           // COO -> dense Tensor
let coo2 = COOTensor(from: dense)    // dense Tensor -> COO
let csr2 = CSRMatrix(from: dense)    // dense Tensor -> CSR
let coo3 = COOTensor(from: csr)      // CSR -> COO
let csr3 = CSRMatrix(from: coo)      // COO -> CSR

// Sparse arithmetic (same API for both COO and CSR)
let sum = coo + coo2       // element-wise addition
let diff = coo - coo2      // element-wise subtraction
let prod = coo * coo2      // element-wise multiplication (intersection)
let scaled = coo * 3       // scalar multiply
let divided = coo / 2.0    // scalar divide (FloatingPoint)
let neg = -coo             // negation
```

### Accelerate optimizations

On Apple platforms, `Float` and `Double` tensors automatically use vDSP and CBLAS for reductions, matrix multiplication, and element-wise arithmetic. No API changes required -- overload resolution selects the optimized path. Generic implementations serve as fallbacks for other element types and non-Apple platforms.

## Benchmarks

SwiftMatrix includes a benchmark suite that compares performance against two other Swift numerics libraries:

| Library | Storage | Accelerate | Notes |
|---------|---------|------------|-------|
| **SwiftMatrix** | Generic `Tensor<Element>`, value type | vDSP/CBLAS for Float/Double | This library |
| **[Surge](https://github.com/Jounce/Surge)** | `Matrix<Scalar>`, value type | All operations | Mature, unmaintained since 2021 |
| **[Matft](https://github.com/jjjkkkjjj/Matft)** | `MfArray`, class-based, type-erased | vDSP/CBLAS for Float/Double | NumPy-like API, actively maintained |

### Results (Float, Apple M4)

#### Matrix Addition (element-wise)

| Size | SwiftMatrix | Surge | Matft |
|------|-------------|-------|-------|
| 64x64 | 0.004 ms | 0.001 ms | 0.311 ms |
| 256x256 | 0.058 ms | 0.006 ms | 4.711 ms |
| 1024x1024 | 0.863 ms | 0.100 ms | 73.387 ms |

#### Matrix Multiplication (GEMM)

| Size | SwiftMatrix | Surge | Matft |
|------|-------------|-------|-------|
| 64x64 | 0.006 ms | 0.001 ms | 0.305 ms |
| 256x256 | 0.075 ms | 0.028 ms | 4.679 ms |
| 1024x1024 | 2.248 ms | 1.439 ms | 74.356 ms |

#### Sum Reduction

| Size | SwiftMatrix | Surge | Matft |
|------|-------------|-------|-------|
| 64x64 | <0.001 ms | <0.001 ms | 0.152 ms |
| 256x256 | 0.003 ms | 0.002 ms | 2.315 ms |
| 1024x1024 | 0.037 ms | 0.036 ms | 36.424 ms |

#### Dot Product

| Size | SwiftMatrix | Surge | Matft |
|------|-------------|-------|-------|
| 64x64 | 0.001 ms | <0.001 ms | 0.611 ms |
| 256x256 | 0.005 ms | 0.005 ms | 9.352 ms |
| 1024x1024 | 0.075 ms | 0.071 ms | 146.942 ms |

### Analysis

At large sizes the Accelerate kernel dominates, so SwiftMatrix and Surge converge to near-identical performance. Both call the same underlying vDSP and CBLAS routines (e.g. `cblas_sgemm` for GEMM, `vDSP_sve` for sum). The gap at small sizes reflects per-call overhead from SwiftMatrix's generic `Tensor` abstraction -- shape validation, stride computation, and `[Element]` allocation -- versus Surge's specialized `Matrix<Scalar>` that stores a flat `grid` array with fixed rank-2 layout.

Matft is 30--1000x slower across all benchmarks. Its `MfArray` uses class-based reference semantics with type-erased storage (`[Any]` bridging, `MfType` enum dispatch), which adds allocation and indirection overhead on every operation.

### Methodology

Each benchmark iteration includes constructing the library's matrix/tensor type from a pre-generated raw `[Float]` or `[Double]` array, performing the operation, and consuming the result. This measures end-to-end cost as a caller would experience it, not just the kernel.

- **Timing**: `ContinuousClock` (monotonic, nanosecond resolution)
- **Reported metric**: Median of all timed iterations
- **Warmup**: 1 untimed iteration before each measurement series
- **Iterations**: 50 (64x64), 20 (256x256), 10 (1024x1024)
- **Dead-code elimination prevention**: `@inline(never) func blackHole<T>(_ value: T)` consumes every result via `withExtendedLifetime`
- **Compiler**: Swift 6.2, release mode (`-c release`)
- **Data**: Random values in [0, 1), generated once per size, shared identically across all three libraries
- **Surge sum note**: Surge's `sum()` operates on the raw `[Float]` array directly (its `Matrix.grid` is internal), which is what its vDSP path does

Run the benchmarks yourself:

```bash
swift run -c release Benchmarks
```
