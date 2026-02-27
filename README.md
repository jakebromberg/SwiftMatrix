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

Element-wise `+`, `-`, `*`, `/`, negation, scalar variants, and compound assignment (`+=`, `*=`).

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
```

### Accelerate optimizations

On Apple platforms, `Float` and `Double` tensors automatically use vDSP and CBLAS for reductions and matrix multiplication. No API changes required -- overload resolution selects the optimized path. Generic implementations serve as fallbacks for other element types and non-Apple platforms.
