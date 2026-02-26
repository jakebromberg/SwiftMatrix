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

### Accelerate optimizations

On Apple platforms, `Float` and `Double` tensors automatically use vDSP and CBLAS for reductions and matrix multiplication. No API changes required -- overload resolution selects the optimized path. Generic implementations serve as fallbacks for other element types and non-Apple platforms.
