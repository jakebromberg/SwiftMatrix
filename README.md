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

### Legacy Types

- `Matrix<E>` -- rank-2 collection backed by nested arrays, indexed by `Pair`
- `Pair` -- 2D index with `x` and `y` components
- `Vector` -- 2D shape with `width` and `depth`
- `CollectionView` -- lazy index-remapping wrapper over any `Collection`

## Roadmap

- Zero-copy transpose, slice, and reshape via stride manipulation
- Element-wise arithmetic
- Compatibility layer bridging `Matrix`/`Vector`/`Pair` to `Tensor`
