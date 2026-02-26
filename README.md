# SwiftMatrix

A Swift library for multi-dimensional array (tensor) operations.

## Building

Requires Swift 6.2+.

```bash
swift build
swift test
```

## Current Types

- `Matrix<E>` -- rank-2 collection backed by nested arrays, indexed by `Pair`
- `Pair` -- 2D index with `x` and `y` components
- `Vector` -- 2D shape with `width` and `depth`
- `CollectionView` -- lazy index-remapping wrapper over any `Collection`

## Roadmap

- Flat-storage rank-N `Tensor` type with shape/strides
- Zero-copy transpose, slice, and reshape via stride manipulation
- Element-wise arithmetic
