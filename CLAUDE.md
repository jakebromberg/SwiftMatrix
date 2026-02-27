# SwiftMatrix -- Project Guide

## Build & Test

```bash
swift build
swift test
```

Tests use Swift Testing (`@Test`, `#expect`). No XCTest.

## Project Structure

```
Sources/SwiftMatrix/     -- library source
Tests/SwiftMatrixTests/  -- tests
```

## Conventions

- Swift 6.2, SPM-based project
- All public API types and members must be explicitly marked `public`
- Tests use Swift Testing framework (`import Testing`, `@Test`, `#expect`)
- Prefer value types (structs)

## Architecture

`Tensor<Element>` uses flat `[Element]` storage with `shape: [Int]`, `strides: [Int]`, and `offset: Int`. Row-major (C-order) strides by default. `count` and `isContiguous` are cached as stored properties.

### Collection

`Index = Int` (linear index) for `RandomAccessCollection`. Multi-dimensional access via `subscript(_ indices: Int...)`. Conditional conformances: `Equatable`, `Hashable`, `Sendable` (when `Element` conforms).

### Views (zero-copy)

`permuted(axes:)`, `transposed()`, `reshaped(to:)`, `slice(axis:range:)` -- share underlying storage via offset/strides manipulation.

### Arithmetic

Element-wise `+`, `-`, `*`, `/`, negation, scalar variants, compound assignment. Constrained on `AdditiveArithmetic`, `Numeric`, `FloatingPoint`, `SignedNumeric` as appropriate. Broadcasting supported (NumPy-style right-aligned shape compatibility with stride-0 views). Contiguous fast paths bypass `storageIndex` overhead; in-place compound assignment mutates storage directly.

### Lazy Evaluation

`TensorExpression` protocol enables deferred computation trees. `Tensor.lazy` returns a `LazyTensor`; operators on expressions return `BinaryExpression`/`UnaryExpression`. `Tensor(evaluating:)` materializes in one pass with zero intermediates. `Tensor` does not conform to `TensorExpression` -- eager and lazy are cleanly separated.

### Reductions

`sum()`, `sum(axis:)`, `mean()`, `mean(axis:)`, `dot(_:_:)`, `matmul(_:_:)`. Constrained on `AdditiveArithmetic`/`Numeric`/`FloatingPoint`.

### Sparse Types

`COOTensor<Element>` stores nonzero entries in Coordinate format using struct-of-arrays layout: `indices[axis][entry]` gives the coordinate along that axis. Entries are sorted in row-major lexicographic order with no duplicates (summed at construction). Works for any rank.

`CSRMatrix<Element>` stores rank-2 sparse matrices in Compressed Sparse Row format: `rowPointers[i]` indexes into parallel `columnIndices` and `values` arrays. Column indices within each row are sorted.

Both are separate types from `Tensor` -- conversions via `init(from:)` and `toTensor()`.

### Performance

- `logicalStrides: [Int]` cached at init time (avoids per-access `computeStrides` allocations)
- `withContiguousStorageIfAvailable` for direct buffer access on contiguous tensors
- `StridedIterator` for O(1) amortized non-contiguous traversal (coordinate increment with carry)
- Contiguous fast paths in `elementwise()` and scalar operators
- In-place compound assignment (`+=`, `-=`, `*=`, `/=`) via `elementwiseInPlace()`

### Accelerate optimizations

On Apple platforms, `Float` and `Double` tensors use vDSP/CBLAS via the `AccelerateFloatingPoint` protocol. Overload resolution selects the Accelerate path automatically; generic implementations remain as fallbacks for other element types and non-Apple platforms. Wrapped in `#if canImport(Accelerate)`. Covers reductions (sum, mean, dot, matmul) and element-wise arithmetic (+, -, *, /, scalar variants, negation).
