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
Sources/Benchmarks/      -- performance benchmark suite
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

Element-wise `+`, `-`, `*`, scalar `*`, scalar `/`, negation, and compound assignment (`+=`, `-=`, `*=`). Uses two-pointer merge (addition/subtraction) and intersection (multiplication) on sorted entries for O(nnz_a + nnz_b) performance. No `sparse + scalar` (densifies) or `sparse / sparse` (divide by implicit zero).

### Sparse Reductions

`COOTensor`: `sum()`, `mean()`, `dot(_:_:)`. Sum/mean operate on stored values in O(nnz); mean divides by total `count` (not nnz). Dot uses two-pointer intersection on sorted rank-1 indices.

`CSRMatrix`: `sum()`, `sum(axis:)`, `mean()`, `mean(axis:)`. Axis 0 (column sums) accumulates into column buckets; axis 1 (row sums) iterates each row's range. All O(nnz). No axis reductions for COO (lacks row structure).

### Performance

- `logicalStrides: [Int]` cached at init time (avoids per-access `computeStrides` allocations)
- `withContiguousStorageIfAvailable` for direct buffer access on contiguous tensors
- `StridedIterator` for O(1) amortized non-contiguous traversal (coordinate increment with carry)
- Contiguous fast paths in `elementwise()` and scalar operators
- In-place compound assignment (`+=`, `-=`, `*=`, `/=`) via `elementwiseInPlace()`

### Accelerate optimizations

On Apple platforms, `Float` and `Double` tensors use vDSP/CBLAS via the `AccelerateFloatingPoint` protocol. Overload resolution selects the Accelerate path automatically; generic implementations remain as fallbacks for other element types and non-Apple platforms. Wrapped in `#if canImport(Accelerate)`. Covers reductions (sum, mean, dot, matmul) and element-wise arithmetic (+, -, *, /, scalar variants, negation).

### Benchmarks

`Sources/Benchmarks/` is an executable target that compares SwiftMatrix against [Surge](https://github.com/Jounce/Surge) and [Matft](https://github.com/jjjkkkjjj/Matft) on four operations (matrix addition, matrix multiplication, sum reduction, dot product) at three sizes (64x64, 256x256, 1024x1024) for both `Float` and `Double`.

```bash
swift run -c release Benchmarks
```

The target uses `.swiftLanguageMode(.v5)` because Surge and Matft lack `Sendable` conformances. Shared data is generated as raw `[Float]`/`[Double]` arrays and converted to each library's types outside the timed section. An `@inline(never)` `blackHole()` function prevents dead-code elimination in release builds.
