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

Element-wise `+`, `-`, `*`, `/`, negation, scalar variants, compound assignment. Constrained on `AdditiveArithmetic`, `Numeric`, `FloatingPoint`, `SignedNumeric` as appropriate.

### Reductions

`sum()`, `sum(axis:)`, `mean()`, `mean(axis:)`, `dot(_:_:)`, `matmul(_:_:)`. Constrained on `AdditiveArithmetic`/`Numeric`/`FloatingPoint`.

### Accelerate optimizations

On Apple platforms, `Float` and `Double` tensors use vDSP/CBLAS via the `AccelerateFloatingPoint` protocol. Overload resolution selects the Accelerate path automatically; generic implementations remain as fallbacks for other element types and non-Apple platforms. Wrapped in `#if canImport(Accelerate)`.
