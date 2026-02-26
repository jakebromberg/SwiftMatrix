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

`Tensor<Element>` uses flat `[Element]` storage with `shape: [Int]`, `strides: [Int]`, and `offset: Int`. Row-major (C-order) strides by default. The `offset` field enables zero-copy views for future slice/transpose operations.

Collection conformance uses `Index = Int` (linear index) for `RandomAccessCollection`. Multi-dimensional access via `subscript(_ indices: Int...)`.

Conditional conformances: `Equatable`, `Hashable`, `Sendable` (when `Element` conforms).
