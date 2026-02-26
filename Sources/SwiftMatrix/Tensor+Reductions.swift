/// Reductions and linear algebra operations for ``Tensor``.

extension Tensor where Element: Numeric {
    /// Computes the dot product (inner product) of two rank-1 tensors.
    ///
    /// ```swift
    /// let a = Tensor(shape: [3], elements: [1, 2, 3])
    /// let b = Tensor(shape: [3], elements: [4, 5, 6])
    /// Tensor.dot(a, b)  // 32  (1*4 + 2*5 + 3*6)
    /// ```
    ///
    /// - Precondition: Both tensors are rank-1 with the same shape.
    /// - Returns: The sum of element-wise products.
    public static func dot(_ lhs: Tensor, _ rhs: Tensor) -> Element {
        precondition(lhs.rank == 1 && rhs.rank == 1,
                     "dot requires rank-1 tensors, got ranks \(lhs.rank) and \(rhs.rank)")
        precondition(lhs.shape == rhs.shape,
                     "Shape mismatch: \(lhs.shape) vs \(rhs.shape)")
        return zip(lhs, rhs).reduce(.zero) { $0 + $1.0 * $1.1 }
    }

    /// Multiplies two rank-2 tensors (matrix multiplication).
    ///
    /// ```swift
    /// let a = Tensor([[1, 2], [3, 4]])       // 2x2
    /// let b = Tensor([[5, 6], [7, 8]])       // 2x2
    /// Tensor.matmul(a, b)  // [[19, 22], [43, 50]]
    /// ```
    ///
    /// - Precondition: Both tensors are rank-2 and `lhs.shape[1] == rhs.shape[0]`.
    /// - Returns: A tensor with shape `[lhs.shape[0], rhs.shape[1]]`.
    public static func matmul(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
        precondition(lhs.rank == 2 && rhs.rank == 2,
                     "matmul requires rank-2 tensors, got ranks \(lhs.rank) and \(rhs.rank)")
        let m = lhs.shape[0]
        let inner = lhs.shape[1]
        let n = rhs.shape[1]
        precondition(inner == rhs.shape[0],
                     "Inner dimensions must match: \(lhs.shape) vs \(rhs.shape)")
        var result = [Element]()
        result.reserveCapacity(m * n)
        for i in 0..<m {
            for j in 0..<n {
                var sum: Element = .zero
                for k in 0..<inner {
                    sum += lhs[i, k] * rhs[k, j]
                }
                result.append(sum)
            }
        }
        return Tensor(shape: [m, n], elements: result)
    }
}

// MARK: - Sum

extension Tensor where Element: AdditiveArithmetic {
    /// Returns the sum of all elements.
    ///
    /// ```swift
    /// Tensor([[1, 2, 3], [4, 5, 6]]).sum()  // 21
    /// ```
    public func sum() -> Element {
        reduce(.zero, +)
    }

    /// Returns a tensor with one axis collapsed by summation.
    ///
    /// The result has rank reduced by 1. For a tensor with shape `[2, 3]`:
    /// - `sum(axis: 0)` produces shape `[3]` (sum across rows)
    /// - `sum(axis: 1)` produces shape `[2]` (sum across columns)
    ///
    /// - Parameter axis: The axis to sum along.
    /// - Precondition: `axis` is in `0..<rank`.
    public func sum(axis: Int) -> Tensor {
        precondition(axis >= 0 && axis < rank,
                     "Axis \(axis) out of range for rank \(rank)")
        var newShape = shape
        newShape.remove(at: axis)
        if newShape.isEmpty {
            return Tensor(shape: [], elements: [sum()])
        }
        let axisSize = shape[axis]
        var result = [Element]()
        result.reserveCapacity(newShape.reduce(1, *))
        // Iterate over every index combination in the output shape
        let outputCount = newShape.reduce(1, *)
        let outputStrides = Self.computeStrides(for: newShape)
        for outputLinear in 0..<outputCount {
            // Convert output linear index to multi-dimensional output indices
            var outputIndices = [Int]()
            outputIndices.reserveCapacity(newShape.count)
            var remaining = outputLinear
            for s in outputStrides {
                outputIndices.append(remaining / s)
                remaining %= s
            }
            // Build the full input index by inserting the summed axis
            var inputIndices = outputIndices
            inputIndices.insert(0, at: axis)
            var total: Element = .zero
            for k in 0..<axisSize {
                inputIndices[axis] = k
                total += self[inputIndices]
            }
            result.append(total)
        }
        return Tensor(shape: newShape, elements: result)
    }
}

// MARK: - Mean

extension Tensor where Element: FloatingPoint {
    /// Returns the arithmetic mean of all elements.
    ///
    /// ```swift
    /// Tensor(shape: [4], elements: [1.0, 2.0, 3.0, 4.0]).mean()  // 2.5
    /// ```
    public func mean() -> Element {
        sum() / Element(count)
    }

    /// Returns a tensor with one axis collapsed by averaging.
    ///
    /// The result has rank reduced by 1. For a tensor with shape `[2, 3]`:
    /// - `mean(axis: 0)` produces shape `[3]` (mean across rows)
    /// - `mean(axis: 1)` produces shape `[2]` (mean across columns)
    ///
    /// - Parameter axis: The axis to average along.
    /// - Precondition: `axis` is in `0..<rank`.
    public func mean(axis: Int) -> Tensor {
        let s = sum(axis: axis)
        let divisor = Element(shape[axis])
        return Tensor(shape: s.shape, elements: s.map { $0 / divisor })
    }
}
