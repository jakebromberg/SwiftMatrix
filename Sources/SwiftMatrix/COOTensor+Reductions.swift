/// Reductions for ``COOTensor``.

extension COOTensor where Element: AdditiveArithmetic {
    /// Returns the sum of all stored (nonzero) elements. O(nnz).
    ///
    /// Implicit zeros do not affect the sum, so this is equivalent to summing
    /// the full dense tensor.
    ///
    /// ```swift
    /// let coo = COOTensor(shape: [4], indices: [[0, 2, 3]], values: [1, 2, 3])
    /// coo.sum()  // 6
    /// ```
    public func sum() -> Element {
        values.reduce(.zero, +)
    }
}

extension COOTensor where Element: FloatingPoint {
    /// Returns the arithmetic mean over all logical elements. O(nnz).
    ///
    /// The divisor is ``count`` (the product of shape dimensions), not ``nnz``,
    /// so implicit zeros are included in the average.
    ///
    /// ```swift
    /// let coo = COOTensor(shape: [4], indices: [[0, 1, 2]], values: [1.0, 2.0, 3.0])
    /// coo.mean()  // 1.5  (sum=6.0, count=4)
    /// ```
    public func mean() -> Element {
        sum() / Element(count)
    }
}

extension COOTensor where Element: Numeric {
    /// Computes the dot product (inner product) of two rank-1 sparse tensors. O(nnz_a + nnz_b).
    ///
    /// Uses two-pointer intersection on sorted indices, accumulating products only
    /// where both tensors have stored entries.
    ///
    /// ```swift
    /// let a = COOTensor(shape: [5], indices: [[0, 2, 4]], values: [1, 3, 5])
    /// let b = COOTensor(shape: [5], indices: [[1, 2, 3]], values: [2, 4, 6])
    /// COOTensor.dot(a, b)  // 12  (only index 2 overlaps: 3*4)
    /// ```
    ///
    /// - Precondition: Both tensors are rank-1 with the same shape.
    /// - Returns: The sum of element-wise products at matching indices.
    public static func dot(_ lhs: COOTensor, _ rhs: COOTensor) -> Element {
        precondition(lhs.rank == 1 && rhs.rank == 1,
                     "dot requires rank-1 tensors, got ranks \(lhs.rank) and \(rhs.rank)")
        precondition(lhs.shape == rhs.shape,
                     "Shape mismatch: \(lhs.shape) vs \(rhs.shape)")

        var result: Element = .zero
        var i = 0, j = 0
        let lhsIdx = lhs.indices[0]
        let rhsIdx = rhs.indices[0]

        while i < lhs.nnz && j < rhs.nnz {
            if lhsIdx[i] < rhsIdx[j] {
                i += 1
            } else if lhsIdx[i] > rhsIdx[j] {
                j += 1
            } else {
                result += lhs.values[i] * rhs.values[j]
                i += 1
                j += 1
            }
        }

        return result
    }
}
