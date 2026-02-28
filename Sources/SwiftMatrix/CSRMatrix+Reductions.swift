/// Reductions for ``CSRMatrix``.

// MARK: - Sum

extension CSRMatrix where Element: AdditiveArithmetic {
    /// Returns the sum of all stored (nonzero) elements. O(nnz).
    ///
    /// Implicit zeros do not affect the sum, so this is equivalent to summing
    /// the full dense matrix.
    ///
    /// ```swift
    /// let csr = CSRMatrix(rows: 2, columns: 2,
    ///     rowPointers: [0, 1, 2], columnIndices: [0, 1], values: [10, 20])
    /// csr.sum()  // 30
    /// ```
    public func sum() -> Element {
        values.reduce(.zero, +)
    }

    /// Returns a tensor with one axis collapsed by summation. O(nnz).
    ///
    /// - `axis: 0` produces column sums with shape `[columns]`.
    /// - `axis: 1` produces row sums with shape `[rows]`.
    ///
    /// ```swift
    /// // [[1, 0, 2], [0, 3, 0]]
    /// csr.sum(axis: 0)  // [1, 3, 2]  (column sums)
    /// csr.sum(axis: 1)  // [3, 3]     (row sums)
    /// ```
    ///
    /// - Parameter axis: The axis to sum along (0 or 1).
    /// - Precondition: `axis` is 0 or 1.
    public func sum(axis: Int) -> Tensor<Element> {
        precondition(axis == 0 || axis == 1,
                     "Axis \(axis) out of range for rank-2 CSR matrix")

        if axis == 1 {
            // Row sums: sum entries in each row's range
            var result = [Element]()
            result.reserveCapacity(rows)
            for row in 0..<rows {
                var total: Element = .zero
                for idx in rowPointers[row]..<rowPointers[row + 1] {
                    total = total + values[idx]
                }
                result.append(total)
            }
            return Tensor(shape: [rows], elements: result)
        } else {
            // Column sums: accumulate into column buckets
            var result = Array(repeating: Element.zero, count: columns)
            for idx in 0..<nnz {
                result[columnIndices[idx]] = result[columnIndices[idx]] + values[idx]
            }
            return Tensor(shape: [columns], elements: result)
        }
    }
}

// MARK: - Mean

extension CSRMatrix where Element: FloatingPoint {
    /// Returns the arithmetic mean over all logical elements. O(nnz).
    ///
    /// The divisor is `rows * columns` (not ``nnz``), so implicit zeros are
    /// included in the average.
    ///
    /// ```swift
    /// // [[1, 0], [0, 3]] -> sum=4, count=4 -> mean=1.0
    /// csr.mean()  // 1.0
    /// ```
    public func mean() -> Element {
        sum() / Element(count)
    }

    /// Returns a tensor with one axis collapsed by averaging. O(nnz).
    ///
    /// Divides `sum(axis:)` by the size of the collapsed axis.
    /// - `axis: 0` divides by `rows` (column means).
    /// - `axis: 1` divides by `columns` (row means).
    ///
    /// - Parameter axis: The axis to average along (0 or 1).
    /// - Precondition: `axis` is 0 or 1.
    public func mean(axis: Int) -> Tensor<Element> {
        let s = sum(axis: axis)
        let divisor = axis == 0 ? Element(rows) : Element(columns)
        return Tensor(shape: s.shape, elements: s.map { $0 / divisor })
    }
}
