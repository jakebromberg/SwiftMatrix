extension Tensor {
    /// Returns a view with axes reordered according to the given permutation.
    ///
    /// This is a zero-copy operation -- the returned tensor shares the same underlying storage
    /// but has rearranged shape and strides.
    ///
    /// ```swift
    /// let t = Tensor([[1, 2, 3], [4, 5, 6]])  // shape [2, 3]
    /// let p = t.permuted(axes: [1, 0])         // shape [3, 2]
    /// Array(p)  // [1, 4, 2, 5, 3, 6]
    /// ```
    ///
    /// - Parameter axes: A permutation of `0..<rank` specifying the new axis order.
    ///   `axes[i]` is the old axis that becomes new axis `i`.
    /// - Precondition: `axes` is a valid permutation of `0..<rank`.
    /// - Returns: A view with permuted axes.
    public func permuted(axes: [Int]) -> Tensor {
        precondition(axes.count == rank,
                     "Expected \(rank) axes, got \(axes.count)")
        precondition(Set(axes) == Set(0..<rank),
                     "axes must be a permutation of 0..<\(rank)")
        let newShape = axes.map { shape[$0] }
        let newStrides = axes.map { strides[$0] }
        let contiguous = newStrides == Self.computeStrides(for: newShape) && offset == 0
        return Tensor(storage: storage, shape: newShape, strides: newStrides,
                      offset: offset, isContiguous: contiguous)
    }

    /// Returns a transposed view of a rank-2 tensor.
    ///
    /// Equivalent to `permuted(axes: [1, 0])`.
    ///
    /// - Precondition: ``rank`` is 2.
    /// - Returns: A view with rows and columns swapped.
    public func transposed() -> Tensor {
        precondition(rank == 2, "transposed() requires rank 2, got rank \(rank)")
        return permuted(axes: [1, 0])
    }

    /// Returns a tensor with the same elements but a different shape.
    ///
    /// This is a zero-copy operation on contiguous tensors. The elements remain in the same
    /// order; only the shape and strides change.
    ///
    /// ```swift
    /// let t = Tensor([[1, 2, 3], [4, 5, 6]])  // shape [2, 3]
    /// let r = t.reshaped(to: [6])              // shape [6]
    /// Array(r)  // [1, 2, 3, 4, 5, 6]
    /// ```
    ///
    /// - Parameter newShape: The desired shape. Its product must equal ``count``.
    /// - Precondition: ``isContiguous`` is `true`.
    /// - Precondition: Product of `newShape` equals ``count``.
    /// - Returns: A tensor with the new shape.
    public func reshaped(to newShape: [Int]) -> Tensor {
        precondition(isContiguous,
                     "Cannot reshape a non-contiguous tensor; make a contiguous copy first")
        let newCount = newShape.reduce(1, *)
        precondition(newCount == count,
                     "New shape \(newShape) (product \(newCount)) does not match count \(count)")
        return Tensor(shape: newShape, elements: storage)
    }

    /// Returns a view selecting a sub-range along one axis.
    ///
    /// This is a zero-copy operation -- the returned tensor shares the same underlying storage
    /// with an adjusted offset and shape.
    ///
    /// ```swift
    /// let t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    /// let s = t.slice(axis: 0, range: 1..<3)  // shape [2, 3]
    /// Array(s)  // [4, 5, 6, 7, 8, 9]
    /// ```
    ///
    /// - Parameters:
    ///   - axis: The axis to slice along.
    ///   - range: The range of indices to select along `axis`.
    /// - Precondition: `axis` is in `0..<rank`.
    /// - Precondition: `range` is within `0..<shape[axis]`.
    /// - Returns: A view over the selected sub-range.
    public func slice(axis: Int, range: Range<Int>) -> Tensor {
        precondition(axis >= 0 && axis < rank,
                     "Axis \(axis) out of range for rank \(rank)")
        precondition(range.lowerBound >= 0 && range.upperBound <= shape[axis],
                     "Range \(range) out of bounds for axis \(axis) with size \(shape[axis])")
        var newShape = shape
        newShape[axis] = range.count
        let newOffset = offset + range.lowerBound * strides[axis]
        return Tensor(storage: storage, shape: newShape, strides: strides,
                      offset: newOffset, isContiguous: false)
    }
}
