/// Multi-dimensional subscript access for ``Tensor``.
///
/// These subscripts convert multi-dimensional coordinates (one index per axis) into a flat
/// storage index using the tensor's strides. Both get and set are supported; mutation through
/// a view triggers a copy-on-write of the underlying storage array.
extension Tensor {
    /// Accesses the element at the given multi-dimensional coordinates.
    ///
    /// ```swift
    /// let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
    /// t[[1, 2]]  // 6
    /// ```
    ///
    /// - Parameter indices: An array with one index per axis.
    /// - Precondition: `indices.count` equals ``rank``.
    public subscript(indices: [Int]) -> Element {
        get { storage[storageIndex(for: indices)] }
        set { storage[storageIndex(for: indices)] = newValue }
    }

    /// Accesses the element at the given multi-dimensional coordinates.
    ///
    /// ```swift
    /// let t = Tensor(shape: [2, 3, 4], repeating: 0)
    /// t[1, 2, 3]  // element at row 1, column 2, depth 3
    /// ```
    ///
    /// For rank-1 tensors, a single-argument call is ambiguous with the `Collection`
    /// position subscript; Swift resolves it to the `Collection` subscript, which produces
    /// the same result since the linear index and single-axis index are identical.
    ///
    /// - Parameter indices: One index per axis.
    /// - Precondition: `indices.count` equals ``rank``.
    public subscript(indices: Int...) -> Element {
        get { storage[storageIndex(for: indices)] }
        set { storage[storageIndex(for: indices)] = newValue }
    }
}
