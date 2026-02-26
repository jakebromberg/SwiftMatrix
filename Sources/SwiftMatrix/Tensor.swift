/// A rank-N multi-dimensional array with flat storage.
///
/// `Tensor` stores elements in a contiguous `[Element]` array and uses `shape` and `strides`
/// to interpret the flat buffer as a multi-dimensional structure. This is the same storage
/// model used by NumPy and PyTorch.
///
/// By default, strides are row-major (C-order): the last axis varies fastest. For a tensor
/// with shape `[2, 3, 4]`, the strides are `[12, 4, 1]`.
///
/// The `offset` property supports zero-copy views. Operations like slicing can produce a new
/// `Tensor` that shares the same underlying storage but reads from a different starting
/// position.
///
/// ```swift
/// let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
/// t[1, 2]    // 6
/// Array(t)   // [1, 2, 3, 4, 5, 6]  (row-major iteration)
/// ```
public struct Tensor<Element> {
    var storage: [Element]

    /// The size of each axis.
    ///
    /// For a 2x3 matrix, `shape` is `[2, 3]`. A scalar has an empty shape `[]`.
    public let shape: [Int]

    /// The number of elements to skip in storage when advancing one position along each axis.
    ///
    /// For row-major layout with shape `[2, 3]`, strides are `[3, 1]`: advancing one step
    /// along axis 0 skips 3 elements, while advancing along axis 1 skips 1.
    public let strides: [Int]

    /// The starting position in storage for this tensor's data.
    ///
    /// Normally `0` for tensors that own their storage. Non-zero offsets arise from
    /// zero-copy slicing, where a view into a larger buffer starts at a later position.
    public let offset: Int

    /// The number of axes (dimensions).
    ///
    /// A scalar has rank 0, a vector has rank 1, a matrix has rank 2, and so on.
    public var rank: Int { shape.count }

    /// Creates a tensor filled with a repeated value.
    ///
    /// - Parameters:
    ///   - shape: The size of each axis. For example, `[2, 3]` creates a 2x3 matrix.
    ///   - repeatedValue: The value to fill every element with.
    public init(shape: [Int], repeating repeatedValue: Element) {
        let count = shape.reduce(1, *)
        self.storage = Array(repeating: repeatedValue, count: count)
        self.shape = shape
        self.strides = Self.computeStrides(for: shape)
        self.offset = 0
    }

    /// Creates a tensor from a flat array of elements and a shape.
    ///
    /// The number of elements must equal the product of the shape dimensions.
    ///
    /// - Parameters:
    ///   - shape: The size of each axis.
    ///   - elements: The elements in row-major order.
    /// - Precondition: `elements.count` equals the product of `shape`.
    public init(shape: [Int], elements: [Element]) {
        let count = shape.reduce(1, *)
        precondition(elements.count == count,
                     "Element count \(elements.count) does not match shape \(shape) (expected \(count))")
        self.storage = elements
        self.shape = shape
        self.strides = Self.computeStrides(for: shape)
        self.offset = 0
    }

    /// Creates a rank-2 tensor from nested arrays.
    ///
    /// The outer array becomes axis 0 (rows) and each inner array becomes axis 1 (columns).
    /// All inner arrays must have the same length.
    ///
    /// ```swift
    /// let t = Tensor([[1, 2, 3],
    ///                 [4, 5, 6]])
    /// // shape: [2, 3]
    /// ```
    ///
    /// - Parameter arrays: The rows of the matrix. Each row must have the same length.
    /// - Precondition: All inner arrays have equal count.
    public init(_ arrays: [[Element]]) {
        let depth = arrays.count
        guard depth > 0 else {
            self.init(shape: [0, 0], elements: [])
            return
        }
        let width = arrays[0].count
        precondition(arrays.allSatisfy { $0.count == width },
                     "All rows must have the same length")
        self.init(shape: [depth, width], elements: arrays.flatMap { $0 })
    }

    /// Creates a tensor with explicit storage layout. Used internally for views.
    init(storage: [Element], shape: [Int], strides: [Int], offset: Int) {
        self.storage = storage
        self.shape = shape
        self.strides = strides
        self.offset = offset
    }

    /// Computes row-major (C-order) strides for the given shape.
    ///
    /// Each stride is the product of all subsequent dimension sizes. For shape `[2, 3, 4]`,
    /// returns `[12, 4, 1]`.
    static func computeStrides(for shape: [Int]) -> [Int] {
        guard !shape.isEmpty else { return [] }
        var strides = Array(repeating: 1, count: shape.count)
        for i in Swift.stride(from: shape.count - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * shape[i + 1]
        }
        return strides
    }

    /// Whether this tensor's storage is laid out contiguously in row-major order with no offset.
    ///
    /// Contiguous tensors can use the linear index directly as the storage index, avoiding
    /// the cost of unraveling multi-dimensional coordinates.
    var isContiguous: Bool {
        strides == Self.computeStrides(for: shape) && offset == 0
    }

    /// Converts multi-dimensional indices to a flat storage index.
    ///
    /// - Parameter indices: One index per axis.
    /// - Returns: The position in `storage` for the given coordinates.
    /// - Precondition: `indices.count` equals ``rank``.
    func storageIndex(for indices: [Int]) -> Int {
        precondition(indices.count == rank,
                     "Expected \(rank) indices, got \(indices.count)")
        return offset + zip(indices, strides).reduce(0) { $0 + $1.0 * $1.1 }
    }

    /// Converts a linear (row-major) index to a flat storage index.
    ///
    /// For contiguous tensors, the linear index equals the storage index directly.
    /// For non-contiguous views (e.g., after transpose), this unravels the linear index
    /// into multi-dimensional coordinates using the logical shape, then computes the
    /// physical storage position using the actual strides.
    ///
    /// - Parameter linear: A linear index in `0..<count`.
    /// - Returns: The position in `storage`.
    func storageIndex(forLinearIndex linear: Int) -> Int {
        guard !isContiguous else { return linear }
        var remaining = linear
        var storageIdx = offset
        for axis in 0..<rank {
            let logicalStride = shape[(axis + 1)...].reduce(1, *)
            let coord = remaining / logicalStride
            remaining %= logicalStride
            storageIdx += coord * strides[axis]
        }
        return storageIdx
    }
}

extension Tensor: Sendable where Element: Sendable {}
