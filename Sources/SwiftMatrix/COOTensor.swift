/// A sparse tensor in Coordinate (COO) format.
///
/// Stores only the nonzero entries using a struct-of-arrays layout: `indices[axis][entry]`
/// gives the coordinate along that axis for a given entry. Entries are stored in row-major
/// lexicographic order with no duplicates (duplicates are summed at construction time).
///
/// Works for any rank. For rank-2 sparse matrices, consider ``CSRMatrix`` for more efficient
/// row-based access patterns.
///
/// ```swift
/// let coo = COOTensor(
///     shape: [3, 4],
///     indices: [[0, 1, 2], [1, 2, 0]],
///     values: [10, 20, 30]
/// )
/// coo.nnz      // 3
/// coo.density   // 0.25
/// ```
public struct COOTensor<Element> {
    /// The size of each axis.
    public let shape: [Int]

    /// Nonzero entry coordinates in struct-of-arrays layout.
    ///
    /// `indices[axis][entry]` gives the coordinate along `axis` for entry at position `entry`.
    /// Each inner array has length ``nnz``.
    public let indices: [[Int]]

    /// The nonzero values, parallel to ``indices``.
    public let values: [Element]

    /// The number of stored (nonzero) entries.
    public var nnz: Int { values.count }

    /// The number of axes (dimensions).
    public var rank: Int { shape.count }

    /// The total number of logical elements (product of shape dimensions).
    public var count: Int { shape.reduce(1, *) }

    /// The fraction of elements that are stored (nonzero).
    public var density: Double {
        let total = count
        guard total > 0 else { return 0 }
        return Double(nnz) / Double(total)
    }

    /// Creates a COO tensor from indices and values, sorting entries and summing duplicates.
    ///
    /// - Parameters:
    ///   - shape: The size of each axis.
    ///   - indices: Struct-of-arrays coordinates. Must have one `[Int]` per axis,
    ///     each of the same length as `values`.
    ///   - values: The nonzero values.
    /// - Precondition: `indices.count` equals the rank, and all inner arrays have length
    ///   equal to `values.count`.
    public init(shape: [Int], indices: [[Int]], values: [Element]) where Element: AdditiveArithmetic {
        precondition(indices.count == shape.count,
                     "Expected \(shape.count) index arrays, got \(indices.count)")
        let nnz = values.count
        precondition(indices.allSatisfy { $0.count == nnz },
                     "All index arrays must have length \(nnz)")

        if nnz == 0 {
            self.shape = shape
            self.indices = indices
            self.values = values
            return
        }

        // Sort entries by row-major lexicographic order
        let strides = Tensor<Element>.computeStrides(for: shape)
        var entryOrder = Array(0..<nnz)
        entryOrder.sort { a, b in
            for axis in 0..<shape.count {
                if indices[axis][a] != indices[axis][b] {
                    return indices[axis][a] < indices[axis][b]
                }
            }
            return false
        }

        // Apply sort and sum duplicates in one pass
        var sortedIndices = Array(repeating: [Int](), count: shape.count)
        var sortedValues = [Element]()

        for i in entryOrder {
            let isDuplicate: Bool
            if sortedValues.isEmpty {
                isDuplicate = false
            } else {
                isDuplicate = (0..<shape.count).allSatisfy { axis in
                    sortedIndices[axis].last! == indices[axis][i]
                }
            }

            if isDuplicate {
                sortedValues[sortedValues.count - 1] = sortedValues[sortedValues.count - 1] + values[i]
            } else {
                for axis in 0..<shape.count {
                    sortedIndices[axis].append(indices[axis][i])
                }
                sortedValues.append(values[i])
            }
        }

        self.shape = shape
        self.indices = sortedIndices
        self.values = sortedValues
    }

    /// Creates a COO tensor from pre-sorted indices with no duplicates.
    ///
    /// This is an internal fast path that skips sorting and deduplication.
    init(shape: [Int], sortedIndices: [[Int]], values: [Element]) {
        self.shape = shape
        self.indices = sortedIndices
        self.values = values
    }

    /// Creates an empty COO tensor with the given shape.
    public init(shape: [Int]) where Element: AdditiveArithmetic {
        self.shape = shape
        self.indices = Array(repeating: [], count: shape.count)
        self.values = []
    }
}

extension COOTensor: Sendable where Element: Sendable {}

extension COOTensor: Equatable where Element: Equatable {
    public static func == (lhs: COOTensor, rhs: COOTensor) -> Bool {
        lhs.shape == rhs.shape && lhs.indices == rhs.indices && lhs.values == rhs.values
    }
}
