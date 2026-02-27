/// A sparse rank-2 matrix in Compressed Sparse Row (CSR) format.
///
/// Stores nonzero entries using three parallel arrays: `rowPointers` indexes into
/// `columnIndices` and `values` to locate the entries for each row. Column indices
/// within each row are sorted.
///
/// For general-rank sparse data, use ``COOTensor``.
///
/// ```swift
/// // [[1, 0, 2],
/// //  [0, 0, 3],
/// //  [4, 0, 0]]
/// let csr = CSRMatrix(
///     rows: 3, columns: 3,
///     rowPointers: [0, 2, 3, 4],
///     columnIndices: [0, 2, 2, 0],
///     values: [1, 2, 3, 4]
/// )
/// ```
public struct CSRMatrix<Element> {
    /// The number of rows.
    public let rows: Int

    /// The number of columns.
    public let columns: Int

    /// Row pointer array of length `rows + 1`.
    ///
    /// The entries for row `i` are stored at positions `rowPointers[i]..<rowPointers[i+1]`
    /// in ``columnIndices`` and ``values``.
    public let rowPointers: [Int]

    /// Column indices of nonzero entries, sorted within each row.
    public let columnIndices: [Int]

    /// The nonzero values, parallel to ``columnIndices``.
    public let values: [Element]

    /// The number of stored (nonzero) entries.
    public var nnz: Int { values.count }

    /// The shape as `[rows, columns]`, for compatibility with ``Tensor`` conventions.
    public var shape: [Int] { [rows, columns] }

    /// The total number of logical elements (`rows * columns`).
    public var count: Int { rows * columns }

    /// The fraction of elements that are stored (nonzero).
    public var density: Double {
        let total = count
        guard total > 0 else { return 0 }
        return Double(nnz) / Double(total)
    }

    /// Creates a CSR matrix from pre-built arrays.
    ///
    /// - Parameters:
    ///   - rows: The number of rows.
    ///   - columns: The number of columns.
    ///   - rowPointers: Row pointer array of length `rows + 1`.
    ///   - columnIndices: Column indices, sorted within each row.
    ///   - values: Nonzero values, parallel to `columnIndices`.
    /// - Precondition: `rowPointers` has length `rows + 1`, `columnIndices` and `values`
    ///   have the same length, and `rowPointers.last == columnIndices.count`.
    public init(rows: Int, columns: Int, rowPointers: [Int], columnIndices: [Int], values: [Element]) {
        precondition(rowPointers.count == rows + 1,
                     "rowPointers must have length \(rows + 1), got \(rowPointers.count)")
        precondition(columnIndices.count == values.count,
                     "columnIndices and values must have the same length")
        precondition(rowPointers.last == columnIndices.count,
                     "rowPointers.last must equal columnIndices.count")

        self.rows = rows
        self.columns = columns
        self.rowPointers = rowPointers
        self.columnIndices = columnIndices
        self.values = values
    }

    /// Creates an empty CSR matrix with the given dimensions.
    public init(rows: Int, columns: Int) {
        self.rows = rows
        self.columns = columns
        self.rowPointers = Array(repeating: 0, count: rows + 1)
        self.columnIndices = []
        self.values = []
    }
}

extension CSRMatrix: Sendable where Element: Sendable {}

extension CSRMatrix: Equatable where Element: Equatable {
    public static func == (lhs: CSRMatrix, rhs: CSRMatrix) -> Bool {
        lhs.rows == rhs.rows
            && lhs.columns == rhs.columns
            && lhs.rowPointers == rhs.rowPointers
            && lhs.columnIndices == rhs.columnIndices
            && lhs.values == rhs.values
    }
}
