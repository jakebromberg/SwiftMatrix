/// Sparse matrix-vector and matrix-matrix multiplication for ``CSRMatrix``.

extension CSRMatrix where Element: Numeric {
    /// Sparse matrix-vector multiplication (SpMV): CSR [m, n] * dense vector [n] -> dense [m].
    ///
    /// For each row of the sparse matrix, accumulates `value * vector[col]` over stored entries.
    ///
    /// ```swift
    /// let a = CSRMatrix(rows: 2, columns: 3,
    ///     rowPointers: [0, 2, 3], columnIndices: [0, 2, 1], values: [1, 2, 3])
    /// let x = Tensor(shape: [3], elements: [4, 5, 6])
    /// CSRMatrix.matvec(a, x)  // [16, 15]  (1*4+2*6, 3*5)
    /// ```
    ///
    /// - Parameters:
    ///   - matrix: A sparse matrix with shape [m, n].
    ///   - vector: A dense rank-1 tensor with shape [n].
    /// - Precondition: `vector` is rank-1 with length equal to `matrix.columns`.
    /// - Returns: A dense rank-1 tensor with shape [m].
    public static func matvec(_ matrix: CSRMatrix, _ vector: Tensor<Element>) -> Tensor<Element> {
        precondition(vector.rank == 1,
                     "matvec requires a rank-1 vector, got rank \(vector.rank)")
        precondition(vector.shape[0] == matrix.columns,
                     "Dimension mismatch: matrix has \(matrix.columns) columns, vector has \(vector.shape[0]) elements")

        let m = matrix.rows
        var result = Array(repeating: Element.zero, count: m)

        for row in 0..<m {
            var sum: Element = .zero
            for idx in matrix.rowPointers[row]..<matrix.rowPointers[row + 1] {
                sum += matrix.values[idx] * vector[matrix.columnIndices[idx]]
            }
            result[row] = sum
        }

        return Tensor(shape: [m], elements: result)
    }

    /// Sparse-dense matrix multiplication (SpMM): CSR [m, k] * Tensor [k, n] -> Tensor [m, n].
    ///
    /// For each nonzero entry `A[row, col]`, scatters `A[row, col] * B[col, j]` across all
    /// columns `j` of the result row.
    ///
    /// ```swift
    /// let a = CSRMatrix(rows: 2, columns: 2,
    ///     rowPointers: [0, 1, 2], columnIndices: [0, 1], values: [1, 2])
    /// let b = Tensor([[3, 4], [5, 6]])
    /// CSRMatrix.matmul(a, b)  // [[3, 4], [10, 12]]
    /// ```
    ///
    /// - Parameters:
    ///   - lhs: A sparse matrix with shape [m, k].
    ///   - rhs: A dense rank-2 tensor with shape [k, n].
    /// - Precondition: `rhs` is rank-2 with `rhs.shape[0] == lhs.columns`.
    /// - Returns: A dense rank-2 tensor with shape [m, n].
    public static func matmul(_ lhs: CSRMatrix, _ rhs: Tensor<Element>) -> Tensor<Element> {
        precondition(rhs.rank == 2,
                     "matmul requires a rank-2 tensor, got rank \(rhs.rank)")
        let k = lhs.columns
        let n = rhs.shape[1]
        precondition(rhs.shape[0] == k,
                     "Dimension mismatch: matrix has \(k) columns, rhs has \(rhs.shape[0]) rows")

        let m = lhs.rows
        var result = Array(repeating: Element.zero, count: m * n)

        for row in 0..<m {
            for idx in lhs.rowPointers[row]..<lhs.rowPointers[row + 1] {
                let col = lhs.columnIndices[idx]
                let aVal = lhs.values[idx]
                for j in 0..<n {
                    result[row * n + j] += aVal * rhs[col, j]
                }
            }
        }

        return Tensor(shape: [m, n], elements: result)
    }

    /// Sparse-sparse matrix multiplication (SpGEMM): CSR [m, k] * CSR [k, n] -> CSR [m, n].
    ///
    /// Uses a row-wise dense accumulator with a boolean marker array to track touched
    /// columns. Column indices are sorted before emission to maintain the CSR invariant.
    /// Explicit zeros from cancellation are kept, consistent with existing sparse arithmetic.
    ///
    /// ```swift
    /// let a = CSRMatrix(rows: 2, columns: 2,
    ///     rowPointers: [0, 1, 2], columnIndices: [0, 1], values: [1, 2])
    /// let b = CSRMatrix(rows: 2, columns: 2,
    ///     rowPointers: [0, 1, 2], columnIndices: [0, 1], values: [3, 4])
    /// CSRMatrix.matmul(a, b)  // diag([3, 8])
    /// ```
    ///
    /// - Parameters:
    ///   - lhs: A sparse matrix with shape [m, k].
    ///   - rhs: A sparse matrix with shape [k, n].
    /// - Precondition: `lhs.columns == rhs.rows`.
    /// - Returns: A sparse CSR matrix with shape [m, n].
    public static func matmul(_ lhs: CSRMatrix, _ rhs: CSRMatrix) -> CSRMatrix {
        let k = lhs.columns
        precondition(k == rhs.rows,
                     "Dimension mismatch: lhs has \(k) columns, rhs has \(rhs.rows) rows")

        let m = lhs.rows
        let n = rhs.columns

        var resultRowPointers = [Int]()
        resultRowPointers.reserveCapacity(m + 1)
        resultRowPointers.append(0)
        var resultColumnIndices = [Int]()
        var resultValues = [Element]()

        // Reusable dense accumulator and marker array
        var accumulator = Array(repeating: Element.zero, count: n)
        var marker = Array(repeating: false, count: n)
        var touchedColumns = [Int]()

        for row in 0..<m {
            touchedColumns.removeAll(keepingCapacity: true)

            for idx in lhs.rowPointers[row]..<lhs.rowPointers[row + 1] {
                let kCol = lhs.columnIndices[idx]
                let aVal = lhs.values[idx]

                for jIdx in rhs.rowPointers[kCol]..<rhs.rowPointers[kCol + 1] {
                    let j = rhs.columnIndices[jIdx]
                    if !marker[j] {
                        marker[j] = true
                        touchedColumns.append(j)
                    }
                    accumulator[j] += aVal * rhs.values[jIdx]
                }
            }

            // Sort columns to maintain CSR invariant
            touchedColumns.sort()

            // Emit nonzeros and reset
            for j in touchedColumns {
                resultColumnIndices.append(j)
                resultValues.append(accumulator[j])
                accumulator[j] = .zero
                marker[j] = false
            }

            resultRowPointers.append(resultColumnIndices.count)
        }

        return CSRMatrix(rows: m, columns: n, rowPointers: resultRowPointers,
                         columnIndices: resultColumnIndices, values: resultValues)
    }
}
