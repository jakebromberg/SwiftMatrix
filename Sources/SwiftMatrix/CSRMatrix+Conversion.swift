/// Conversions between `CSRMatrix` and dense `Tensor`.
extension CSRMatrix where Element: AdditiveArithmetic & Equatable {
    /// Creates a CSR matrix from a dense rank-2 tensor, storing only nonzero entries.
    ///
    /// - Parameter tensor: A rank-2 tensor to convert.
    /// - Precondition: `tensor.rank == 2`.
    public init(from tensor: Tensor<Element>) {
        precondition(tensor.rank == 2, "CSRMatrix requires a rank-2 tensor, got rank \(tensor.rank)")

        let rows = tensor.shape[0]
        let cols = tensor.shape[1]

        var rowPointers = [Int]()
        rowPointers.reserveCapacity(rows + 1)
        var columnIndices = [Int]()
        var values = [Element]()

        rowPointers.append(0)
        for row in 0..<rows {
            for col in 0..<cols {
                let element = tensor[row, col]
                if element != .zero {
                    columnIndices.append(col)
                    values.append(element)
                }
            }
            rowPointers.append(columnIndices.count)
        }

        self.init(rows: rows, columns: cols, rowPointers: rowPointers,
                  columnIndices: columnIndices, values: values)
    }
}

/// Conversion from COO format.
extension CSRMatrix {
    /// Creates a CSR matrix from a rank-2 COO tensor.
    ///
    /// Since COO entries are already sorted in row-major order, this builds the row pointer
    /// array by scanning the sorted row indices.
    ///
    /// - Parameter coo: A rank-2 COO tensor to convert.
    /// - Precondition: `coo.rank == 2`.
    public init(from coo: COOTensor<Element>) {
        precondition(coo.rank == 2, "CSRMatrix requires a rank-2 COO tensor, got rank \(coo.rank)")

        let rows = coo.shape[0]
        let cols = coo.shape[1]
        let rowIndices = coo.indices[0]
        let colIndices = coo.indices[1]

        var rowPointers = Array(repeating: 0, count: rows + 1)

        // Count entries per row
        for row in rowIndices {
            rowPointers[row + 1] += 1
        }
        // Cumulative sum
        for i in 1...rows {
            rowPointers[i] += rowPointers[i - 1]
        }

        self.init(rows: rows, columns: cols, rowPointers: rowPointers,
                  columnIndices: colIndices, values: coo.values)
    }
}

extension CSRMatrix {
    /// Converts this CSR matrix to a dense rank-2 tensor, filling unstored positions
    /// with `defaultValue`.
    ///
    /// - Parameter defaultValue: The value for positions not in the sparse representation.
    /// - Returns: A dense rank-2 `Tensor` with shape `[rows, columns]`.
    public func toTensor(defaultValue: Element) -> Tensor<Element> {
        var elements = Array(repeating: defaultValue, count: rows * columns)

        for row in 0..<rows {
            for idx in rowPointers[row]..<rowPointers[row + 1] {
                let col = columnIndices[idx]
                elements[row * columns + col] = values[idx]
            }
        }

        return Tensor(shape: [rows, columns], elements: elements)
    }
}

extension CSRMatrix where Element: AdditiveArithmetic {
    /// Converts this CSR matrix to a dense rank-2 tensor, filling unstored positions
    /// with `.zero`.
    ///
    /// - Returns: A dense rank-2 `Tensor` with shape `[rows, columns]`.
    public func toTensor() -> Tensor<Element> {
        toTensor(defaultValue: .zero)
    }
}
