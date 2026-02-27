/// Conversions between `COOTensor` and dense `Tensor`.
extension COOTensor where Element: AdditiveArithmetic & Equatable {
    /// Creates a COO tensor from a dense tensor, storing only nonzero entries.
    ///
    /// Iterates the tensor in row-major order and records any element that is not `.zero`.
    /// Works correctly for both contiguous and non-contiguous (e.g., transposed) tensors.
    ///
    /// - Parameter tensor: The dense tensor to convert.
    public init(from tensor: Tensor<Element>) {
        let shape = tensor.shape
        let rank = shape.count

        var indices = Array(repeating: [Int](), count: rank)
        var values = [Element]()

        let strides = Tensor<Element>.computeStrides(for: shape)
        let count = shape.reduce(1, *)

        for linearIndex in 0..<count {
            let element = tensor[linearIndex: linearIndex]
            guard element != .zero else { continue }

            // Unravel linear index into multi-dimensional coordinates
            var remaining = linearIndex
            for axis in 0..<rank {
                let coord: Int
                if strides[axis] == 0 {
                    coord = 0
                } else {
                    coord = remaining / strides[axis]
                    remaining %= strides[axis]
                }
                indices[axis].append(coord)
            }
            values.append(element)
        }

        // Already in row-major order since we iterated linearly
        self.init(shape: shape, sortedIndices: indices, values: values)
    }
}

/// Conversion from CSR format.
extension COOTensor {
    /// Creates a rank-2 COO tensor from a CSR matrix.
    ///
    /// Expands the row pointer array into per-entry row indices. The resulting entries
    /// are already in row-major order since CSR stores rows sequentially.
    ///
    /// - Parameter csr: The CSR matrix to convert.
    public init(from csr: CSRMatrix<Element>) {
        let shape = [csr.rows, csr.columns]
        var rowIndices = [Int]()
        rowIndices.reserveCapacity(csr.nnz)

        for row in 0..<csr.rows {
            let count = csr.rowPointers[row + 1] - csr.rowPointers[row]
            rowIndices.append(contentsOf: Array(repeating: row, count: count))
        }

        self.init(shape: shape, sortedIndices: [rowIndices, csr.columnIndices],
                  values: csr.values)
    }
}

extension COOTensor {
    /// Converts this COO tensor to a dense tensor, filling unstored positions with `defaultValue`.
    ///
    /// - Parameter defaultValue: The value for positions not in the sparse representation.
    /// - Returns: A dense `Tensor` with the same shape.
    public func toTensor(defaultValue: Element) -> Tensor<Element> {
        let count = shape.reduce(1, *)
        var elements = Array(repeating: defaultValue, count: count)
        let strides = Tensor<Element>.computeStrides(for: shape)

        for entry in 0..<nnz {
            var flatIndex = 0
            for axis in 0..<rank {
                flatIndex += indices[axis][entry] * strides[axis]
            }
            elements[flatIndex] = values[entry]
        }

        return Tensor(shape: shape, elements: elements)
    }
}

extension COOTensor where Element: AdditiveArithmetic {
    /// Converts this COO tensor to a dense tensor, filling unstored positions with `.zero`.
    ///
    /// - Returns: A dense `Tensor` with the same shape.
    public func toTensor() -> Tensor<Element> {
        toTensor(defaultValue: .zero)
    }
}
