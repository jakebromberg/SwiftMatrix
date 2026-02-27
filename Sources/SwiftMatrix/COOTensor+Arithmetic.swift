/// Element-wise arithmetic for COO sparse tensors.
///
/// All operations require operands to have the same shape. No broadcasting is performed.
/// Sparse + scalar and sparse - scalar are intentionally omitted because they would
/// produce dense results. Sparse / sparse is omitted because it would divide by implicit zeros.

/// Computes a scalar linear index for one COO entry using row-major strides.
private func cooLinearIndex<Element>(
    _ tensor: COOTensor<Element>, entry: Int, strides: [Int]
) -> Int {
    var index = 0
    for axis in 0..<tensor.rank {
        index += tensor.indices[axis][entry] * strides[axis]
    }
    return index
}

/// Two-pointer merge of two COO tensors, producing a sorted COO result.
///
/// For entries present in both operands, `body` combines the values.
/// For entries in only the left operand, `lhsOnly` transforms the value.
/// For entries in only the right operand, `rhsOnly` transforms the value.
private func cooMerge<Element>(
    _ lhs: COOTensor<Element>,
    _ rhs: COOTensor<Element>,
    body: (Element, Element) -> Element,
    lhsOnly: (Element) -> Element,
    rhsOnly: (Element) -> Element
) -> COOTensor<Element> {
    precondition(lhs.shape == rhs.shape,
                 "Shape mismatch: \(lhs.shape) vs \(rhs.shape)")

    let strides = Tensor<Element>.computeStrides(for: lhs.shape)
    let rank = lhs.rank

    var resultIndices = Array(repeating: [Int](), count: rank)
    var resultValues = [Element]()

    var i = 0, j = 0
    while i < lhs.nnz && j < rhs.nnz {
        let li = cooLinearIndex(lhs, entry: i, strides: strides)
        let lj = cooLinearIndex(rhs, entry: j, strides: strides)

        if li < lj {
            for axis in 0..<rank {
                resultIndices[axis].append(lhs.indices[axis][i])
            }
            resultValues.append(lhsOnly(lhs.values[i]))
            i += 1
        } else if li > lj {
            for axis in 0..<rank {
                resultIndices[axis].append(rhs.indices[axis][j])
            }
            resultValues.append(rhsOnly(rhs.values[j]))
            j += 1
        } else {
            for axis in 0..<rank {
                resultIndices[axis].append(lhs.indices[axis][i])
            }
            resultValues.append(body(lhs.values[i], rhs.values[j]))
            i += 1
            j += 1
        }
    }

    while i < lhs.nnz {
        for axis in 0..<rank {
            resultIndices[axis].append(lhs.indices[axis][i])
        }
        resultValues.append(lhsOnly(lhs.values[i]))
        i += 1
    }

    while j < rhs.nnz {
        for axis in 0..<rank {
            resultIndices[axis].append(rhs.indices[axis][j])
        }
        resultValues.append(rhsOnly(rhs.values[j]))
        j += 1
    }

    return COOTensor(shape: lhs.shape, sortedIndices: resultIndices, values: resultValues)
}

/// Two-pointer intersection of two COO tensors (matching indices only).
private func cooIntersect<Element>(
    _ lhs: COOTensor<Element>,
    _ rhs: COOTensor<Element>,
    body: (Element, Element) -> Element
) -> COOTensor<Element> {
    precondition(lhs.shape == rhs.shape,
                 "Shape mismatch: \(lhs.shape) vs \(rhs.shape)")

    let strides = Tensor<Element>.computeStrides(for: lhs.shape)
    let rank = lhs.rank

    var resultIndices = Array(repeating: [Int](), count: rank)
    var resultValues = [Element]()

    var i = 0, j = 0
    while i < lhs.nnz && j < rhs.nnz {
        let li = cooLinearIndex(lhs, entry: i, strides: strides)
        let lj = cooLinearIndex(rhs, entry: j, strides: strides)

        if li < lj {
            i += 1
        } else if li > lj {
            j += 1
        } else {
            for axis in 0..<rank {
                resultIndices[axis].append(lhs.indices[axis][i])
            }
            resultValues.append(body(lhs.values[i], rhs.values[j]))
            i += 1
            j += 1
        }
    }

    return COOTensor(shape: lhs.shape, sortedIndices: resultIndices, values: resultValues)
}

// MARK: - COOTensor + COOTensor

extension COOTensor where Element: AdditiveArithmetic {
    public static func + (lhs: COOTensor, rhs: COOTensor) -> COOTensor {
        cooMerge(lhs, rhs, body: +, lhsOnly: { $0 }, rhsOnly: { $0 })
    }

    public static func - (lhs: COOTensor, rhs: COOTensor) -> COOTensor {
        cooMerge(lhs, rhs, body: -, lhsOnly: { $0 }, rhsOnly: { .zero - $0 })
    }
}

extension COOTensor where Element: Numeric {
    public static func * (lhs: COOTensor, rhs: COOTensor) -> COOTensor {
        cooIntersect(lhs, rhs, body: *)
    }
}

extension COOTensor where Element: SignedNumeric {
    public static prefix func - (operand: COOTensor) -> COOTensor {
        COOTensor(shape: operand.shape, sortedIndices: operand.indices, values: operand.values.map { -$0 })
    }
}

// MARK: - COOTensor + Scalar / Scalar + COOTensor

extension COOTensor where Element: Numeric {
    public static func * (lhs: COOTensor, rhs: Element) -> COOTensor {
        COOTensor(shape: lhs.shape, sortedIndices: lhs.indices, values: lhs.values.map { $0 * rhs })
    }

    public static func * (lhs: Element, rhs: COOTensor) -> COOTensor {
        COOTensor(shape: rhs.shape, sortedIndices: rhs.indices, values: rhs.values.map { lhs * $0 })
    }
}

extension COOTensor where Element: FloatingPoint {
    public static func / (lhs: COOTensor, rhs: Element) -> COOTensor {
        COOTensor(shape: lhs.shape, sortedIndices: lhs.indices, values: lhs.values.map { $0 / rhs })
    }
}

// MARK: - Compound assignment

extension COOTensor where Element: AdditiveArithmetic {
    public static func += (lhs: inout COOTensor, rhs: COOTensor) {
        lhs = lhs + rhs
    }

    public static func -= (lhs: inout COOTensor, rhs: COOTensor) {
        lhs = lhs - rhs
    }
}

extension COOTensor where Element: Numeric {
    public static func *= (lhs: inout COOTensor, rhs: COOTensor) {
        lhs = lhs * rhs
    }
}
