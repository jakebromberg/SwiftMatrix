/// Element-wise arithmetic for CSR sparse matrices.
///
/// All operations require operands to have the same dimensions. No broadcasting is performed.
/// Sparse + scalar and sparse - scalar are intentionally omitted because they would
/// produce dense results. Sparse / sparse is omitted because it would divide by implicit zeros.

/// Row-by-row two-pointer merge of two CSR matrices.
///
/// For entries present in both operands, `body` combines the values.
/// For entries in only the left operand, `lhsOnly` transforms the value.
/// For entries in only the right operand, `rhsOnly` transforms the value.
private func csrMerge<Element>(
    _ lhs: CSRMatrix<Element>,
    _ rhs: CSRMatrix<Element>,
    body: (Element, Element) -> Element,
    lhsOnly: (Element) -> Element,
    rhsOnly: (Element) -> Element
) -> CSRMatrix<Element> {
    precondition(lhs.rows == rhs.rows && lhs.columns == rhs.columns,
                 "Shape mismatch: [\(lhs.rows), \(lhs.columns)] vs [\(rhs.rows), \(rhs.columns)]")

    var rowPointers = [0]
    var columnIndices = [Int]()
    var values = [Element]()

    for row in 0..<lhs.rows {
        let aStart = lhs.rowPointers[row], aEnd = lhs.rowPointers[row + 1]
        let bStart = rhs.rowPointers[row], bEnd = rhs.rowPointers[row + 1]

        var i = aStart, j = bStart
        while i < aEnd && j < bEnd {
            let colA = lhs.columnIndices[i]
            let colB = rhs.columnIndices[j]

            if colA < colB {
                columnIndices.append(colA)
                values.append(lhsOnly(lhs.values[i]))
                i += 1
            } else if colA > colB {
                columnIndices.append(colB)
                values.append(rhsOnly(rhs.values[j]))
                j += 1
            } else {
                columnIndices.append(colA)
                values.append(body(lhs.values[i], rhs.values[j]))
                i += 1
                j += 1
            }
        }

        while i < aEnd {
            columnIndices.append(lhs.columnIndices[i])
            values.append(lhsOnly(lhs.values[i]))
            i += 1
        }

        while j < bEnd {
            columnIndices.append(rhs.columnIndices[j])
            values.append(rhsOnly(rhs.values[j]))
            j += 1
        }

        rowPointers.append(values.count)
    }

    return CSRMatrix(
        rows: lhs.rows, columns: lhs.columns,
        rowPointers: rowPointers, columnIndices: columnIndices, values: values
    )
}

/// Row-by-row two-pointer intersection of two CSR matrices (matching column indices only).
private func csrIntersect<Element>(
    _ lhs: CSRMatrix<Element>,
    _ rhs: CSRMatrix<Element>,
    body: (Element, Element) -> Element
) -> CSRMatrix<Element> {
    precondition(lhs.rows == rhs.rows && lhs.columns == rhs.columns,
                 "Shape mismatch: [\(lhs.rows), \(lhs.columns)] vs [\(rhs.rows), \(rhs.columns)]")

    var rowPointers = [0]
    var columnIndices = [Int]()
    var values = [Element]()

    for row in 0..<lhs.rows {
        let aStart = lhs.rowPointers[row], aEnd = lhs.rowPointers[row + 1]
        let bStart = rhs.rowPointers[row], bEnd = rhs.rowPointers[row + 1]

        var i = aStart, j = bStart
        while i < aEnd && j < bEnd {
            let colA = lhs.columnIndices[i]
            let colB = rhs.columnIndices[j]

            if colA < colB {
                i += 1
            } else if colA > colB {
                j += 1
            } else {
                columnIndices.append(colA)
                values.append(body(lhs.values[i], rhs.values[j]))
                i += 1
                j += 1
            }
        }

        rowPointers.append(values.count)
    }

    return CSRMatrix(
        rows: lhs.rows, columns: lhs.columns,
        rowPointers: rowPointers, columnIndices: columnIndices, values: values
    )
}

// MARK: - CSRMatrix + CSRMatrix

extension CSRMatrix where Element: AdditiveArithmetic {
    public static func + (lhs: CSRMatrix, rhs: CSRMatrix) -> CSRMatrix {
        csrMerge(lhs, rhs, body: +, lhsOnly: { $0 }, rhsOnly: { $0 })
    }

    public static func - (lhs: CSRMatrix, rhs: CSRMatrix) -> CSRMatrix {
        csrMerge(lhs, rhs, body: -, lhsOnly: { $0 }, rhsOnly: { .zero - $0 })
    }
}

extension CSRMatrix where Element: Numeric {
    public static func * (lhs: CSRMatrix, rhs: CSRMatrix) -> CSRMatrix {
        csrIntersect(lhs, rhs, body: *)
    }
}

extension CSRMatrix where Element: SignedNumeric {
    public static prefix func - (operand: CSRMatrix) -> CSRMatrix {
        CSRMatrix(
            rows: operand.rows, columns: operand.columns,
            rowPointers: operand.rowPointers,
            columnIndices: operand.columnIndices,
            values: operand.values.map { -$0 }
        )
    }
}

// MARK: - CSRMatrix + Scalar / Scalar + CSRMatrix

extension CSRMatrix where Element: Numeric {
    public static func * (lhs: CSRMatrix, rhs: Element) -> CSRMatrix {
        CSRMatrix(
            rows: lhs.rows, columns: lhs.columns,
            rowPointers: lhs.rowPointers,
            columnIndices: lhs.columnIndices,
            values: lhs.values.map { $0 * rhs }
        )
    }

    public static func * (lhs: Element, rhs: CSRMatrix) -> CSRMatrix {
        CSRMatrix(
            rows: rhs.rows, columns: rhs.columns,
            rowPointers: rhs.rowPointers,
            columnIndices: rhs.columnIndices,
            values: rhs.values.map { lhs * $0 }
        )
    }
}

extension CSRMatrix where Element: FloatingPoint {
    public static func / (lhs: CSRMatrix, rhs: Element) -> CSRMatrix {
        CSRMatrix(
            rows: lhs.rows, columns: lhs.columns,
            rowPointers: lhs.rowPointers,
            columnIndices: lhs.columnIndices,
            values: lhs.values.map { $0 / rhs }
        )
    }
}

// MARK: - Compound assignment

extension CSRMatrix where Element: AdditiveArithmetic {
    public static func += (lhs: inout CSRMatrix, rhs: CSRMatrix) {
        lhs = lhs + rhs
    }

    public static func -= (lhs: inout CSRMatrix, rhs: CSRMatrix) {
        lhs = lhs - rhs
    }
}

extension CSRMatrix where Element: Numeric {
    public static func *= (lhs: inout CSRMatrix, rhs: CSRMatrix) {
        lhs = lhs * rhs
    }
}
