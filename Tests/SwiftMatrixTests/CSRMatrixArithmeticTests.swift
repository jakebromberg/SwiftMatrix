import Testing
@testable import SwiftMatrix

struct CSRMatrixAdditionTests {
    @Test func sameStructure() {
        // [[1, 0, 2], [0, 0, 3], [4, 0, 0]]
        let a = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 2, 3, 4],
            columnIndices: [0, 2, 2, 0],
            values: [1, 2, 3, 4]
        )
        let b = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 2, 3, 4],
            columnIndices: [0, 2, 2, 0],
            values: [10, 20, 30, 40]
        )
        let result = a + b
        #expect(result.rows == 3)
        #expect(result.columns == 3)
        #expect(result.rowPointers == [0, 2, 3, 4])
        #expect(result.columnIndices == [0, 2, 2, 0])
        #expect(result.values == [11, 22, 33, 44])
    }

    @Test func differentStructure() {
        // a: [[1, 0], [0, 0]]
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 1],
            columnIndices: [0],
            values: [1]
        )
        // b: [[0, 0], [0, 2]]
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 0, 1],
            columnIndices: [1],
            values: [2]
        )
        let result = a + b
        #expect(result.rowPointers == [0, 1, 2])
        #expect(result.columnIndices == [0, 1])
        #expect(result.values == [1, 2])
    }

    @Test func mixedStructure() {
        // a: [[1, 0, 2], [0, 3, 0]]
        let a = CSRMatrix(
            rows: 2, columns: 3,
            rowPointers: [0, 2, 3],
            columnIndices: [0, 2, 1],
            values: [1, 2, 3]
        )
        // b: [[0, 4, 2], [5, 0, 0]]
        let b = CSRMatrix(
            rows: 2, columns: 3,
            rowPointers: [0, 2, 3],
            columnIndices: [1, 2, 0],
            values: [4, 2, 5]
        )
        let result = a + b
        #expect(result.rowPointers == [0, 3, 5])
        #expect(result.columnIndices == [0, 1, 2, 0, 1])
        #expect(result.values == [1, 4, 4, 5, 3])
    }

    @Test func emptyOperand() {
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [10, 20]
        )
        let b = CSRMatrix<Int>(rows: 2, columns: 2)
        let result = a + b
        #expect(result == a)
    }
}

struct CSRMatrixSubtractionTests {
    @Test func sameStructure() {
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [10, 20]
        )
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1, 2]
        )
        let result = a - b
        #expect(result.values == [9, 18])
    }

    @Test func cancellation() {
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [5, 10]
        )
        let result = a - a
        #expect(result.columnIndices == [0, 1])
        #expect(result.values == [0, 0])
    }

    @Test func differentStructure() {
        // a: [[10, 0], [0, 0]]
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 1],
            columnIndices: [0],
            values: [10]
        )
        // b: [[0, 0], [0, 5]]
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 0, 1],
            columnIndices: [1],
            values: [5]
        )
        let result = a - b
        #expect(result.rowPointers == [0, 1, 2])
        #expect(result.columnIndices == [0, 1])
        #expect(result.values == [10, -5])
    }
}

struct CSRMatrixMultiplicationTests {
    @Test func sameStructure() {
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [10, 20]
        )
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [2, 3]
        )
        let result = a * b
        #expect(result.values == [20, 60])
    }

    @Test func disjoint() {
        // a: [[1, 0], [0, 0]]
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 1],
            columnIndices: [0],
            values: [1]
        )
        // b: [[0, 2], [0, 0]]
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 1],
            columnIndices: [1],
            values: [2]
        )
        let result = a * b
        #expect(result.nnz == 0)
    }

    @Test func partialOverlap() {
        // a: [[1, 2], [3, 0]]
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 2, 3],
            columnIndices: [0, 1, 0],
            values: [1, 2, 3]
        )
        // b: [[0, 4], [0, 5]]
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 1],
            columnIndices: [1],
            values: [4]
        )
        let result = a * b
        #expect(result.rowPointers == [0, 1, 1])
        #expect(result.columnIndices == [1])
        #expect(result.values == [8])
    }
}

struct CSRMatrixScalarTests {
    @Test func multiply() {
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [10, 20]
        )
        let result = a * 3
        #expect(result.values == [30, 60])
        #expect(result.columnIndices == [0, 1])
    }

    @Test func multiplyCommutative() {
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [10, 20]
        )
        #expect((a * 3).values == (3 * a).values)
    }

    @Test func divide() {
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [10.0, 20.0]
        )
        let result = a / 2.0
        #expect(result.values == [5.0, 10.0])
    }
}

struct CSRMatrixNegationTests {
    @Test func negation() {
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [10, -20]
        )
        let result = -a
        #expect(result.values == [-10, 20])
        #expect(result.columnIndices == [0, 1])
    }

    @Test func negationEmpty() {
        let a = CSRMatrix<Int>(rows: 2, columns: 2)
        let result = -a
        #expect(result.nnz == 0)
    }
}

struct CSRMatrixCompoundAssignmentTests {
    @Test func plusEquals() {
        var a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [10, 20]
        )
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1, 2]
        )
        a += b
        #expect(a.values == [11, 22])
    }

    @Test func minusEquals() {
        var a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [10, 20]
        )
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1, 2]
        )
        a -= b
        #expect(a.values == [9, 18])
    }

    @Test func timesEquals() {
        var a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [10, 20]
        )
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [2, 3]
        )
        a *= b
        #expect(a.values == [20, 60])
    }
}

struct CSRMatrixEmptyRowTests {
    @Test func addWithEmptyRows() {
        // Row 1 is empty in both
        let a = CSRMatrix(
            rows: 3, columns: 2,
            rowPointers: [0, 1, 1, 2],
            columnIndices: [0, 1],
            values: [10, 20]
        )
        let b = CSRMatrix(
            rows: 3, columns: 2,
            rowPointers: [0, 0, 0, 1],
            columnIndices: [0],
            values: [5]
        )
        let result = a + b
        #expect(result.rowPointers == [0, 1, 1, 3])
        #expect(result.columnIndices == [0, 0, 1])
        #expect(result.values == [10, 5, 20])
    }

    @Test func multiplyWithEmptyRows() {
        // a has entries in rows 0 and 2; b has entries in rows 1 and 2
        let a = CSRMatrix(
            rows: 3, columns: 2,
            rowPointers: [0, 1, 1, 2],
            columnIndices: [0, 1],
            values: [10, 20]
        )
        let b = CSRMatrix(
            rows: 3, columns: 2,
            rowPointers: [0, 0, 1, 2],
            columnIndices: [0, 1],
            values: [5, 3]
        )
        // Only row 2 has overlap (col 1)
        let result = a * b
        #expect(result.rowPointers == [0, 0, 0, 1])
        #expect(result.columnIndices == [1])
        #expect(result.values == [60])
    }
}
