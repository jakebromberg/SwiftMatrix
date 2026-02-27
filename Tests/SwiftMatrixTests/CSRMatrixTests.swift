import Testing
@testable import SwiftMatrix

struct CSRMatrixCreationTests {
    @Test func createEmpty() {
        let csr = CSRMatrix<Int>(rows: 3, columns: 4)
        #expect(csr.rows == 3)
        #expect(csr.columns == 4)
        #expect(csr.nnz == 0)
        #expect(csr.rowPointers == [0, 0, 0, 0])
        #expect(csr.columnIndices.isEmpty)
        #expect(csr.values.isEmpty)
    }

    @Test func createFromArrays() {
        // [[1, 0, 2],
        //  [0, 0, 3],
        //  [4, 0, 0]]
        let csr = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 2, 3, 4],
            columnIndices: [0, 2, 2, 0],
            values: [1, 2, 3, 4]
        )
        #expect(csr.rows == 3)
        #expect(csr.columns == 3)
        #expect(csr.nnz == 4)
        #expect(csr.rowPointers == [0, 2, 3, 4])
        #expect(csr.columnIndices == [0, 2, 2, 0])
        #expect(csr.values == [1, 2, 3, 4])
    }

    @Test func identityMatrix() {
        // 3x3 identity
        let csr = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 1, 2, 3],
            columnIndices: [0, 1, 2],
            values: [1, 1, 1]
        )
        #expect(csr.nnz == 3)
        #expect(csr.shape == [3, 3])
    }

    @Test func propertiesNnzRowsColumns() {
        let csr = CSRMatrix(
            rows: 4, columns: 5,
            rowPointers: [0, 2, 2, 3, 5],
            columnIndices: [0, 3, 2, 1, 4],
            values: [1, 2, 3, 4, 5]
        )
        #expect(csr.nnz == 5)
        #expect(csr.rows == 4)
        #expect(csr.columns == 5)
        #expect(csr.shape == [4, 5])
        #expect(csr.count == 20)
    }

    @Test func densityCalculation() {
        let csr = CSRMatrix(
            rows: 2, columns: 5,
            rowPointers: [0, 2, 3],
            columnIndices: [1, 3, 4],
            values: [10, 20, 30]
        )
        #expect(csr.density == 3.0 / 10.0)
    }

    @Test func emptyRows() {
        // Row 1 has no entries
        let csr = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 1, 1, 2],
            columnIndices: [0, 2],
            values: [1, 2]
        )
        #expect(csr.nnz == 2)
        // Row 1 range: rowPointers[1]..<rowPointers[2] == 1..<1 (empty)
        #expect(csr.rowPointers[1] == csr.rowPointers[2])
    }
}

struct CSRMatrixEquatableTests {
    @Test func equalMatrices() {
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1, 2]
        )
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1, 2]
        )
        #expect(a == b)
    }

    @Test func differentValues() {
        let a = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1, 2]
        )
        let b = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1, 99]
        )
        #expect(a != b)
    }

    @Test func differentStructure() {
        let a = CSRMatrix(
            rows: 2, columns: 3,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1, 2]
        )
        let b = CSRMatrix(
            rows: 2, columns: 3,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 2],
            values: [1, 2]
        )
        #expect(a != b)
    }
}
