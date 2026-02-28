import Testing
@testable import SwiftMatrix

struct CSRMatrixSumTests {
    @Test func basicSum() {
        // [[1, 0, 2], [0, 0, 3], [4, 0, 0]]
        let csr = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 2, 3, 4],
            columnIndices: [0, 2, 2, 0],
            values: [1, 2, 3, 4]
        )
        #expect(csr.sum() == 10)
    }

    @Test func emptyMatrix() {
        let csr = CSRMatrix<Int>(rows: 3, columns: 3)
        #expect(csr.sum() == 0)
    }

    @Test func matchesDense() {
        let csr = CSRMatrix(
            rows: 2, columns: 3,
            rowPointers: [0, 2, 3],
            columnIndices: [0, 2, 1],
            values: [10, 20, 30]
        )
        #expect(csr.sum() == csr.toTensor().sum())
    }
}

struct CSRMatrixSumAxisTests {
    @Test func sumAxis0() {
        // [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
        // column sums: [5, 3, 7]
        let csr = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 2, 3, 5],
            columnIndices: [0, 2, 1, 0, 2],
            values: [1, 2, 3, 4, 5]
        )
        let result = csr.sum(axis: 0)
        #expect(result.shape == [3])
        #expect(Array(result) == [5, 3, 7])
    }

    @Test func sumAxis1() {
        // [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
        // row sums: [3, 3, 9]
        let csr = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 2, 3, 5],
            columnIndices: [0, 2, 1, 0, 2],
            values: [1, 2, 3, 4, 5]
        )
        let result = csr.sum(axis: 1)
        #expect(result.shape == [3])
        #expect(Array(result) == [3, 3, 9])
    }

    @Test func emptyMatrix() {
        let csr = CSRMatrix<Int>(rows: 2, columns: 3)
        let axis0 = csr.sum(axis: 0)
        #expect(axis0.shape == [3])
        #expect(Array(axis0) == [0, 0, 0])

        let axis1 = csr.sum(axis: 1)
        #expect(axis1.shape == [2])
        #expect(Array(axis1) == [0, 0])
    }

    @Test func emptyRows() {
        // Row 1 is empty: [[1, 2], [0, 0], [3, 0]]
        let csr = CSRMatrix(
            rows: 3, columns: 2,
            rowPointers: [0, 2, 2, 3],
            columnIndices: [0, 1, 0],
            values: [1, 2, 3]
        )
        let axis1 = csr.sum(axis: 1)
        #expect(Array(axis1) == [3, 0, 3])
    }

    @Test func matchesDenseAxis0() {
        let csr = CSRMatrix(
            rows: 3, columns: 4,
            rowPointers: [0, 2, 4, 5],
            columnIndices: [1, 3, 0, 2, 1],
            values: [10, 20, 30, 40, 50]
        )
        let expected = csr.toTensor().sum(axis: 0)
        let result = csr.sum(axis: 0)
        #expect(Array(result) == Array(expected))
    }

    @Test func matchesDenseAxis1() {
        let csr = CSRMatrix(
            rows: 3, columns: 4,
            rowPointers: [0, 2, 4, 5],
            columnIndices: [1, 3, 0, 2, 1],
            values: [10, 20, 30, 40, 50]
        )
        let expected = csr.toTensor().sum(axis: 1)
        let result = csr.sum(axis: 1)
        #expect(Array(result) == Array(expected))
    }

    @Test func singleRow() {
        let csr = CSRMatrix(
            rows: 1, columns: 4,
            rowPointers: [0, 2],
            columnIndices: [1, 3],
            values: [5, 10]
        )
        let axis0 = csr.sum(axis: 0)
        #expect(Array(axis0) == [0, 5, 0, 10])

        let axis1 = csr.sum(axis: 1)
        #expect(Array(axis1) == [15])
    }

    @Test func singleColumn() {
        let csr = CSRMatrix(
            rows: 3, columns: 1,
            rowPointers: [0, 1, 1, 1],
            columnIndices: [0],
            values: [7]
        )
        let axis0 = csr.sum(axis: 0)
        #expect(Array(axis0) == [7])

        let axis1 = csr.sum(axis: 1)
        #expect(Array(axis1) == [7, 0, 0])
    }
}

struct CSRMatrixMeanTests {
    @Test func basicMean() {
        // [[1, 0], [0, 3]] -> sum=4, count=4 -> mean=1.0
        let csr = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [1.0, 3.0]
        )
        #expect(csr.mean() == 1.0)
    }

    @Test func emptyMatrix() {
        let csr = CSRMatrix<Double>(rows: 2, columns: 2)
        #expect(csr.mean() == 0.0)
    }

    @Test func matchesDense() {
        let csr = CSRMatrix(
            rows: 2, columns: 3,
            rowPointers: [0, 2, 3],
            columnIndices: [0, 2, 1],
            values: [6.0, 3.0, 9.0]
        )
        #expect(csr.mean() == csr.toTensor().mean())
    }
}

struct CSRMatrixMeanAxisTests {
    @Test func meanAxis0() {
        // [[2, 0], [0, 4]] -> col means: [1.0, 2.0]
        let csr = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [2.0, 4.0]
        )
        let result = csr.mean(axis: 0)
        #expect(result.shape == [2])
        #expect(Array(result) == [1.0, 2.0])
    }

    @Test func meanAxis1() {
        // [[2, 0], [0, 4]] -> row means: [1.0, 2.0]
        let csr = CSRMatrix(
            rows: 2, columns: 2,
            rowPointers: [0, 1, 2],
            columnIndices: [0, 1],
            values: [2.0, 4.0]
        )
        let result = csr.mean(axis: 1)
        #expect(result.shape == [2])
        #expect(Array(result) == [1.0, 2.0])
    }

    @Test func matchesDenseAxis0() {
        let csr = CSRMatrix(
            rows: 3, columns: 2,
            rowPointers: [0, 1, 2, 3],
            columnIndices: [0, 1, 0],
            values: [3.0, 6.0, 9.0]
        )
        let expected = csr.toTensor().mean(axis: 0)
        let result = csr.mean(axis: 0)
        #expect(Array(result) == Array(expected))
    }

    @Test func matchesDenseAxis1() {
        let csr = CSRMatrix(
            rows: 3, columns: 2,
            rowPointers: [0, 1, 2, 3],
            columnIndices: [0, 1, 0],
            values: [3.0, 6.0, 9.0]
        )
        let expected = csr.toTensor().mean(axis: 1)
        let result = csr.mean(axis: 1)
        #expect(Array(result) == Array(expected))
    }

    @Test func emptyMatrix() {
        let csr = CSRMatrix<Double>(rows: 2, columns: 3)
        let axis0 = csr.mean(axis: 0)
        #expect(Array(axis0) == [0.0, 0.0, 0.0])

        let axis1 = csr.mean(axis: 1)
        #expect(Array(axis1) == [0.0, 0.0])
    }
}
