import Testing
@testable import SwiftMatrix

struct CSRMatrixDenseConversionTests {
    @Test func toDenseSmall() {
        // [[1, 0, 2],
        //  [0, 0, 3],
        //  [4, 0, 0]]
        let csr = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 2, 3, 4],
            columnIndices: [0, 2, 2, 0],
            values: [1, 2, 3, 4]
        )
        let dense = csr.toTensor()
        #expect(dense == Tensor([[1, 0, 2], [0, 0, 3], [4, 0, 0]]))
    }

    @Test func toDenseIdentity() {
        let csr = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 1, 2, 3],
            columnIndices: [0, 1, 2],
            values: [1, 1, 1]
        )
        let dense = csr.toTensor()
        #expect(dense == Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    }

    @Test(arguments: [
        (
            elements: [[1, 0, 0, 2], [0, 0, 3, 0], [0, 4, 0, 0]],
            label: "sparse"
        ),
        (
            elements: [[0, 0], [0, 0]],
            label: "all zeros"
        ),
        (
            elements: [[1, 2], [3, 4]],
            label: "fully dense"
        ),
        (
            elements: [[0, 0, 7, 0, 0]],
            label: "single row"
        ),
    ])
    func roundTrip(elements: [[Int]], label: String) {
        let original = Tensor(elements)
        let csr = CSRMatrix(from: original)
        let reconstructed = csr.toTensor()
        #expect(reconstructed == original)
    }

    @Test func emptyMatrix() {
        let csr = CSRMatrix<Int>(rows: 0, columns: 3)
        let dense = csr.toTensor()
        #expect(dense.shape == [0, 3])
        #expect(dense.count == 0)
    }
}

struct CSRMatrixCOOConversionTests {
    @Test func fromCOOTensor() {
        let coo = COOTensor(
            shape: [3, 3],
            indices: [[0, 1, 2], [0, 2, 1]],
            values: [1, 2, 3]
        )
        let csr = CSRMatrix(from: coo)
        #expect(csr.rows == 3)
        #expect(csr.columns == 3)
        #expect(csr.rowPointers == [0, 1, 2, 3])
        #expect(csr.columnIndices == [0, 2, 1])
        #expect(csr.values == [1, 2, 3])
    }

    @Test func roundTripCSRToCOOToCSR() {
        let original = CSRMatrix(
            rows: 3, columns: 4,
            rowPointers: [0, 2, 3, 5],
            columnIndices: [0, 3, 2, 1, 3],
            values: [1, 2, 3, 4, 5]
        )
        let coo = COOTensor(from: original)
        let reconstructed = CSRMatrix(from: coo)
        #expect(reconstructed == original)
    }
}
