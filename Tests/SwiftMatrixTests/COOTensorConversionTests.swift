import Testing
@testable import SwiftMatrix

struct COOTensorDenseConversionTests {
    @Test func toDenseRank1() {
        let coo = COOTensor(
            shape: [5],
            indices: [[1, 3]],
            values: [3, 7]
        )
        let dense = coo.toTensor()
        #expect(dense == Tensor(shape: [5], elements: [0, 3, 0, 7, 0]))
    }

    @Test func toDenseRank2() {
        let coo = COOTensor(
            shape: [2, 3],
            indices: [[0, 1], [2, 0]],
            values: [5, 10]
        )
        let dense = coo.toTensor()
        #expect(dense == Tensor(shape: [2, 3], elements: [0, 0, 5, 10, 0, 0]))
    }

    @Test func toDenseRank3() {
        let coo = COOTensor(
            shape: [2, 2, 2],
            indices: [[0, 1], [1, 0], [0, 1]],
            values: [5, 10]
        )
        let dense = coo.toTensor()
        // (0,1,0) -> flat index 2, (1,0,1) -> flat index 5
        #expect(dense == Tensor(shape: [2, 2, 2], elements: [0, 0, 5, 0, 0, 10, 0, 0]))
    }

    @Test(arguments: [
        (
            shape: [5],
            elements: [0, 3, 0, 0, 7],
            label: "rank-1 sparse vector"
        ),
        (
            shape: [3, 4],
            elements: [1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0],
            label: "rank-2 sparse matrix"
        ),
        (
            shape: [2, 2],
            elements: [0, 0, 0, 0],
            label: "rank-2 all zeros"
        ),
        (
            shape: [2, 2],
            elements: [1, 2, 3, 4],
            label: "rank-2 fully dense"
        ),
    ])
    func roundTrip(shape: [Int], elements: [Int], label: String) {
        let original = Tensor(shape: shape, elements: elements)
        let coo = COOTensor(from: original)
        let reconstructed = coo.toTensor()
        #expect(reconstructed == original)
    }

    @Test func emptyTensor() {
        let coo = COOTensor<Int>(shape: [0, 3])
        let dense = coo.toTensor()
        #expect(dense.shape == [0, 3])
        #expect(dense.count == 0)
    }

    @Test func allZeroTensor() {
        let original = Tensor(shape: [2, 2], repeating: 0)
        let coo = COOTensor(from: original)
        #expect(coo.nnz == 0)
        #expect(coo.toTensor() == original)
    }

    @Test func fullyDenseTensor() {
        let original = Tensor(shape: [2, 2], elements: [1, 2, 3, 4])
        let coo = COOTensor(from: original)
        #expect(coo.nnz == 4)
        #expect(coo.toTensor() == original)
    }

    @Test func toDenseWithCustomDefault() {
        let coo = COOTensor(
            shape: [3],
            indices: [[1]],
            values: [5]
        )
        let dense = coo.toTensor(defaultValue: -1)
        #expect(dense == Tensor(shape: [3], elements: [-1, 5, -1]))
    }

    @Test func nonContiguousTensorInput() {
        // Create a 2x3 tensor and transpose it to 3x2
        let original = Tensor([[1, 0, 3], [0, 5, 0]])
        let transposed = original.transposed()
        // transposed: [[1, 0], [0, 5], [3, 0]]
        let coo = COOTensor(from: transposed)
        #expect(coo.shape == [3, 2])
        // Nonzeros: (0,0)=1, (1,1)=5, (2,0)=3
        #expect(coo.nnz == 3)
        #expect(coo.toTensor() == Tensor(shape: [3, 2], elements: [1, 0, 0, 5, 3, 0]))
    }
}

struct COOTensorCSRConversionTests {
    @Test func fromCSRMatrix() {
        let csr = CSRMatrix(
            rows: 3, columns: 3,
            rowPointers: [0, 2, 3, 4],
            columnIndices: [0, 2, 2, 0],
            values: [1, 2, 3, 4]
        )
        let coo = COOTensor(from: csr)
        #expect(coo.shape == [3, 3])
        #expect(coo.nnz == 4)
        #expect(coo.indices == [[0, 0, 1, 2], [0, 2, 2, 0]])
        #expect(coo.values == [1, 2, 3, 4])
    }

    @Test func roundTripCOOToCSRToCOO() {
        let original = COOTensor(
            shape: [3, 4],
            indices: [[0, 0, 1, 2, 2], [0, 3, 2, 1, 3]],
            values: [1, 2, 3, 4, 5]
        )
        let csr = CSRMatrix(from: original)
        let reconstructed = COOTensor(from: csr)
        #expect(reconstructed == original)
    }
}
