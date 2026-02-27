import Testing
@testable import SwiftMatrix

struct COOTensorCreationTests {
    @Test func createEmpty() {
        let coo = COOTensor<Int>(shape: [3, 4])
        #expect(coo.shape == [3, 4])
        #expect(coo.nnz == 0)
        #expect(coo.indices.count == 2)
        #expect(coo.indices[0].isEmpty)
        #expect(coo.indices[1].isEmpty)
        #expect(coo.values.isEmpty)
    }

    @Test func createFromIndicesAndValues() {
        let coo = COOTensor(
            shape: [3, 4],
            indices: [[0, 1, 2], [1, 2, 0]],
            values: [10, 20, 30]
        )
        #expect(coo.shape == [3, 4])
        #expect(coo.nnz == 3)
        #expect(coo.indices == [[0, 1, 2], [1, 2, 0]])
        #expect(coo.values == [10, 20, 30])
    }

    @Test func sortsEntriesByRowMajorOrder() {
        // Pass entries in reverse order
        let coo = COOTensor(
            shape: [3, 3],
            indices: [[2, 0, 1], [1, 2, 0]],
            values: [30, 10, 20]
        )
        // Should be sorted to (0,2), (1,0), (2,1)
        #expect(coo.indices == [[0, 1, 2], [2, 0, 1]])
        #expect(coo.values == [10, 20, 30])
    }

    @Test func sumsDuplicateEntries() {
        // Two entries at (1, 2) with values 10 and 25
        let coo = COOTensor(
            shape: [3, 3],
            indices: [[1, 0, 1], [2, 0, 2]],
            values: [10, 5, 25]
        )
        // After sorting and summing: (0,0)=5, (1,2)=35
        #expect(coo.nnz == 2)
        #expect(coo.indices == [[0, 1], [0, 2]])
        #expect(coo.values == [5, 35])
    }

    @Test func propertiesNnzRankCountDensity() {
        let coo = COOTensor(
            shape: [4, 5],
            indices: [[0, 2, 3], [1, 3, 0]],
            values: [1, 2, 3]
        )
        #expect(coo.nnz == 3)
        #expect(coo.rank == 2)
        #expect(coo.count == 20)
        #expect(coo.density == 3.0 / 20.0)
    }

    @Test func rank1Sparse() {
        let coo = COOTensor(
            shape: [5],
            indices: [[1, 3]],
            values: [10, 20]
        )
        #expect(coo.rank == 1)
        #expect(coo.nnz == 2)
        #expect(coo.indices == [[1, 3]])
        #expect(coo.values == [10, 20])
    }

    @Test func rank3Sparse() {
        let coo = COOTensor(
            shape: [2, 3, 4],
            indices: [[0, 1], [2, 0], [3, 1]],
            values: [10, 20]
        )
        #expect(coo.rank == 3)
        #expect(coo.nnz == 2)
        #expect(coo.shape == [2, 3, 4])
    }
}

struct COOTensorEquatableTests {
    @Test func equalTensors() {
        let a = COOTensor(
            shape: [3, 3],
            indices: [[0, 1], [1, 2]],
            values: [10, 20]
        )
        let b = COOTensor(
            shape: [3, 3],
            indices: [[0, 1], [1, 2]],
            values: [10, 20]
        )
        #expect(a == b)
    }

    @Test func differentValues() {
        let a = COOTensor(
            shape: [3, 3],
            indices: [[0, 1], [1, 2]],
            values: [10, 20]
        )
        let b = COOTensor(
            shape: [3, 3],
            indices: [[0, 1], [1, 2]],
            values: [10, 99]
        )
        #expect(a != b)
    }

    @Test func differentShapes() {
        let a = COOTensor(
            shape: [3, 3],
            indices: [[0], [1]],
            values: [10]
        )
        let b = COOTensor(
            shape: [4, 4],
            indices: [[0], [1]],
            values: [10]
        )
        #expect(a != b)
    }
}
