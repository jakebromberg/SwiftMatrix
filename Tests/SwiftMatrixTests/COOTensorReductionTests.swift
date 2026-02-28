import Testing
@testable import SwiftMatrix

struct COOTensorSumTests {
    @Test func basicSum() {
        let coo = COOTensor(
            shape: [3],
            indices: [[0, 1, 2]],
            values: [10, 20, 30]
        )
        #expect(coo.sum() == 60)
    }

    @Test func emptyTensor() {
        let coo = COOTensor<Int>(shape: [5])
        #expect(coo.sum() == 0)
    }

    @Test func rank3() {
        // 2x2x2 with 3 nonzero entries
        let coo = COOTensor(
            shape: [2, 2, 2],
            indices: [[0, 0, 1], [0, 1, 0], [1, 0, 1]],
            values: [1, 2, 3]
        )
        #expect(coo.sum() == 6)
    }

    @Test func matchesDense() {
        let coo = COOTensor(
            shape: [3, 4],
            indices: [[0, 1, 2, 2], [1, 2, 0, 3]],
            values: [10, 20, 30, 40]
        )
        #expect(coo.sum() == coo.toTensor().sum())
    }
}

struct COOTensorMeanTests {
    @Test func basicMean() {
        // shape [4], values at 0,1,2 -> sum=6.0, count=4 -> mean=1.5
        let coo = COOTensor(
            shape: [4],
            indices: [[0, 1, 2]],
            values: [1.0, 2.0, 3.0]
        )
        #expect(coo.mean() == 1.5)
    }

    @Test func emptyTensor() {
        let coo = COOTensor<Double>(shape: [4])
        #expect(coo.mean() == 0.0)
    }

    @Test func matchesDense() {
        let coo = COOTensor(
            shape: [2, 3],
            indices: [[0, 1, 1], [0, 1, 2]],
            values: [6.0, 3.0, 9.0]
        )
        #expect(coo.mean() == coo.toTensor().mean())
    }

    @Test func allNonzero() {
        // Every element is nonzero
        let coo = COOTensor(
            shape: [3],
            indices: [[0, 1, 2]],
            values: [2.0, 4.0, 6.0]
        )
        #expect(coo.mean() == 4.0)
    }
}

struct COOTensorDotProductTests {
    @Test func basicDot() {
        let a = COOTensor(
            shape: [4],
            indices: [[0, 1, 3]],
            values: [1, 2, 3]
        )
        let b = COOTensor(
            shape: [4],
            indices: [[0, 1, 2]],
            values: [10, 20, 30]
        )
        // overlap at indices 0,1: 1*10 + 2*20 = 50
        #expect(COOTensor.dot(a, b) == 50)
    }

    @Test func disjointIndices() {
        let a = COOTensor(
            shape: [4],
            indices: [[0, 1]],
            values: [1, 2]
        )
        let b = COOTensor(
            shape: [4],
            indices: [[2, 3]],
            values: [3, 4]
        )
        #expect(COOTensor.dot(a, b) == 0)
    }

    @Test func fullOverlap() {
        let a = COOTensor(
            shape: [3],
            indices: [[0, 1, 2]],
            values: [1, 2, 3]
        )
        let b = COOTensor(
            shape: [3],
            indices: [[0, 1, 2]],
            values: [4, 5, 6]
        )
        // 1*4 + 2*5 + 3*6 = 32
        #expect(COOTensor.dot(a, b) == 32)
    }

    @Test func partialOverlap() {
        // a has entries at 0,2,4; b has entries at 1,2,3
        let a = COOTensor(
            shape: [5],
            indices: [[0, 2, 4]],
            values: [10, 20, 30]
        )
        let b = COOTensor(
            shape: [5],
            indices: [[1, 2, 3]],
            values: [5, 6, 7]
        )
        // overlap at index 2: 20*6 = 120
        #expect(COOTensor.dot(a, b) == 120)
    }

    @Test func emptyOperands() {
        let a = COOTensor<Int>(shape: [5])
        let b = COOTensor(
            shape: [5],
            indices: [[0, 2]],
            values: [10, 20]
        )
        #expect(COOTensor.dot(a, b) == 0)
    }

    @Test func matchesDense() {
        let a = COOTensor(
            shape: [5],
            indices: [[0, 2, 4]],
            values: [1, 3, 5]
        )
        let b = COOTensor(
            shape: [5],
            indices: [[1, 2, 3]],
            values: [2, 4, 6]
        )
        #expect(COOTensor.dot(a, b) == Tensor.dot(a.toTensor(), b.toTensor()))
    }
}
