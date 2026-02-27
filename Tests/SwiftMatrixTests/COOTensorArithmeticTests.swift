import Testing
@testable import SwiftMatrix

struct COOTensorAdditionTests {
    @Test func sameStructure() {
        let a = COOTensor(
            shape: [3, 3],
            indices: [[0, 1, 2], [1, 2, 0]],
            values: [10, 20, 30]
        )
        let b = COOTensor(
            shape: [3, 3],
            indices: [[0, 1, 2], [1, 2, 0]],
            values: [1, 2, 3]
        )
        let result = a + b
        #expect(result.shape == [3, 3])
        #expect(result.indices == [[0, 1, 2], [1, 2, 0]])
        #expect(result.values == [11, 22, 33])
    }

    @Test func differentStructure() {
        let a = COOTensor(
            shape: [3, 3],
            indices: [[0, 2], [0, 2]],
            values: [10, 30]
        )
        let b = COOTensor(
            shape: [3, 3],
            indices: [[1], [1]],
            values: [20]
        )
        let result = a + b
        #expect(result.shape == [3, 3])
        #expect(result.indices == [[0, 1, 2], [0, 1, 2]])
        #expect(result.values == [10, 20, 30])
    }

    @Test func mixedStructure() {
        // Some indices overlap, some don't
        let a = COOTensor(
            shape: [3, 3],
            indices: [[0, 1], [0, 1]],
            values: [10, 20]
        )
        let b = COOTensor(
            shape: [3, 3],
            indices: [[1, 2], [1, 2]],
            values: [5, 30]
        )
        let result = a + b
        #expect(result.shape == [3, 3])
        #expect(result.indices == [[0, 1, 2], [0, 1, 2]])
        #expect(result.values == [10, 25, 30])
    }

    @Test func emptyOperand() {
        let a = COOTensor(
            shape: [3, 3],
            indices: [[0, 1], [1, 2]],
            values: [10, 20]
        )
        let b = COOTensor<Int>(shape: [3, 3])
        let result = a + b
        #expect(result == a)
    }

    @Test func bothEmpty() {
        let a = COOTensor<Int>(shape: [3, 3])
        let b = COOTensor<Int>(shape: [3, 3])
        let result = a + b
        #expect(result.nnz == 0)
        #expect(result.shape == [3, 3])
    }
}

struct COOTensorSubtractionTests {
    @Test func sameStructure() {
        let a = COOTensor(
            shape: [3, 3],
            indices: [[0, 1, 2], [1, 2, 0]],
            values: [10, 20, 30]
        )
        let b = COOTensor(
            shape: [3, 3],
            indices: [[0, 1, 2], [1, 2, 0]],
            values: [1, 2, 3]
        )
        let result = a - b
        #expect(result.indices == [[0, 1, 2], [1, 2, 0]])
        #expect(result.values == [9, 18, 27])
    }

    @Test func cancellation() {
        let a = COOTensor(
            shape: [2, 2],
            indices: [[0, 1], [0, 1]],
            values: [5, 10]
        )
        // Subtracting same values produces stored zeros (no elimination)
        let result = a - a
        #expect(result.indices == [[0, 1], [0, 1]])
        #expect(result.values == [0, 0])
    }

    @Test func differentStructure() {
        let a = COOTensor(
            shape: [3, 3],
            indices: [[0], [0]],
            values: [10]
        )
        let b = COOTensor(
            shape: [3, 3],
            indices: [[2], [2]],
            values: [5]
        )
        let result = a - b
        #expect(result.indices == [[0, 2], [0, 2]])
        #expect(result.values == [10, -5])
    }
}

struct COOTensorMultiplicationTests {
    @Test func sameStructure() {
        let a = COOTensor(
            shape: [3, 3],
            indices: [[0, 1, 2], [1, 2, 0]],
            values: [10, 20, 30]
        )
        let b = COOTensor(
            shape: [3, 3],
            indices: [[0, 1, 2], [1, 2, 0]],
            values: [2, 3, 4]
        )
        let result = a * b
        #expect(result.indices == [[0, 1, 2], [1, 2, 0]])
        #expect(result.values == [20, 60, 120])
    }

    @Test func disjoint() {
        let a = COOTensor(
            shape: [3, 3],
            indices: [[0], [0]],
            values: [10]
        )
        let b = COOTensor(
            shape: [3, 3],
            indices: [[2], [2]],
            values: [20]
        )
        let result = a * b
        #expect(result.nnz == 0)
    }

    @Test func partialOverlap() {
        let a = COOTensor(
            shape: [3, 3],
            indices: [[0, 1], [0, 1]],
            values: [10, 20]
        )
        let b = COOTensor(
            shape: [3, 3],
            indices: [[1, 2], [1, 2]],
            values: [3, 4]
        )
        let result = a * b
        #expect(result.indices == [[1], [1]])
        #expect(result.values == [60])
    }
}

struct COOTensorScalarTests {
    @Test func multiply() {
        let a = COOTensor(
            shape: [2, 3],
            indices: [[0, 1], [1, 2]],
            values: [10, 20]
        )
        let result = a * 3
        #expect(result.indices == [[0, 1], [1, 2]])
        #expect(result.values == [30, 60])
    }

    @Test func multiplyCommutative() {
        let a = COOTensor(
            shape: [2, 3],
            indices: [[0, 1], [1, 2]],
            values: [10, 20]
        )
        #expect((a * 3).values == (3 * a).values)
    }

    @Test func divide() {
        let a = COOTensor(
            shape: [2, 3],
            indices: [[0, 1], [1, 2]],
            values: [10.0, 20.0]
        )
        let result = a / 2.0
        #expect(result.values == [5.0, 10.0])
    }
}

struct COOTensorNegationTests {
    @Test func negation() {
        let a = COOTensor(
            shape: [2, 2],
            indices: [[0, 1], [0, 1]],
            values: [10, -20]
        )
        let result = -a
        #expect(result.indices == [[0, 1], [0, 1]])
        #expect(result.values == [-10, 20])
    }

    @Test func negationEmpty() {
        let a = COOTensor<Int>(shape: [3, 3])
        let result = -a
        #expect(result.nnz == 0)
    }
}

struct COOTensorCompoundAssignmentTests {
    @Test func plusEquals() {
        var a = COOTensor(
            shape: [2, 2],
            indices: [[0, 1], [0, 1]],
            values: [10, 20]
        )
        let b = COOTensor(
            shape: [2, 2],
            indices: [[0, 1], [0, 1]],
            values: [1, 2]
        )
        a += b
        #expect(a.values == [11, 22])
    }

    @Test func minusEquals() {
        var a = COOTensor(
            shape: [2, 2],
            indices: [[0, 1], [0, 1]],
            values: [10, 20]
        )
        let b = COOTensor(
            shape: [2, 2],
            indices: [[0, 1], [0, 1]],
            values: [1, 2]
        )
        a -= b
        #expect(a.values == [9, 18])
    }

    @Test func timesEquals() {
        var a = COOTensor(
            shape: [2, 2],
            indices: [[0, 1], [0, 1]],
            values: [10, 20]
        )
        let b = COOTensor(
            shape: [2, 2],
            indices: [[0, 1], [0, 1]],
            values: [2, 3]
        )
        a *= b
        #expect(a.values == [20, 60])
    }
}

struct COOTensorArithmeticRankTests {
    @Test func rank1Addition() {
        let a = COOTensor(
            shape: [5],
            indices: [[1, 3]],
            values: [10, 20]
        )
        let b = COOTensor(
            shape: [5],
            indices: [[0, 3]],
            values: [5, 15]
        )
        let result = a + b
        #expect(result.indices == [[0, 1, 3]])
        #expect(result.values == [5, 10, 35])
    }

    @Test func rank3Multiplication() {
        let a = COOTensor(
            shape: [2, 3, 4],
            indices: [[0, 1], [2, 0], [3, 1]],
            values: [10, 20]
        )
        let b = COOTensor(
            shape: [2, 3, 4],
            indices: [[0, 1], [2, 1], [3, 0]],
            values: [3, 4]
        )
        // Only (0,2,3) matches
        let result = a * b
        #expect(result.nnz == 1)
        #expect(result.indices == [[0], [2], [3]])
        #expect(result.values == [30])
    }
}
