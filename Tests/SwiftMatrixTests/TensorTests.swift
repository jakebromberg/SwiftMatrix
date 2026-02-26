import Testing
@testable import SwiftMatrix

struct TensorCreationTests {
    @Test func createFromShapeAndRepeatingValue() {
        let t = Tensor(shape: [2, 3], repeating: 0.0)
        #expect(t.shape == [2, 3])
        #expect(t.rank == 2)
        #expect(t.count == 6)
        #expect(t.strides == [3, 1])
    }

    @Test func createFromFlatArrayAndShape() {
        let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        #expect(t.shape == [2, 3])
        #expect(t[0, 0] == 1)
        #expect(t[0, 2] == 3)
        #expect(t[1, 0] == 4)
        #expect(t[1, 2] == 6)
    }

    @Test func createFromNestedArrays() {
        let t = Tensor([[1, 2, 3], [4, 5, 6]])
        #expect(t.shape == [2, 3])
        #expect(t.rank == 2)
        #expect(t[0, 0] == 1)
        #expect(t[0, 2] == 3)
        #expect(t[1, 0] == 4)
        #expect(t[1, 2] == 6)
    }

    @Test func createScalar() {
        let t = Tensor(shape: [], elements: [42])
        #expect(t.shape == [])
        #expect(t.rank == 0)
        #expect(t.count == 1)
        #expect(t.strides == [])
    }

    @Test func createRank1() {
        let t = Tensor(shape: [4], elements: [10, 20, 30, 40])
        #expect(t.shape == [4])
        #expect(t.rank == 1)
        #expect(t.count == 4)
        #expect(t.strides == [1])
        #expect(t[2] == 30)
    }
}

struct TensorStridesTests {
    @Test(arguments: [
        (shape: [Int](),     expected: [Int]()),
        (shape: [4],         expected: [1]),
        (shape: [2, 3],      expected: [3, 1]),
        (shape: [2, 3, 4],   expected: [12, 4, 1]),
        (shape: [5, 4, 3, 2], expected: [24, 6, 2, 1]),
    ])
    func rowMajorStrides(shape: [Int], expected: [Int]) {
        let t = Tensor(shape: shape, repeating: 0)
        #expect(t.strides == expected)
    }
}

struct TensorSubscriptTests {
    @Test func getAndSetRank2() {
        var t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        #expect(t[1, 1] == 5)
        t[1, 1] = 99
        #expect(t[1, 1] == 99)
    }

    @Test func getAndSetRank3() {
        var t = Tensor(shape: [2, 3, 4], repeating: 0)
        t[1, 2, 3] = 42
        #expect(t[1, 2, 3] == 42)
        // Verify other elements remain 0
        #expect(t[0, 0, 0] == 0)
        #expect(t[1, 2, 2] == 0)
    }

    @Test func getAndSetRank1() {
        var t = Tensor(shape: [3], elements: [10, 20, 30])
        #expect(t[1] == 20)
        t[1] = 99
        #expect(t[1] == 99)
    }

    @Test func arraySubscriptGet() {
        let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        #expect(t[[1, 2]] == 6)
    }

    @Test func arraySubscriptSet() {
        var t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        t[[0, 1]] = 99
        #expect(t[[0, 1]] == 99)
    }
}

struct TensorEdgeCaseTests {
    @Test func emptyDimension() {
        let t = Tensor(shape: [0, 3], repeating: 0)
        #expect(t.count == 0)
        #expect(Array(t).isEmpty)
    }

    @Test func singleElement() {
        let t = Tensor(shape: [1, 1], repeating: 42)
        #expect(t.count == 1)
        #expect(t[0, 0] == 42)
    }

    @Test func highRank() {
        let t = Tensor(shape: [2, 2, 2, 2], repeating: 1)
        #expect(t.count == 16)
        #expect(t.rank == 4)
        #expect(t.strides == [8, 4, 2, 1])
    }
}

struct TensorContiguityTests {
    @Test func standardTensorIsContiguous() {
        let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        #expect(t.isContiguous)
    }

    @Test func scalarIsContiguous() {
        let t = Tensor(shape: [], elements: [42])
        #expect(t.isContiguous)
    }

    @Test func nonContiguousViewReportsNonContiguous() {
        // A 2x3 tensor with transposed strides [1, 2] instead of row-major [3, 1]
        let t = Tensor(
            storage: [1, 2, 3, 4, 5, 6],
            shape: [3, 2],
            strides: [1, 3],
            offset: 0,
            isContiguous: false
        )
        #expect(!t.isContiguous)
    }
}

struct TensorNonContiguousAccessTests {
    @Test func linearIndexOnNonContiguousTensor() {
        // Simulates a transpose of a 2x3 tensor:
        // Original (row-major): [[1, 2, 3], [4, 5, 6]]
        // Transposed shape: [3, 2], strides: [1, 3]
        // Logical iteration order: (0,0)=1, (0,1)=4, (1,0)=2, (1,1)=5, (2,0)=3, (2,1)=6
        let t = Tensor(
            storage: [1, 2, 3, 4, 5, 6],
            shape: [3, 2],
            strides: [1, 3],
            offset: 0,
            isContiguous: false
        )
        #expect(Array(t) == [1, 4, 2, 5, 3, 6])
    }

    @Test func linearIndexWithOffset() {
        // A view into a larger storage with offset
        let t = Tensor(
            storage: [0, 0, 1, 2, 3, 4, 5, 6],
            shape: [2, 3],
            strides: [3, 1],
            offset: 2,
            isContiguous: false
        )
        #expect(Array(t) == [1, 2, 3, 4, 5, 6])
    }
}
