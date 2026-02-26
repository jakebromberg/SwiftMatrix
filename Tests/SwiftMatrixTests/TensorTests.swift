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
