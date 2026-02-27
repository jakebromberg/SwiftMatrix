import Testing
@testable import SwiftMatrix

struct TensorBroadcastShapeTests {
    @Test(arguments: [
        (a: [3, 1], b: [1, 4], expected: [3, 4]),
        (a: [3], b: [2, 3], expected: [2, 3]),
        (a: [1], b: [5], expected: [5]),
        (a: [2, 1, 3], b: [4, 3], expected: [2, 4, 3]),
        (a: [5], b: [5], expected: [5]),
        (a: [1, 1], b: [3, 4], expected: [3, 4]),
    ])
    func compatibleShapes(a: [Int], b: [Int], expected: [Int]) {
        let result = Tensor<Int>.broadcastShape(a, b)
        #expect(result == expected)
    }

    @Test(arguments: [
        (a: [3], b: [4]),
        (a: [2, 3], b: [2, 4]),
        (a: [3, 2], b: [3, 3]),
    ])
    func incompatibleShapes(a: [Int], b: [Int]) {
        let result = Tensor<Int>.broadcastShape(a, b)
        #expect(result == nil)
    }
}

struct TensorBroadcastViewTests {
    @Test func broadcastToLargerShape() {
        let t = Tensor(shape: [3, 1], elements: [1, 2, 3])
        let b = t.broadcast(to: [3, 4])
        #expect(b.shape == [3, 4])
        // Row 0: [1,1,1,1], Row 1: [2,2,2,2], Row 2: [3,3,3,3]
        #expect(Array(b) == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    }

    @Test func broadcastAddsLeadingDimension() {
        let t = Tensor(shape: [3], elements: [10, 20, 30])
        let b = t.broadcast(to: [2, 3])
        #expect(b.shape == [2, 3])
        #expect(Array(b) == [10, 20, 30, 10, 20, 30])
    }

    @Test func broadcastIdentity() {
        let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        let b = t.broadcast(to: [2, 3])
        #expect(Array(b) == Array(t))
    }
}

struct TensorBroadcastArithmeticTests {
    @Test func addColumnToMatrix() {
        // [3,1] + [1,4] -> [3,4]
        let a = Tensor(shape: [3, 1], elements: [1, 2, 3])
        let b = Tensor(shape: [1, 4], elements: [10, 20, 30, 40])
        let result = a + b
        #expect(result.shape == [3, 4])
        #expect(result == Tensor(shape: [3, 4], elements: [
            11, 21, 31, 41,
            12, 22, 32, 42,
            13, 23, 33, 43,
        ]))
    }

    @Test func addVectorToMatrix() {
        // [3] + [2,3] -> [2,3]
        let a = Tensor(shape: [3], elements: [1, 2, 3])
        let b = Tensor([[10, 20, 30], [40, 50, 60]])
        let result = a + b
        #expect(result.shape == [2, 3])
        #expect(result == Tensor([[11, 22, 33], [41, 52, 63]]))
    }

    @Test func mulWithBroadcasting() {
        let a = Tensor(shape: [2, 1], elements: [2, 3])
        let b = Tensor(shape: [1, 3], elements: [10, 20, 30])
        let result = a * b
        #expect(result.shape == [2, 3])
        #expect(result == Tensor([[20, 40, 60], [30, 60, 90]]))
    }

    @Test func subWithBroadcasting() {
        let a = Tensor(shape: [2, 3], elements: [10, 20, 30, 40, 50, 60])
        let b = Tensor(shape: [1, 3], elements: [1, 2, 3])
        let result = a - b
        #expect(result.shape == [2, 3])
        #expect(result == Tensor([[9, 18, 27], [39, 48, 57]]))
    }

    @Test func divWithBroadcasting() {
        let a = Tensor(shape: [2, 3], elements: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        let b = Tensor(shape: [1, 3], elements: [2.0, 5.0, 10.0])
        let result = a / b
        #expect(result.shape == [2, 3])
        #expect(result == Tensor([[5.0, 4.0, 3.0], [20.0, 10.0, 6.0]]))
    }
}

struct TensorBroadcastEdgeCaseTests {
    @Test func broadcastViewEquatable() {
        let a = Tensor(shape: [1, 3], elements: [1, 2, 3])
        let broadcasted = a.broadcast(to: [2, 3])
        let explicit = Tensor([[1, 2, 3], [1, 2, 3]])
        #expect(broadcasted == explicit)
    }

    @Test func broadcastViewHashable() {
        let a = Tensor(shape: [1, 3], elements: [1, 2, 3])
        let broadcasted = a.broadcast(to: [2, 3])
        let explicit = Tensor([[1, 2, 3], [1, 2, 3]])
        #expect(broadcasted.hashValue == explicit.hashValue)
    }

    @Test func nonContiguousPlusBroadcast() {
        // Transposed tensor [3,2] + broadcast [1,2]
        let a = Tensor([[1, 2, 3], [4, 5, 6]]).transposed() // shape [3,2]
        let b = Tensor(shape: [1, 2], elements: [10, 100])
        let result = a + b
        #expect(result.shape == [3, 2])
        // a iterates as [1,4,2,5,3,6]
        #expect(Array(result) == [11, 104, 12, 105, 13, 106])
    }

    @Test func sameShapeStillWorks() {
        let a = Tensor(shape: [3], elements: [1, 2, 3])
        let b = Tensor(shape: [3], elements: [4, 5, 6])
        #expect(a + b == Tensor(shape: [3], elements: [5, 7, 9]))
    }
}
