import Testing
@testable import SwiftMatrix

struct TensorIterationTests {
    @Test func iterationOrderRowMajor() {
        let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        let elements = Array(t)
        #expect(elements == [1, 2, 3, 4, 5, 6])
    }

    @Test func iterationRank3() {
        let t = Tensor(shape: [2, 2, 2], elements: Array(1...8))
        let elements = Array(t)
        #expect(elements == [1, 2, 3, 4, 5, 6, 7, 8])
    }

    @Test func iterationRank1() {
        let t = Tensor(shape: [4], elements: [10, 20, 30, 40])
        let elements = Array(t)
        #expect(elements == [10, 20, 30, 40])
    }

    @Test func linearSubscriptMatchesIteration() {
        let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        for i in 0..<t.count {
            #expect(t[linearIndex: i] == Array(t)[i])
        }
    }

    @Test func emptyTensorIteration() {
        let t = Tensor(shape: [0, 3], repeating: 0)
        #expect(Array(t).isEmpty)
    }

    @Test func collectionStartAndEndIndex() {
        let t = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        #expect(t.startIndex == 0)
        #expect(t.endIndex == 6)
    }

    @Test func collectionCount() {
        let t = Tensor(shape: [3, 4], repeating: 0)
        #expect(t.count == 12)
    }
}

struct TensorEquatableTests {
    @Test func equalTensors() {
        let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        let b = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        #expect(a == b)
    }

    @Test func differentElements() {
        let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        let b = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 7])
        #expect(a != b)
    }

    @Test func differentShapes() {
        let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        let b = Tensor(shape: [3, 2], elements: [1, 2, 3, 4, 5, 6])
        #expect(a != b)
    }
}

struct TensorHashableTests {
    @Test func equalTensorsHashEqual() {
        let a = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        let b = Tensor(shape: [2, 3], elements: [1, 2, 3, 4, 5, 6])
        #expect(a.hashValue == b.hashValue)
    }

    @Test func usableInSet() {
        let a = Tensor(shape: [2], elements: [1, 2])
        let b = Tensor(shape: [2], elements: [1, 2])
        let c = Tensor(shape: [2], elements: [3, 4])
        let set: Set = [a, b, c]
        #expect(set.count == 2)
    }
}
